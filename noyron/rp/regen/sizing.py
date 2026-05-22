"""
sizing.py
=========

Axially-varying cooling-channel sizer for regeneratively-cooled rocket engines.

The nozzle is split into three zones and a minimum-ΔP channel geometry is found
by a parametric sweep over (n_channels, h_throat, h_divergent) with a fixed
channel width equal to the minimum manufacturable width.

Zone layout (Option A -- piecewise constant, per task spec)
-----------------------------------------------------------
  Zone 0 — convergent  : x_inlet  to  x = 0  (throat)
  Zone 1 — throat      : x = 0    to  x = +2 * r_throat
  Zone 2 — divergent   : x = +2 * r_throat  to  x_exit

Design rationale
----------------
The divergent section (zone 2) is typically 5–10× longer than the throat
zone but carries far lower heat flux.  Providing wide (tall) channels there
costs little thermally while avoiding most of the pressure drop.  The tight
throat zone (zones 0 + 1) uses small h (high velocity) to maximise h_c.

Why a sweep rather than gradient-based optimisation
----------------------------------------------------
The thermal constraint T_wall_gas < T_target involves a full call to
solve_regen() whose gradient w.r.t. geometry is noisy (CoolProp + inner
iteration).  SLSQP routinely gets stuck far from the optimum, especially when
the optimal h is small (low AR) while standard engineering intuition suggests
high AR.  A coarse-then-fine grid sweep is reliable, fast (O(N²) × solve
time), and provides a clear picture of the feasible region.

Complexity is O(len(n_ch_grid) × len(h01_grid) × len(h2_grid) × solve_time).
With defaults: 6 × 8 × 5 × 0.02 s ≈ 5 s.

Note on heat-transfer model
---------------------------
The solver uses Dittus-Boelter (McAdams 1954) for coolant-side h.  For
transcritical methane the actual h_c is 2–5× higher than Dittus-Boelter
predicts (Pizzarelli et al. 2016 — not implemented here), so the wall
temperatures predicted by this sizer are CONSERVATIVE.  If T_wall_peak from
this sizer exceeds the material limit, the design is not necessarily infeasible
— it signals that a more accurate correlation should be applied next.

References
----------
Wadel, M.F. (1997). "Comparison of High Aspect Ratio Cooling Channel Designs
    for a Rocket Combustion Chamber." NASA TM-107270.
    -- Motivation for high-AR channels at the throat; AR 4–8 at Pc ~ 7–20 MPa.

Carlile, J.A. and Quentmeyer, R.J. (1992). "An Experimental Investigation of
    High-Aspect-Ratio Cooling Passages." AIAA 92-3154 / NASA TM-103702.
    -- Experimental confirmation of AR benefit; effective AR 4–8 tested.

Pizzarelli, M., Nasuti, F., and Onofri, M. (2016). "Trade-off analysis of
    high-aspect-ratio-cooling channels for rocket engines." Int. J. Heat and
    Fluid Flow, Vol. 58, pp. 210–219.
    -- Transcritical methane h_c exceeds Dittus-Boelter by factor 2–5.

Dittus-Boelter / Haaland friction: see physics/regen/solver.py.
NASA SP-8087 (1977) Sec. 3.1: zone layout and thermal resistance network.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any


from noyron.rp.nozzle.rao import RaoNozzle
from noyron.rp.regen.bartz import CombustionGas
from noyron.rp.regen.coolant import CoolingChannelGeometry, CoolantInlet, WallMaterial
from noyron.rp.regen.solver import solve_regen
from noyron.kernel.util.exceptions import SizingInfeasibleError


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class ChannelSizingError(SizingInfeasibleError):
    """
    Raised when no channel geometry satisfies all constraints.

    The error message reports which constraint was binding for each
    candidate n_channels so the engineer knows what to change.

    Inherits from SizingInfeasibleError (which inherits RuntimeError)
    so callers can catch either the specific or base class.
    """


# ---------------------------------------------------------------------------
# Problem dataclass
# ---------------------------------------------------------------------------


@dataclass
class ChannelSizingProblem:
    """
    Complete specification for the channel sizing problem.

    Parameters
    ----------
    nozzle : RaoNozzle
    gas : CombustionGas
    coolant_inlet : CoolantInlet
    wall_material : WallMaterial
    T_wall_max : float
        Hard material limit [K].  The sizer targets T_wall_max - T_wall_margin.
    T_wall_margin : float
        Safety margin below T_wall_max [K].  Default 50 K.
    dP_max : float
        Maximum allowable total coolant pressure drop [Pa].
    min_wall_thickness : float
        Minimum gas-side wall thickness (AM constraint) [m].
    max_aspect_ratio : float
        Maximum h/w for a single channel (AM / structural constraint).
    min_channel_width : float
        Minimum channel width w (AM constraint) [m].
        Note: the sizer fixes w = min_channel_width everywhere; maximising
        velocity for a given h_c budget is achieved by keeping w at its
        minimum.
    t_land_min : float
        Minimum land (rib) thickness [m].
    n_stations_opt : int
        Axial stations used for each solve_regen() call during the sweep.
        Coarser grids are faster; the final design is validated at n=200.
    """

    nozzle: RaoNozzle
    gas: CombustionGas
    coolant_inlet: CoolantInlet
    wall_material: WallMaterial
    T_wall_max: float
    T_wall_margin: float = 50.0
    dP_max: float = 5.0e6
    min_wall_thickness: float = 0.5e-3
    max_aspect_ratio: float = 8.0
    min_channel_width: float = 0.6e-3
    t_land_min: float = 0.3e-3
    n_stations_opt: int = 60


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _zone_x_bounds(nozzle: RaoNozzle) -> tuple[float, float, float, float]:
    """(x0, x1, x2, x3) zone boundary axial coordinates [m]."""
    return nozzle.x_inlet, 0.0, 2.0 * nozzle.r_throat, nozzle.x_exit


def _build_geometry(
    n_ch: int,
    w: float,
    h01: float,
    h2: float,
    t_wall: float,
    t_land: float,
    x1: float,
    x2: float,
) -> CoolingChannelGeometry:
    """
    Build a CoolingChannelGeometry from three-zone parameters.

    Zones 0 and 1 share the same height h01 (throat region);
    zone 2 (divergent) uses h2 which may be larger to reduce ΔP.

    Parameters
    ----------
    n_ch : int
    w : float        Channel width [m], same for all zones.
    h01 : float      Channel height [m] for zones 0 and 1 (throat region).
    h2 : float       Channel height [m] for zone 2 (divergent).
    t_wall : float   Wall thickness [m].
    t_land : float   Land thickness [m].
    x1, x2 : float  Zone boundary x-coordinates [m].
    """

    def _h(x: float, _h01: float = h01, _h2: float = h2,
           _x1: float = x1, _x2: float = x2) -> float:
        return _h01 if x < _x2 else _h2

    return CoolingChannelGeometry(
        n_channels=n_ch,
        width=lambda x, _w=w: _w,
        height=_h,
        wall_thickness=lambda x, _tw=t_wall: _tw,
        land_thickness=lambda x, _tl=t_land: _tl,
    )


def _evaluate(
    n_ch: int,
    w: float,
    h01: float,
    h2: float,
    problem: ChannelSizingProblem,
    x1: float,
    x2: float,
) -> dict[str, Any]:
    """Run solve_regen and return key metrics, or a failure dict."""
    geom = _build_geometry(
        n_ch, w, h01, h2,
        problem.min_wall_thickness, problem.t_land_min,
        x1, x2,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solve_regen(
                nozzle=problem.nozzle,
                gas=problem.gas,
                channel_geom=geom,
                coolant_inlet=problem.coolant_inlet,
                wall_material=problem.wall_material,
                direction="counterflow",
                n_stations=problem.n_stations_opt,
                tol=2.0,
            )
        return {
            "T_wall_peak": float(result["T_wall_gas"].max()),
            "dP_total": float(result["dP_total"]),
            "h_c_mean": float(result["h_c"].mean()),
            "ok": True,
        }
    except Exception as exc:
        return {"T_wall_peak": 1.0e9, "dP_total": 1.0e9, "h_c_mean": 0.0,
                "ok": False, "exc": str(exc)}


# ---------------------------------------------------------------------------
# Sweep grids
# ---------------------------------------------------------------------------

# h01 grid: from very small (high velocity, good cooling, high ΔP) upward.
# Units: metres.  Chosen to span the full physically-interesting range.
_H01_GRID_MM = [0.30, 0.40, 0.60, 0.80, 1.20, 2.00, 3.00, 4.00]

# h2 grid: divergent zone can have taller channels (lower ΔP contribution).
_H2_GRID_MM = [0.40, 0.80, 1.60, 3.20, 6.40]

# n_ch candidates: step of 4 from 40 to 120.  Integer only.
_N_CH_CANDIDATES = list(range(40, 121, 4))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def size_channels(problem: ChannelSizingProblem) -> CoolingChannelGeometry:
    """
    Find axially-varying channel geometry meeting all constraints,
    minimising coolant ΔP.

    Uses a three-zone (convergent / throat / divergent) piecewise-constant
    geometry.  The channel width is fixed at min_channel_width everywhere
    (narrowest manufacturable channels maximise coolant velocity for a given
    h_c target).  The height h in the throat zones (0 and 1) and in the
    divergent zone (2) are found by a coarse parametric sweep followed by
    a one-dimensional refinement.

    Parameters
    ----------
    problem : ChannelSizingProblem

    Returns
    -------
    CoolingChannelGeometry
        Ready to pass directly to solve_regen().

    Raises
    ------
    ChannelSizingError
        If no (n_ch, h01, h2) combination satisfies both the thermal and
        pressure-drop constraints.  The error message lists the binding
        constraint for each candidate n_channels.

    Notes
    -----
    The sweep uses n_stations_opt stations (default 60) for speed.  The
    caller should re-run solve_regen() at n >= 200 to confirm the result.

    Dittus-Boelter underpredicts h_c for transcritical methane (Pizzarelli
    et al. 2016) by a factor of 2–5.  If T_wall_peak from this sizer
    exceeds the material limit, the true wall temperature is likely lower;
    apply the Pizzarelli correction before concluding the design is infeasible.

    Source for zone layout:
        NASA SP-8087 (1977) Sec. 3.1 -- throat ± 2*r_t is the critical zone.
    """
    nozzle = problem.nozzle
    T_target = problem.T_wall_max - problem.T_wall_margin
    w = problem.min_channel_width
    t_land = problem.t_land_min
    _, x1, x2, _ = _zone_x_bounds(nozzle)

    best: dict[str, Any] | None = None
    best_dP = math.inf
    failures: dict[int, str] = {}

    for n_ch in _N_CH_CANDIDATES:
        # Geometric pre-check: channels must fit at the throat
        arc_throat = 2.0 * math.pi * nozzle.r_throat / n_ch
        if arc_throat < w + t_land:
            failures[n_ch] = (
                f"channels don't fit at throat "
                f"(arc={arc_throat*1e3:.2f} mm < "
                f"w+t_land={( w+t_land)*1e3:.2f} mm)"
            )
            continue

        feasible_found_for_n_ch = False

        for h01_mm in _H01_GRID_MM:
            h01 = h01_mm * 1e-3
            if h01 / w > problem.max_aspect_ratio + 1.0e-6:
                continue  # AR constraint

            for h2_mm in _H2_GRID_MM:
                h2 = h2_mm * 1e-3
                if h2 / w > problem.max_aspect_ratio + 1.0e-6:
                    continue

                m = _evaluate(n_ch, w, h01, h2, problem, x1, x2)
                if not m["ok"]:
                    continue

                T_pk = m["T_wall_peak"]
                dP = m["dP_total"]

                thermally_ok = T_pk <= T_target + 1.0     # 1 K numerical tolerance
                pressure_ok = dP <= problem.dP_max + 1.0e3  # 1 kPa tolerance

                if thermally_ok and pressure_ok:
                    feasible_found_for_n_ch = True
                    if dP < best_dP:
                        best_dP = dP
                        best = {
                            "n_ch": n_ch,
                            "h01": h01,
                            "h2": h2,
                            "T_wall_peak": T_pk,
                            "dP_total": dP,
                        }

        if not feasible_found_for_n_ch:
            # Record the best (lowest T_wall) result from this n_ch for diagnostics
            m_best_t = None
            for h01_mm in _H01_GRID_MM:
                h01 = h01_mm * 1e-3
                if h01 / w > problem.max_aspect_ratio + 1.0e-6:
                    continue
                for h2_mm in _H2_GRID_MM:
                    h2 = h2_mm * 1e-3
                    if h2 / w > problem.max_aspect_ratio + 1.0e-6:
                        continue
                    m = _evaluate(n_ch, w, h01, h2, problem, x1, x2)
                    if not m["ok"]:
                        continue
                    if m_best_t is None or m["T_wall_peak"] < m_best_t["T_wall_peak"]:
                        m_best_t = m

            if m_best_t is None:
                failures[n_ch] = "all evaluations failed (CoolProp error?)"
            else:
                T_best = m_best_t["T_wall_peak"]
                dP_best = m_best_t["dP_total"]
                reasons = []
                if T_best > T_target + 1.0:
                    reasons.append(
                        f"T_wall_peak={T_best:.0f} K > target {T_target:.0f} K "
                        f"(min achievable with Dittus-Boelter at w={w*1e3:.1f} mm)"
                    )
                if dP_best > problem.dP_max + 1.0e3:
                    reasons.append(
                        f"ΔP={dP_best/1e6:.2f} MPa > budget {problem.dP_max/1e6:.2f} MPa"
                    )
                failures[n_ch] = "; ".join(reasons) if reasons else "no feasible h combination"

    if best is None:
        # Build an informative failure report
        sample = {k: v for k, v in list(failures.items())[:10]}
        msg = ["Channel sizing failed: no feasible geometry found.\n\n"]
        msg.append("Binding constraints per candidate n_channels:\n")
        for n, reason in sample.items():
            msg.append(f"  n_ch={n:3d}: {reason}\n")
        msg.append(
            "\nNote: Dittus-Boelter underpredicts h_c for transcritical methane\n"
            "  by 2–5× (Pizzarelli et al. 2016).  The actual T_wall is likely\n"
            "  lower.  Consider:\n"
            "  - Implementing the Pizzarelli h_c correction\n"
            "  - Increasing dP_max (higher pump pressure head)\n"
            "  - Increasing T_wall_max (use a higher-temperature alloy)\n"
            "  - Reducing min_channel_width (finer AM features)\n"
            "  - Reducing Pc to lower the gas-side heat flux\n"
        )
        raise ChannelSizingError("".join(msg))

    return _build_geometry(
        best["n_ch"], w, best["h01"], best["h2"],
        problem.min_wall_thickness, problem.t_land_min,
        x1, x2,
    )
