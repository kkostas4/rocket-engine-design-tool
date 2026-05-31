"""
solver.py
=========

Domain-agnostic 1D coupled heat exchanger solver.

Algorithm
---------
Discretise the axial domain [x_start, x_end] into N stations.  At each
station (marched in the coolant flow direction):

  1. Evaluate T_recovery(x) from the HotSideModel (precomputed; independent
     of T_wall for most hot-side models, including Bartz).
  2. Evaluate cold-side h_c from the ColdSideModel (Dittus-Boelter or other).
  3. Fixed-point iteration on T_wall_gas:
     a. hot_state <- hot_model.evaluate(x, geometry, T_wall_gas)  [sigma included]
     b. k_wall   <- geometry.wall_conductivity(x, T_wall_gas)
     c. eta_fin  <- geometry.fin_efficiency(x, h_c, k_wall)
     d. eta_0    <- geometry.overall_surface_efficiency(x, eta_fin)
     e. q_prime = (T_recovery - T_coolant) / (R_g + R_wall + R_c_eff)  [W/m/channel]
     f. T_wall_gas_new = T_recovery - q_prime * R_g
     g. Repeat until |T_wall_gas_new - T_wall_gas| < tol.
  4. Update coolant: dT = q_prime * dx / (m_dot_ch * cp).
  5. Update pressure: dP = f_Haaland * rho * V^2 / (2*D_h) * dx.

No rocket-specific imports.  The only physics knowledge here is the generic
thermal resistance network and the Darcy-Weisbach pressure drop formula.

Sources
-------
Thermal resistance network:
    NASA SP-8087 (1977). "Liquid Rocket Engine Fluid-Cooled Combustion
    Chambers." Sec. 3.1.  (Notation and resistance decomposition.)
    Incropera et al. (2011). "Fundamentals of Heat and Mass Transfer,"
    7th ed., Wiley. Ch. 3.6 (composite wall resistance).

Darcy-Weisbach pressure drop:
    White, F.M. (2011). "Fluid Mechanics," 7th ed., McGraw-Hill. Ch. 6.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from noyron.kernel.heat_exchanger.base import (
    ColdSideModel,
    FluidProperties,
    HeatExchangerGeometry,
    HotSideModel,
)
from noyron.kernel.heat_exchanger.correlations import haaland_f

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from physics.regen.coolant import CoolantInlet

try:
    from CoolProp.CoolProp import PropsSI  # type: ignore[import-untyped]
    _COOLPROP_AVAILABLE = True
except ImportError:
    _COOLPROP_AVAILABLE = False


def _coolant_props(T: float, P: float, fluid: str) -> FluidProperties:
    """
    Return FluidProperties for the coolant at (T, P) via CoolProp.

    Raises RuntimeError if CoolProp is not installed.

    Source: CoolProp library; Bell et al. (2014). Ind. Eng. Chem. Res.
    """
    if not _COOLPROP_AVAILABLE:
        raise RuntimeError(
            "CoolProp is required for the heat exchanger solver.  "
            "Install with: pip install CoolProp"
        )
    return FluidProperties(
        rho=PropsSI("D", "T", T, "P", P, fluid),
        cp=PropsSI("C", "T", T, "P", P, fluid),
        mu=PropsSI("V", "T", T, "P", P, fluid),
        k=PropsSI("L", "T", T, "P", P, fluid),
        Pr=PropsSI("Prandtl", "T", T, "P", P, fluid),
    )


def solve_heat_exchanger(
    *,
    hot_model: HotSideModel,
    cold_model: ColdSideModel,
    geometry: HeatExchangerGeometry,
    x_start: float,
    x_end: float,
    coolant_inlet: CoolantInlet,
    direction: str = "counterflow",
    n_stations: int = 200,
    tol: float = 1.0,
) -> dict[str, Any]:
    """
    1D coupled heat exchanger solver with generic hot/cold models.

    Marches along the heat exchanger axis in the coolant flow direction.  At
    each station the wall temperature is found by fixed-point iteration
    coupling the hot-side model to the cold-side model through a three-
    resistance thermal network: R_gas + R_wall + R_coolant_effective.

    Parameters
    ----------
    hot_model : HotSideModel
        Provides T_recovery(x) and h_hot(x, T_wall).
    cold_model : ColdSideModel
        Provides h_cold(x, fluid_props, m_dot_ch, geometry).
    geometry : HeatExchangerGeometry
        Provides areas, wall thickness, fin efficiency, wall conductivity.
    x_start : float
        Upstream boundary of the heat exchanger domain [m].
    x_end : float
        Downstream boundary of the heat exchanger domain [m].
    coolant_inlet : CoolantInlet
        Coolant entry conditions (T, P, m_dot, fluid name).
    direction : str
        'counterflow' (coolant flows from x_end to x_start; default) or
        'coflow' (coolant flows from x_start to x_end).
    n_stations : int
        Number of axial discretisation stations.
    tol : float
        Convergence tolerance on T_wall_gas [K] for the inner iteration.

    Returns
    -------
    dict with keys:
        x               ndarray [m]      -- axial stations (x_start to x_end)
        T_wall_gas      ndarray [K]      -- gas-side wall surface temperature
        T_wall_coolant  ndarray [K]      -- coolant-side wall surface temperature
        T_coolant       ndarray [K]      -- coolant bulk temperature
        P_coolant       ndarray [Pa]     -- coolant pressure
        q_dot           ndarray [W/m2]   -- heat flux at hot-side wall
        h_g             ndarray [W/m2.K] -- hot-side h
        h_c             ndarray [W/m2.K] -- cold-side h
        dP_total        float [Pa]       -- total coolant pressure drop
    """
    if direction not in ("counterflow", "coflow"):
        raise ValueError(
            f"direction must be 'counterflow' or 'coflow', got {direction!r}"
        )

    x_arr = np.linspace(x_start, x_end, n_stations)
    dx = float(x_arr[1] - x_arr[0])

    T_wg = np.zeros(n_stations)
    T_wc = np.zeros(n_stations)
    T_c = np.zeros(n_stations)
    P_c = np.zeros(n_stations)
    q_arr = np.zeros(n_stations)
    h_g_arr = np.zeros(n_stations)
    h_c_arr = np.zeros(n_stations)

    m_dot_ch = coolant_inlet.m_dot / geometry.n_channels

    if direction == "counterflow":
        T_c[-1] = coolant_inlet.T
        P_c[-1] = coolant_inlet.P
        march_idx = range(n_stations - 1, -1, -1)

        def next_station(i: int) -> int:
            return i - 1

    else:
        T_c[0] = coolant_inlet.T
        P_c[0] = coolant_inlet.P
        march_idx = range(n_stations)

        def next_station(i: int) -> int:
            return i + 1

    # Precompute T_recovery (independent of T_wall for most hot-side models,
    # including Bartz).  Use a representative guess for T_wall; if T_recovery
    # depends on T_wall the inner loop will recompute it anyway.
    _T_wall_ref = 300.0
    T_recovery_arr = np.array(
        [hot_model.evaluate(float(x), geometry, _T_wall_ref).T_recovery for x in x_arr]
    )

    T_wg_guess = 0.30 * float(T_recovery_arr.max())  # warm start: ~30% of T_hot
    peak_T_wg = 0.0

    for i in march_idx:
        x = float(x_arr[i])
        T_cool = float(T_c[i])
        P_cool = float(P_c[i])
        T_recovery = float(T_recovery_arr[i])

        # Coolant properties
        try:
            props = _coolant_props(T_cool, P_cool, coolant_inlet.fluid)
        except Exception as exc:
            warnings.warn(
                f"CoolProp failed at x={x:.4f} m, T={T_cool:.1f} K, "
                f"P={P_cool:.0f} Pa: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            props = _coolant_props(coolant_inlet.T, coolant_inlet.P, coolant_inlet.fluid)

        # Cold-side h (independent of T_wall for Dittus-Boelter)
        cold = cold_model.evaluate(x, props, m_dot_ch, geometry)
        h_c = cold.h_convective

        # Inner fixed-point iteration: converge T_wall_gas
        T_wg_i = T_wg_guess
        T_wg_new = T_wg_i
        for _iter in range(60):
            hot = hot_model.evaluate(x, geometry, T_wg_i)
            h_g = hot.h_convective
            # T_recovery may be T_wall-dependent; use the returned value.
            T_aw = hot.T_recovery

            k_w = geometry.wall_conductivity(x, T_wg_i)
            eta_fin = geometry.fin_efficiency(x, h_c, k_wall=k_w)
            eta_0 = geometry.overall_surface_efficiency(x, eta_fin)

            A_hot = geometry.wetted_perimeter_hot(x)   # per-channel, per unit length [m]
            A_cool = geometry.wetted_perimeter_cold(x)  # per-channel, per unit length [m]
            t_wall = geometry.wall_thickness(x)

            R_g = 1.0 / (h_g * A_hot)
            R_wall = t_wall / (k_w * A_hot)
            R_c_eff = 1.0 / (h_c * eta_0 * A_cool)
            R_total = R_g + R_wall + R_c_eff

            q_prime = (T_aw - T_cool) / R_total
            T_wg_new = T_aw - q_prime * R_g
            if abs(T_wg_new - T_wg_i) < tol:
                T_wg_i = T_wg_new
                break
            T_wg_i = T_wg_new
        else:
            warnings.warn(
                f"Inner iteration did not converge at x={x:.4f} m "
                f"(|dT|={abs(T_wg_new - T_wg_i):.2f} K > {tol} K).",
                RuntimeWarning,
                stacklevel=2,
            )

        T_wc_i = T_cool + q_prime * R_c_eff
        q_dot_i = q_prime / A_hot

        T_wg[i] = T_wg_i
        T_wc[i] = T_wc_i
        q_arr[i] = q_dot_i
        h_g_arr[i] = h_g
        h_c_arr[i] = h_c

        peak_T_wg = max(peak_T_wg, T_wg_i)
        T_wg_guess = T_wg_i  # warm-start next station

        j = next_station(i)
        if 0 <= j < n_stations:
            dT_cool = q_prime * dx / (m_dot_ch * props.cp)
            T_c[j] = T_cool + dT_cool

            D_h = geometry.hydraulic_diameter_cold(x)
            f_fric = haaland_f(cold.Re, roughness=0.0, D_h=D_h)
            dP_fric = f_fric * props.rho * cold.V**2 / (2.0 * D_h) * dx
            P_c[j] = P_cool - dP_fric

    if peak_T_wg > geometry.temperature_limit():
        warnings.warn(
            f"Peak T_wall_gas = {peak_T_wg:.1f} K exceeds {geometry.material_name()} "
            f"continuous limit of {geometry.temperature_limit():.1f} K.",
            RuntimeWarning,
            stacklevel=2,
        )

    dP_total = abs(float(P_c[0] - P_c[-1]))

    return {
        "x": x_arr,
        "T_wall_gas": T_wg,
        "T_wall_coolant": T_wc,
        "T_coolant": T_c,
        "P_coolant": P_c,
        "q_dot": q_arr,
        "h_g": h_g_arr,
        "h_c": h_c_arr,
        "dP_total": dP_total,
    }
