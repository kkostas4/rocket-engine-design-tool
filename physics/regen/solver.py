"""
solver.py
=========

1-D regenerative cooling solver: couples Bartz gas-side heat transfer with
Dittus-Boelter coolant-side convection, fin efficiency, and a conduction
resistance through the wall.

Algorithm
---------
Discretise the nozzle axis into N stations.  At each station (marched in
the coolant flow direction):

  1. Evaluate T_aw(x) from Bartz / isentropic relations.
  2. Fixed-point iteration on T_wall_gas:
     a. h_g <- bartz_h(x, nozzle, gas, T_wall=T_wall_gas)   [sigma included]
     b. h_c <- Dittus-Boelter(Re_c, Pr_c) using CoolProp properties
     c. eta_fin <- straight-fin efficiency (Incropera Eq. 3.80)
     d. eta_0  <- overall surface efficiency (Incropera Eq. 3.86)
     e. q_prime = (T_aw - T_c) / (R_g + R_wall + R_c)    [W/m per channel]
        where R = 1/(h*A) and A = areas per unit axial length per channel.
     f. T_wall_gas_new = T_aw - q_prime * R_g
     g. Repeat a-f until |T_wall_gas_new - T_wall_gas| < tol.
  3. Update coolant: dT_c = q_prime * dx / (m_dot_ch * cp_c)
  4. Update pressure: dP_c = f_Haaland * rho_c * V_c^2 / (2*D_h) * dx

Sources
-------
Dittus-Boelter correlation:
    Dittus, F.W. and Boelter, L.M.K. (1930). "Heat Transfer in Automobile
    Radiators of the Tubular Type." Univ. of California Publ. Eng. Vol. 2,
    pp. 443-461.  Re-stated in McAdams, W.H. (1954). "Heat Transmission,"
    3rd ed., McGraw-Hill. Ch. 8.
    Nu = 0.023 * Re^0.8 * Pr^0.4  (heating the coolant).
    Validity: Re > 10,000; 0.7 < Pr < 160; L/D > 10.

Haaland friction factor:
    Haaland, S.E. (1983). "Simple and Explicit Formulas for the Friction
    Factor in Turbulent Pipe Flow." J. Fluids Eng., Vol. 105, pp. 89-90.
    Eq. (3): 1/sqrt(f) = -1.8 * log10[(eps_r/D/3.7)^1.11 + 6.9/Re].
    For smooth channels (eps_r -> 0): 1/sqrt(f) = -1.8 * log10(6.9/Re).

Fin efficiency and overall surface efficiency:
    Incropera et al. (2011). "Fundamentals of Heat and Mass Transfer," 7th ed.
    Table 3.4 (Case A) and Eq. (3.86).

Thermal resistance network:
    NASA SP-8087 (1977). "Liquid Rocket Engine Fluid-Cooled Combustion
    Chambers." Sec. 3.1.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from physics.nozzle.rao import RaoNozzle
from physics.regen.bartz import CombustionGas, adiabatic_wall_temperature, bartz_h
from physics.regen.coolant import CoolingChannelGeometry, CoolantInlet, WallMaterial

try:
    from CoolProp.CoolProp import PropsSI  # type: ignore[import-untyped]

    _COOLPROP_AVAILABLE = True
except ImportError:
    _COOLPROP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coolant_props(T: float, P: float, fluid: str) -> tuple[float, float, float, float, float]:
    """
    Return (rho, cp, mu, k, Pr) for the coolant at given T, P via CoolProp.

    Raises RuntimeError if CoolProp is not installed.
    """
    if not _COOLPROP_AVAILABLE:
        raise RuntimeError(
            "CoolProp is required for the regen solver.  "
            "Install with: pip install CoolProp"
        )
    rho = PropsSI("D", "T", T, "P", P, fluid)
    cp = PropsSI("C", "T", T, "P", P, fluid)
    mu = PropsSI("V", "T", T, "P", P, fluid)
    k = PropsSI("L", "T", T, "P", P, fluid)
    Pr = PropsSI("Prandtl", "T", T, "P", P, fluid)
    return rho, cp, mu, k, Pr


def _dittus_boelter(Re: float, Pr: float, D_h: float, k_c: float) -> float:
    """
    Coolant-side h from Dittus-Boelter (heating the fluid).

    Nu = 0.023 * Re^0.8 * Pr^0.4

    Source: McAdams (1954), restating Dittus & Boelter (1930).
    Validity: Re > 10,000; 0.7 < Pr < 160.
    """
    if Re < 1.0e4:
        warnings.warn(
            f"Re = {Re:.0f} < 10,000: Dittus-Boelter is outside validity range.",
            RuntimeWarning,
            stacklevel=3,
        )
    if not (0.7 <= Pr <= 160.0):
        warnings.warn(
            f"Pr = {Pr:.3f} outside Dittus-Boelter validity range [0.7, 160].",
            RuntimeWarning,
            stacklevel=3,
        )
    Nu = 0.023 * Re**0.8 * Pr**0.4
    return Nu * k_c / D_h


def _haaland_f(Re: float, roughness: float = 0.0, D_h: float = 1.0) -> float:
    """
    Darcy friction factor from Haaland (1983) Eq. (3).

    For smooth channels (roughness = 0):
        1/sqrt(f) = -1.8 * log10(6.9 / Re)

    Source: Haaland, S.E. (1983). J. Fluids Eng., Vol. 105, pp. 89-90.
    Validity: Re > ~3,000 (turbulent flow).
    """
    if roughness <= 0.0:
        inv_sqrt_f = -1.8 * math.log10(6.9 / max(Re, 1.0))
    else:
        eps_r = roughness / D_h
        inv_sqrt_f = -1.8 * math.log10((eps_r / 3.7) ** 1.11 + 6.9 / max(Re, 1.0))
    if inv_sqrt_f == 0.0:
        return 0.02  # fallback
    return 1.0 / inv_sqrt_f**2


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------


def solve_regen(
    *,
    nozzle: RaoNozzle,
    gas: CombustionGas,
    channel_geom: CoolingChannelGeometry,
    coolant_inlet: CoolantInlet,
    wall_material: WallMaterial,
    direction: str = "counterflow",
    n_stations: int = 200,
    tol: float = 1.0,
) -> dict[str, Any]:
    """
    1-D coupled regenerative cooling solver.

    Marches along the nozzle in the coolant flow direction.  At each station
    the gas-side wall temperature is found by fixed-point iteration coupling
    the Bartz h_g (with sigma) to the Dittus-Boelter h_c through the thermal
    resistance network.

    Parameters
    ----------
    nozzle : RaoNozzle
        Nozzle contour with r(x) and area_ratio(x) accessors.
    gas : CombustionGas
        Combustion gas stagnation properties.
    channel_geom : CoolingChannelGeometry
        Cooling channel geometry (width, height, wall_thickness, land_thickness
        as callables of x).
    coolant_inlet : CoolantInlet
        Coolant inlet state (T, P, m_dot, fluid name).
    wall_material : WallMaterial
        Wall thermal conductivity k(T) and T_max_continuous.
    direction : str
        'counterflow' (coolant flows from exit to inlet; default) or 'coflow'.
    n_stations : int
        Number of axial discretisation stations.
    tol : float
        Convergence tolerance on T_wall_gas [K] for the inner iteration.

    Returns
    -------
    dict with keys:
        x               ndarray [m]      -- axial stations (inlet to exit)
        r               ndarray [m]      -- wall radius
        T_wall_gas      ndarray [K]      -- gas-side wall surface temperature
        T_wall_coolant  ndarray [K]      -- coolant-side wall surface temperature
        T_coolant       ndarray [K]      -- coolant bulk temperature
        P_coolant       ndarray [Pa]     -- coolant pressure
        q_dot           ndarray [W/m2]   -- heat flux at gas-side wall
        h_g             ndarray [W/m2.K] -- gas-side h
        h_c             ndarray [W/m2.K] -- coolant-side h
        dP_total        float [Pa]       -- total coolant pressure drop
    """
    if direction not in ("counterflow", "coflow"):
        raise ValueError(f"direction must be 'counterflow' or 'coflow', got {direction!r}")

    x_arr = np.linspace(nozzle.x_inlet, nozzle.x_exit, n_stations)
    dx = float(x_arr[1] - x_arr[0])

    # Allocate output arrays (indexed inlet-to-exit, i=0 is x_inlet)
    T_wg = np.zeros(n_stations)  # T_wall_gas
    T_wc = np.zeros(n_stations)  # T_wall_coolant
    T_c = np.zeros(n_stations)  # coolant bulk temperature
    P_c = np.zeros(n_stations)  # coolant pressure
    q_arr = np.zeros(n_stations)  # heat flux [W/m2]
    h_g_arr = np.zeros(n_stations)
    h_c_arr = np.zeros(n_stations)
    r_arr = np.array([nozzle.r(x) for x in x_arr])

    m_dot_ch = coolant_inlet.m_dot / channel_geom.n_channels  # per channel

    # Set coolant inlet boundary condition
    if direction == "counterflow":
        # Coolant enters at exit (i = n-1), marches toward inlet (i = 0)
        T_c[-1] = coolant_inlet.T
        P_c[-1] = coolant_inlet.P
        march_idx = range(n_stations - 1, -1, -1)

        def next_station(i: int) -> int:
            return i - 1

    else:  # coflow
        # Coolant enters at inlet (i = 0), marches toward exit (i = n-1)
        T_c[0] = coolant_inlet.T
        P_c[0] = coolant_inlet.P
        march_idx = range(n_stations)

        def next_station(i: int) -> int:
            return i + 1

    # Precompute T_aw (independent of T_wall and coolant state)
    T_aw_arr = np.array([adiabatic_wall_temperature(x, nozzle, gas) for x in x_arr])

    # ------------------------------------------------------------------
    # Main march
    # ------------------------------------------------------------------
    T_wg_guess = gas.Tc * 0.30  # initial guess: ~30% of Tc
    peak_T_wg = 0.0

    for i in march_idx:
        x = float(x_arr[i])
        T_cool = float(T_c[i])
        P_cool = float(P_c[i])
        T_aw = float(T_aw_arr[i])

        # Local channel geometry
        w = channel_geom.width(x)
        H = channel_geom.height(x)
        t_wall = channel_geom.wall_thickness(x)
        t_land = channel_geom.land_thickness(x)

        # Gas-side area per unit axial length per channel [m2/m]
        A_gas = 2.0 * math.pi * r_arr[i] / channel_geom.n_channels
        # Coolant-side area (3-sided: base + 2 fin sides) per unit length per channel
        A_cool = w + 2.0 * H
        # Channel flow area and hydraulic diameter
        A_c = w * H
        D_h = 4.0 * A_c / (2.0 * (w + H))

        # Coolant properties via CoolProp
        try:
            rho_c, cp_c, mu_c, k_c, Pr_c = _coolant_props(T_cool, P_cool, coolant_inlet.fluid)
        except Exception as exc:
            warnings.warn(
                f"CoolProp failed at x={x:.4f} m, T={T_cool:.1f} K, P={P_cool:.0f} Pa: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            # Fall back to inlet properties to avoid crash
            rho_c, cp_c, mu_c, k_c, Pr_c = _coolant_props(
                coolant_inlet.T, coolant_inlet.P, coolant_inlet.fluid
            )

        V_c = m_dot_ch / (rho_c * A_c)
        Re_c = rho_c * V_c * D_h / mu_c

        h_c = _dittus_boelter(Re_c, Pr_c, D_h, k_c)

        # ----------------------------------------------------------
        # Inner fixed-point iteration: converge T_wall_gas
        # ----------------------------------------------------------
        T_wg_i = T_wg_guess
        for _iter in range(60):
            h_g = bartz_h(x=x, nozzle=nozzle, gas=gas, T_wall=T_wg_i)
            k_w = wall_material.k(T_wg_i)

            eta_fin = channel_geom.fin_efficiency(x, h_c, k_w)
            eta_0 = channel_geom.overall_surface_efficiency(x, eta_fin)

            # Thermal resistances per unit axial length per channel [K.m/W]
            R_g = 1.0 / (h_g * A_gas)
            R_wall = t_wall / (k_w * A_gas)
            R_c_eff = 1.0 / (h_c * eta_0 * A_cool)
            R_total = R_g + R_wall + R_c_eff

            q_prime = (T_aw - T_cool) / R_total  # W/m per channel

            T_wg_new = T_aw - q_prime * R_g
            if abs(T_wg_new - T_wg_i) < tol:
                T_wg_i = T_wg_new
                break
            T_wg_i = T_wg_new
        else:
            warnings.warn(
                f"Inner iteration did not converge at x={x:.4f} m (|dT|="
                f"{abs(T_wg_new - T_wg_i):.2f} K > {tol} K).",
                RuntimeWarning,
                stacklevel=2,
            )

        T_wc_i = T_cool + q_prime * R_c_eff
        q_dot_i = q_prime / A_gas  # W/m2, referenced to gas-side area

        T_wg[i] = T_wg_i
        T_wc[i] = T_wc_i
        q_arr[i] = q_dot_i
        h_g_arr[i] = h_g
        h_c_arr[i] = h_c

        peak_T_wg = max(peak_T_wg, T_wg_i)
        T_wg_guess = T_wg_i  # warm-start next station

        # ----------------------------------------------------------
        # March coolant to next station (if not at end of domain)
        # ----------------------------------------------------------
        j = next_station(i)
        if 0 <= j < n_stations:
            # Energy balance
            dT_cool = q_prime * dx / (m_dot_ch * cp_c)
            T_c[j] = T_cool + dT_cool  # temperature rises in coolant flow direction

            # Darcy-Weisbach pressure drop
            f_fric = _haaland_f(Re_c, roughness=0.0, D_h=D_h)
            dP_fric = f_fric * rho_c * V_c**2 / (2.0 * D_h) * dx
            P_c[j] = P_cool - dP_fric  # pressure drops in coolant flow direction

    # ------------------------------------------------------------------
    # Post-process
    # ------------------------------------------------------------------
    if peak_T_wg > wall_material.T_max_continuous:
        warnings.warn(
            f"Peak T_wall_gas = {peak_T_wg:.1f} K exceeds {wall_material.name} "
            f"continuous limit of {wall_material.T_max_continuous:.1f} K.",
            RuntimeWarning,
            stacklevel=2,
        )

    dP_total = abs(float(P_c[0] - P_c[-1]))

    return {
        "x": x_arr,
        "r": r_arr,
        "T_wall_gas": T_wg,
        "T_wall_coolant": T_wc,
        "T_coolant": T_c,
        "P_coolant": P_c,
        "q_dot": q_arr,
        "h_g": h_g_arr,
        "h_c": h_c_arr,
        "dP_total": dP_total,
    }
