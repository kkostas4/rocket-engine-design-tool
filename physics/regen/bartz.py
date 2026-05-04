"""
bartz.py
========

Bartz (1957) gas-side convective heat transfer coefficient for rocket nozzles.

Two public APIs are provided:

``bartz_h_simple``
    The original function-level API (backward-compatible).  Takes scalar
    geometry and gas property arguments; returns h [W/(m2.K)] at a single
    area ratio.  No sigma correction.

``bartz_h``
    Full API: takes an axial station x, a RaoNozzle object, a CombustionGas
    dataclass, and a wall temperature; returns h [W/(m2.K)] with the
    property-correction factor sigma applied.

``heat_flux_profile``
    Marches from convergent inlet to nozzle exit and returns arrays of x, r,
    Mach number, T_aw, h_g, and q_dot_wall for an uncoupled (constant T_wall)
    analysis.

Sources
-------
    Bartz, D.R. (1957). "A Simple Equation for Rapid Estimation of Rocket
    Nozzle Convective Heat Transfer Coefficients."
    Jet Propulsion, Vol. 27, No. 1, pp. 49-51.

    Bartz, D.R. (1953). "An Investigation of Fluid Flow Conditions in the
    Throat of a Nozzle." NACA TN-2974.  (Derivation of the full equation.)

Validity
--------
    Subsonic chamber through supersonic nozzle.
    gamma approx 1.1-1.3.
    Prandtl number Pr approx 0.5-1.0.
    Turbulent boundary layer (Re_throat >> 1e5).
    Non-radiative environment.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq  # type: ignore[import-untyped]

from physics.nozzle.rao import RaoNozzle


# ---------------------------------------------------------------------------
# Combustion gas state dataclass
# ---------------------------------------------------------------------------


@dataclass
class CombustionGas:
    """
    Frozen stagnation state of the combustion gas entering the nozzle.

    All properties are stagnation (chamber) values.  For the simplified
    Bartz equation, mu, cp, and Pr are treated as constant along the nozzle
    (stagnation-property approximation per Bartz 1957, p. 50).

    Parameters
    ----------
    Tc : float
        Stagnation (chamber) temperature [K].
    Pc : float
        Stagnation (chamber) pressure [Pa].
    gamma : float
        Ratio of specific heats [-].
    MW : float
        Molecular weight of combustion products [kg/kmol].
    cp : float
        Specific heat at constant pressure [J/(kg.K)].
    mu : float
        Dynamic viscosity [Pa.s].
    Pr : float
        Prandtl number [-].
    c_star : float
        Characteristic velocity c* [m/s].
    """

    Tc: float
    Pc: float
    gamma: float
    MW: float
    cp: float
    mu: float
    Pr: float
    c_star: float


# ---------------------------------------------------------------------------
# Isentropic Mach number from area ratio
# ---------------------------------------------------------------------------


def mach_from_area_ratio(area_ratio: float, gamma: float, supersonic: bool = True) -> float:
    """
    Solve A/A* = f(M) for M using isentropic flow relations.

    Source: Anderson, J.D. (2003). "Modern Compressible Flow," 3rd ed.,
            McGraw-Hill.  Eq. (5.20).
    Validity: perfect gas, isentropic, 1-D.

    Parameters
    ----------
    area_ratio : float
        Local area ratio A/A* (= A(x) / A_throat).  Must be >= 1.
    gamma : float
        Ratio of specific heats.
    supersonic : bool
        True to return the supersonic root (M > 1); False for subsonic.

    Returns
    -------
    float
        Mach number M.
    """
    if abs(area_ratio - 1.0) < 1.0e-9:
        return 1.0

    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))

    def _f(M: float) -> float:
        t = 1.0 + (gamma - 1.0) / 2.0 * M**2
        return (1.0 / M) * (2.0 / (gamma + 1.0) * t) ** exponent - area_ratio

    if supersonic:
        return float(brentq(_f, 1.0 + 1.0e-9, 30.0))
    else:
        return float(brentq(_f, 1.0e-9, 1.0 - 1.0e-9))


# ---------------------------------------------------------------------------
# Simplified Bartz (backward-compatible function-level API)
# ---------------------------------------------------------------------------


def bartz_h_simple(
    D_t: float,
    Pc: float,
    c_star: float,
    A_ratio: float,
    gamma: float,
    Pr: float,
    mu: float,
    cp: float,
    r_c: float,
) -> float:
    """
    Bartz simplified gas-side heat transfer coefficient [W/(m2.K)].

    Implements Eq. (5) from Bartz (1957) without the sigma wall-temperature
    correction.  Use ``bartz_h`` for the full equation with sigma.

    .. math::

        h = (0.026 / D_t^{0.2}) * (mu^{0.2} * cp / Pr^{0.6})
            * (Pc / c*)^{0.8} * (D_t / r_c)^{0.1} * (A_t / A)^{0.9}

    Source: Bartz, D.R. (1957). Jet Propulsion, Vol. 27, No. 1, pp. 49-51.
    Equation: Eq. (5).
    Validity: subsonic chamber through supersonic nozzle, gamma ~ 1.1-1.3,
              Pr ~ 0.5-1.0.

    Parameters
    ----------
    D_t : float
        Throat diameter [m].
    Pc : float
        Chamber pressure [Pa].
    c_star : float
        Characteristic velocity c* [m/s].
    A_ratio : float
        Local area ratio A/At (= 1.0 at the throat; > 1 elsewhere).
    gamma : float
        Ratio of specific heats (accepted but not used in simplified Eq. 5).
    Pr : float
        Prandtl number of combustion gas.
    mu : float
        Dynamic viscosity of combustion gas [Pa.s].
    cp : float
        Specific heat at constant pressure [J/(kg.K)].
    r_c : float
        Radius of curvature of the nozzle wall at the throat [m].

    Returns
    -------
    float
        Gas-side convective heat transfer coefficient h [W/(m2.K)].
    """
    if A_ratio <= 0.0:
        raise ValueError(f"A_ratio must be positive; got {A_ratio}")
    if r_c <= 0.0:
        raise ValueError(f"r_c must be positive; got {r_c}")
    if D_t <= 0.0:
        raise ValueError(f"D_t must be positive; got {D_t}")

    return (
        (0.026 / D_t**0.2)
        * (mu**0.2 * cp / Pr**0.6)
        * (Pc / c_star) ** 0.8
        * (D_t / r_c) ** 0.1
        * (1.0 / A_ratio) ** 0.9
    )


# ---------------------------------------------------------------------------
# Full Bartz with sigma correction  (new API)
# ---------------------------------------------------------------------------


def bartz_h(
    *,
    x: float,
    nozzle: RaoNozzle,
    gas: CombustionGas,
    T_wall: float,
    R_c: float | None = None,
) -> float:
    """
    Gas-side convective h [W/(m2.K)] at axial station x with sigma correction.

    Implements Bartz (1957) Eq. (4): the simplified coefficient from Eq. (5)
    multiplied by the property-correction factor sigma that accounts for the
    ratio of wall temperature to local stagnation temperature.

    Sigma (Bartz 1957, Eq. 4):

    .. math::

        sigma = 1 / { [0.5 * (T_w/T_0) * (1 + (gamma-1)/2 * M^2) + 0.5]^{0.68}
                       * [1 + (gamma-1)/2 * M^2]^{0.12} }

    where T_0 = Tc (stagnation temperature, constant for isentropic flow).

    Adiabatic wall temperature (turbulent recovery, Eckert 1956, cited in
    Bartz 1957):

    .. math::

        T_aw = T_0 * (1 + r_f * (gamma-1)/2 * M^2) / (1 + (gamma-1)/2 * M^2)

    with recovery factor r_f = Pr^{1/3} for turbulent boundary layers.

    Source: Bartz, D.R. (1957). "A Simple Equation for Rapid Estimation of
            Rocket Nozzle Convective Heat Transfer Coefficients."
            Jet Propulsion, Vol. 27, No. 1, pp. 49-51. Eqs. (4) and (5).
    Validity: subsonic chamber through supersonic nozzle, gamma ~ 1.1-1.3,
              Pr ~ 0.5-1.0, turbulent boundary layer.

    Parameters
    ----------
    x : float
        Axial station [m].  x < 0 in convergent section, x = 0 at throat,
        x > 0 in divergent section.
    nozzle : RaoNozzle
        Nozzle geometry object with r(x), A(x), area_ratio(x) accessors.
    gas : CombustionGas
        Stagnation combustion gas properties.
    T_wall : float
        Wall temperature (gas-side surface) [K].
    R_c : float | None
        Throat radius of curvature [m].  Defaults to 0.382 * r_throat
        (downstream throat arc radius per Sutton & Biblarz Sec. 3.3).

    Returns
    -------
    float
        Gas-side h [W/(m2.K)] including sigma correction.
    """
    if R_c is None:
        R_c = 0.382 * nozzle.r_throat

    D_t = 2.0 * nozzle.r_throat
    A_ratio = nozzle.area_ratio(x)
    supersonic = x > 0.0

    # Local Mach number from isentropic area ratio
    M = mach_from_area_ratio(A_ratio, gas.gamma, supersonic=supersonic)

    # Local stagnation enthalpy term
    M2_term = 1.0 + (gas.gamma - 1.0) / 2.0 * M**2

    # Property-correction factor sigma  (Bartz 1957, Eq. 4)
    T_w_ratio = T_wall / gas.Tc
    sigma = 1.0 / ((0.5 * T_w_ratio * M2_term + 0.5) ** 0.68 * M2_term**0.12)

    # Simplified Bartz coefficient (Eq. 5) times sigma
    h0 = bartz_h_simple(
        D_t=D_t,
        Pc=gas.Pc,
        c_star=gas.c_star,
        A_ratio=A_ratio,
        gamma=gas.gamma,
        Pr=gas.Pr,
        mu=gas.mu,
        cp=gas.cp,
        r_c=R_c,
    )

    # Validity warnings
    if not (0.5 <= gas.Pr <= 1.0):
        warnings.warn(
            f"Pr = {gas.Pr:.3f} outside Bartz validity range [0.5, 1.0].",
            RuntimeWarning,
            stacklevel=2,
        )
    if not (1.1 <= gas.gamma <= 1.3):
        warnings.warn(
            f"gamma = {gas.gamma:.3f} outside Bartz validity range [1.1, 1.3].",
            RuntimeWarning,
            stacklevel=2,
        )

    return h0 * sigma


def adiabatic_wall_temperature(x: float, nozzle: RaoNozzle, gas: CombustionGas) -> float:
    """
    Adiabatic wall temperature [K] at axial station x.

    Uses turbulent recovery factor r_f = Pr^{1/3} (Eckert 1956, cited in
    Bartz 1957, and Sutton & Biblarz 9th ed. Eq. 8-23).

    T_aw = T_0 * (1 + r_f*(gamma-1)/2*M^2) / (1 + (gamma-1)/2*M^2)

    Source: Sutton, G.P. and Biblarz, O. (2017). "Rocket Propulsion
            Elements," 9th ed., Wiley. Eq. (8-23).
    """
    A_ratio = nozzle.area_ratio(x)
    supersonic = x > 0.0
    M = mach_from_area_ratio(A_ratio, gas.gamma, supersonic=supersonic)
    r_f = gas.Pr ** (1.0 / 3.0)
    M2_half = (gas.gamma - 1.0) / 2.0 * M**2
    return gas.Tc * (1.0 + r_f * M2_half) / (1.0 + M2_half)


def heat_flux_profile(
    *,
    nozzle: RaoNozzle,
    gas: CombustionGas,
    T_wall_guess: float,
    n: int = 200,
) -> dict[str, np.ndarray]:
    """
    Uncoupled heat flux profile along the nozzle with constant T_wall.

    Marches from the convergent inlet to the nozzle exit and returns arrays
    of x, r, Mach number, adiabatic wall temperature, gas-side h, and wall
    heat flux for a fixed (uncoupled) wall temperature T_wall_guess.

    Parameters
    ----------
    nozzle : RaoNozzle
        Nozzle geometry.
    gas : CombustionGas
        Combustion gas stagnation properties.
    T_wall_guess : float
        Constant wall temperature used for sigma correction [K].
    n : int
        Number of axial stations.

    Returns
    -------
    dict with keys:
        x          : axial coordinates [m]
        r          : wall radius [m]
        M          : Mach number [-]
        T_aw       : adiabatic wall temperature [K]
        h_g        : gas-side h [W/(m2.K)]
        q_dot_wall : heat flux at wall [W/m2]  (positive into wall)
    """
    x_arr = np.linspace(nozzle.x_inlet, nozzle.x_exit, n)
    r_arr = np.array([nozzle.r(x) for x in x_arr])
    M_arr = np.zeros(n)
    T_aw_arr = np.zeros(n)
    h_g_arr = np.zeros(n)
    q_dot_arr = np.zeros(n)

    for i, x in enumerate(x_arr):
        A_ratio = nozzle.area_ratio(x)
        supersonic = x > 0.0
        M = mach_from_area_ratio(A_ratio, gas.gamma, supersonic=supersonic)
        M2_half = (gas.gamma - 1.0) / 2.0 * M**2
        r_f = gas.Pr ** (1.0 / 3.0)
        T_aw = gas.Tc * (1.0 + r_f * M2_half) / (1.0 + M2_half)

        h_g = bartz_h(x=x, nozzle=nozzle, gas=gas, T_wall=T_wall_guess)
        q_dot = h_g * (T_aw - T_wall_guess)

        M_arr[i] = M
        T_aw_arr[i] = T_aw
        h_g_arr[i] = h_g
        q_dot_arr[i] = q_dot

    return {
        "x": x_arr,
        "r": r_arr,
        "M": M_arr,
        "T_aw": T_aw_arr,
        "h_g": h_g_arr,
        "q_dot_wall": q_dot_arr,
    }
