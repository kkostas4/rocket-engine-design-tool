"""
bartz.py
========

Bartz (1957) simplified gas-side convective heat transfer coefficient for
rocket nozzles.

Source
------
    Bartz, D.R. (1957). "A Simple Equation for Rapid Estimation of Rocket
    Nozzle Convective Heat Transfer Coefficients."
    Jet Propulsion, Vol. 27, No. 1, pp. 49–51.
    Equation (5) — simplified form.

Validity
--------
    Subsonic chamber through supersonic nozzle.
    gamma ≈ 1.1–1.3.
    Prandtl number Pr ≈ 0.5–1.0.
    Convective (non-radiative) environments.
    Moderate enthalpy difference (no dissociation correction needed if
    gas is treated as ideal-frozen at the local station).
"""

from __future__ import annotations



def bartz_h(
    D_t: float,
    Pc: float,
    c_star: float,
    A_ratio: float,
    gamma: float,   # noqa: ARG001 — kept for API completeness; see note below
    Pr: float,
    mu: float,
    cp: float,
    r_c: float,
) -> float:
    """
    Bartz simplified gas-side heat transfer coefficient [W/(m²·K)].

    Uses the simplified Bartz equation (Eq. 5 in the original paper):

    .. math::

        h = \\frac{0.026}{D_t^{0.2}} \\cdot
            \\frac{\\mu^{0.2} c_p}{Pr^{0.6}} \\cdot
            \\left(\\frac{P_c}{c^*}\\right)^{0.8} \\cdot
            \\left(\\frac{D_t}{r_c}\\right)^{0.1} \\cdot
            \\left(\\frac{A_t}{A}\\right)^{0.9}

    where :math:`A_t/A = 1 / \\text{A\\_ratio}`.

    Source: Bartz, D.R. (1957). "A Simple Equation for Rapid Estimation of
            Rocket Nozzle Convective Heat Transfer Coefficients."
            Jet Propulsion, Vol. 27, No. 1, pp. 49–51.
    Equation: Eq. (5).
    Validity: subsonic chamber through supersonic nozzle, gamma ≈ 1.1–1.3,
              Pr ≈ 0.5–1.0.

    Note on gamma
    -------------
    ``gamma`` is accepted in the function signature for API completeness and
    future use (e.g. wall-temperature recovery factor, viscosity ratio
    correction per the full Bartz form, Eq. 4).  The simplified Eq. (5) used
    here does not require gamma explicitly; its effect is already absorbed into
    the empirical coefficient 0.026 and the (Pc/c*)^0.8 group.

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
        Prandtl number of combustion gas at the film temperature.
    mu : float
        Dynamic viscosity of combustion gas [Pa·s].
    cp : float
        Specific heat at constant pressure [J/(kg·K)].
    r_c : float
        Radius of curvature of the nozzle wall at the throat [m].
        Commonly taken as 0.5 * D_t (Bartz 1957, p. 50).

    Returns
    -------
    float
        Gas-side convective heat transfer coefficient h [W/(m²·K)].

    Examples
    --------
    >>> # Reference engine, throat conditions (A_ratio = 1):
    >>> h = bartz_h(
    ...     D_t=0.022313, Pc=6.8948e6, c_star=1893.3,
    ...     A_ratio=1.0, gamma=1.15, Pr=0.60,
    ...     mu=8.0e-5, cp=2000.0, r_c=0.022313/2,
    ... )
    >>> 5_000 < h < 200_000
    True
    """
    if A_ratio <= 0.0:
        raise ValueError(f"A_ratio must be positive; got {A_ratio}")
    if r_c <= 0.0:
        raise ValueError(f"r_c must be positive; got {r_c}")
    if D_t <= 0.0:
        raise ValueError(f"D_t must be positive; got {D_t}")

    h = (
        (0.026 / D_t**0.2)
        * (mu**0.2 * cp / Pr**0.6)
        * (Pc / c_star) ** 0.8
        * (D_t / r_c) ** 0.1
        * (1.0 / A_ratio) ** 0.9
    )
    return h
