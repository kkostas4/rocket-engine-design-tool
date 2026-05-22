"""
orifice.py
==========

Incompressible orifice flow primitives for liquid rocket injector sizing.

These are the fundamental relations used to size fuel and oxidiser injection
orifices. The incompressible Bernoulli-based orifice equation is valid for
liquid propellants at moderate injection velocities (Ma_liquid << 1).

Source
------
    Sutton, G.P. and Biblarz, O. (2017). *Rocket Propulsion Elements*,
    9th edition, John Wiley & Sons. Chapter 8, "Thrust Chambers".
    Eq. (8-1) through (8-3) — injector orifice sizing.

Validity
--------
    Incompressible liquid flow (density effectively constant across orifice).
    Injection velocities typically 5–100 m/s for liquid propellants.
    Discharge coefficient Cd typically 0.5–0.8 depending on orifice geometry.
"""

from __future__ import annotations

import math


def orifice_mass_flow(Cd: float, A: float, rho: float, dP: float) -> float:
    """
    Mass flow rate through an incompressible orifice.

    Applies the standard Bernoulli-based orifice equation:

    .. math::

        \\dot{m} = C_d \\cdot A \\cdot \\sqrt{2 \\rho \\, \\Delta P}

    Source: Sutton & Biblarz (2017), *Rocket Propulsion Elements*, 9th ed.,
            Ch. 8, Eq. (8-1).

    Parameters
    ----------
    Cd : float
        Discharge coefficient [dimensionless], typically 0.5–0.8.
    A : float
        Total orifice area [m²].
    rho : float
        Propellant density [kg/m³].
    dP : float
        Pressure drop across the orifice [Pa].

    Returns
    -------
    float
        Mass flow rate [kg/s].

    Raises
    ------
    ValueError
        If any input is zero or negative.

    Examples
    --------
    >>> m = orifice_mass_flow(0.6, 1.054e-5, 386.0, 3.1026e6)
    >>> abs(m - 0.3096) / 0.3096 < 0.01
    True
    """
    if Cd <= 0.0:
        raise ValueError(f"Cd must be positive; got {Cd}")
    if A <= 0.0:
        raise ValueError(f"A must be positive; got {A}")
    if rho <= 0.0:
        raise ValueError(f"rho must be positive; got {rho}")
    if dP <= 0.0:
        raise ValueError(f"dP must be positive; got {dP}")

    return Cd * A * math.sqrt(2.0 * rho * dP)


def orifice_area_required(
    m_dot: float, Cd: float, rho: float, dP: float
) -> float:
    """
    Orifice area required to pass a given mass flow rate.

    Rearrangement of the incompressible orifice equation:

    .. math::

        A = \\frac{\\dot{m}}{C_d \\sqrt{2 \\rho \\, \\Delta P}}

    Source: Sutton & Biblarz (2017), *Rocket Propulsion Elements*, 9th ed.,
            Ch. 8, Eq. (8-1) rearranged.

    Parameters
    ----------
    m_dot : float
        Required mass flow rate [kg/s].
    Cd : float
        Discharge coefficient [dimensionless].
    rho : float
        Propellant density [kg/m³].
    dP : float
        Pressure drop across the orifice [Pa].

    Returns
    -------
    float
        Required total orifice area [m²].

    Raises
    ------
    ValueError
        If any input is zero or negative.

    Examples
    --------
    >>> A = orifice_area_required(0.3096, 0.6, 386.0, 3.1026e6)
    >>> abs(A - 1.054e-5) / 1.054e-5 < 0.01
    True
    """
    if m_dot <= 0.0:
        raise ValueError(f"m_dot must be positive; got {m_dot}")
    if Cd <= 0.0:
        raise ValueError(f"Cd must be positive; got {Cd}")
    if rho <= 0.0:
        raise ValueError(f"rho must be positive; got {rho}")
    if dP <= 0.0:
        raise ValueError(f"dP must be positive; got {dP}")

    return m_dot / (Cd * math.sqrt(2.0 * rho * dP))


def injection_velocity(m_dot: float, A: float, rho: float) -> float:
    """
    Mean injection velocity from continuity.

    .. math::

        V = \\frac{\\dot{m}}{\\rho \\cdot A}

    Source: Sutton & Biblarz (2017), *Rocket Propulsion Elements*, 9th ed.,
            Ch. 8 — continuity equation applied to injector element.

    Parameters
    ----------
    m_dot : float
        Mass flow rate [kg/s].
    A : float
        Total flow area [m²].
    rho : float
        Propellant density [kg/m³].

    Returns
    -------
    float
        Mean injection velocity [m/s].

    Raises
    ------
    ValueError
        If any input is zero or negative.

    Examples
    --------
    >>> V = injection_velocity(0.3096, 1.054e-5, 386.0)
    >>> 10.0 < V < 200.0
    True
    """
    if m_dot <= 0.0:
        raise ValueError(f"m_dot must be positive; got {m_dot}")
    if A <= 0.0:
        raise ValueError(f"A must be positive; got {A}")
    if rho <= 0.0:
        raise ValueError(f"rho must be positive; got {rho}")

    return m_dot / (rho * A)
