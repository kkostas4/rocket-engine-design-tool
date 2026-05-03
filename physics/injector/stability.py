"""
stability.py
============

Injector stability criteria for liquid rocket engines.

The stiffness ratio (dP_inj / Pc) is the primary hydraulic stability
parameter. A high stiffness ratio decouples chamber pressure oscillations
from propellant flow rate, preventing feed-system coupled instability.

Source
------
    Huzel, D.K. and Huang, D.H. (1992). *Modern Engineering for Design of
    Liquid-Propellant Rocket Engines.* NASA SP-125 (revised). Chapter 6,
    "Design of Liquid Propellant Rocket Engines." Section 6.3 — injector
    stability, stiffness criterion.

    NASA SP-8089 (1976). *Liquid Rocket Engine Injectors.* Section 4.2 —
    hydraulic stability, minimum dP/Pc requirement.
"""

from __future__ import annotations


def stiffness_ratio(dP_inj: float, Pc: float) -> float:
    """
    Injector stiffness ratio: dP_inj / Pc [-].

    The stiffness ratio quantifies how much the injector pressure drop
    attenuates feed-system disturbances relative to the chamber pressure.
    Higher ratios improve hydraulic stability at the cost of pumping power.

    Threshold: >= 0.20 (20%) per Huzel & Huang Ch. 6 and NASA SP-8089 §4.2.
    Values of 0.30–0.50 are typical for stable engines.

    Source: Huzel, D.K. and Huang, D.H. (1992). *Modern Engineering for
            Design of Liquid-Propellant Rocket Engines.* NASA SP-125,
            Chapter 6, Section 6.3.
            NASA SP-8089 (1976). *Liquid Rocket Engine Injectors.* §4.2.

    Parameters
    ----------
    dP_inj : float
        Injector pressure drop [Pa].
    Pc : float
        Chamber pressure [Pa].

    Returns
    -------
    float
        Dimensionless stiffness ratio dP_inj / Pc.

    Raises
    ------
    ValueError
        If either input is zero or negative.

    Examples
    --------
    >>> r = stiffness_ratio(3.1026e6, 6.8948e6)
    >>> abs(r - 0.45) < 0.01
    True
    """
    if dP_inj <= 0.0:
        raise ValueError(f"dP_inj must be positive; got {dP_inj}")
    if Pc <= 0.0:
        raise ValueError(f"Pc must be positive; got {Pc}")

    return dP_inj / Pc


def check_stiffness(
    dP_inj: float,
    Pc: float,
    threshold: float = 0.20,
) -> tuple[bool, float, str]:
    """
    Check whether the injector stiffness ratio meets the minimum threshold.

    Evaluates dP_inj / Pc against the supplied threshold (default 0.20 = 20%).
    Returns a structured result suitable for automated design checks.

    Source: Huzel, D.K. and Huang, D.H. (1992). *Modern Engineering for
            Design of Liquid-Propellant Rocket Engines.* NASA SP-125,
            Chapter 6, Section 6.3.
            NASA SP-8089 (1976). *Liquid Rocket Engine Injectors.* §4.2.

    Parameters
    ----------
    dP_inj : float
        Injector pressure drop [Pa].
    Pc : float
        Chamber pressure [Pa].
    threshold : float, optional
        Minimum acceptable stiffness ratio (default 0.20).

    Returns
    -------
    passed : bool
        True if stiffness ratio >= threshold.
    ratio : float
        Computed dP_inj / Pc.
    message : str
        Human-readable assessment string.

    Examples
    --------
    >>> passed, ratio, msg = check_stiffness(3.1026e6, 6.8948e6)
    >>> passed
    True
    >>> abs(ratio - 0.45) < 0.01
    True
    """
    ratio = stiffness_ratio(dP_inj, Pc)
    passed = ratio >= threshold

    if passed:
        message = (
            f"PASS — stiffness ratio {ratio:.3f} >= threshold {threshold:.3f} "
            f"(Huzel & Huang Ch. 6 / NASA SP-8089 §4.2)"
        )
    else:
        message = (
            f"FAIL — stiffness ratio {ratio:.3f} < threshold {threshold:.3f} "
            f"(Huzel & Huang Ch. 6 / NASA SP-8089 §4.2): "
            f"risk of feed-coupled combustion instability"
        )

    return passed, ratio, message
