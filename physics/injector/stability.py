"""
stability.py
============

Injector stability criteria for liquid rocket engines.

The stiffness ratio (dP_inj / Pc) is the primary hydraulic stability
parameter.  A high stiffness ratio decouples chamber pressure oscillations
from propellant flow rate, preventing feed-system coupled instability.

Oxidiser and fuel streams are now checked independently so that a designer
can identify which stream is the marginal one.

Sources
-------
    Huzel, D.K. and Huang, D.H. (1992). *Modern Engineering for Design of
    Liquid-Propellant Rocket Engines.* NASA SP-125 (revised). Chapter 6,
    Section 6.3 — injector stability, stiffness criterion.

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
    dP_ox: float,
    dP_fuel: float,
    Pc: float,
    threshold: float = 0.20,
) -> dict[str, tuple[bool, float]]:
    """
    Check injector stiffness for oxidiser and fuel streams independently.

    Evaluates dP_ox / Pc and dP_fuel / Pc against the minimum threshold
    (default 0.20 = 20%, per Huzel & Huang Ch. 6 and NASA SP-8089 §4.2).
    Returns a per-stream result so the designer can identify which side is
    marginal when the two streams carry different pressure drops.

    Source: Huzel, D.K. and Huang, D.H. (1992). *Modern Engineering for
            Design of Liquid-Propellant Rocket Engines.* NASA SP-125,
            Chapter 6, Section 6.3.
            NASA SP-8089 (1976). *Liquid Rocket Engine Injectors.* §4.2.

    Parameters
    ----------
    dP_ox : float
        Oxidiser injector pressure drop [Pa].
    dP_fuel : float
        Fuel injector pressure drop [Pa].
    Pc : float
        Chamber pressure [Pa].
    threshold : float, optional
        Minimum acceptable stiffness ratio (default 0.20).

    Returns
    -------
    dict[str, tuple[bool, float]]
        ``{"ox": (passed, ratio), "fuel": (passed, ratio)}``
        where *passed* is True when ratio >= threshold.

    Examples
    --------
    >>> result = check_stiffness(3.1026e6, 3.1026e6, 6.8948e6)
    >>> result["ox"][0]   # passed
    True
    >>> abs(result["ox"][1] - 0.45) < 0.01   # ratio
    True
    >>> result["fuel"][0]
    True
    """
    ratio_ox = stiffness_ratio(dP_ox, Pc)
    ratio_fuel = stiffness_ratio(dP_fuel, Pc)
    return {
        "ox": (ratio_ox >= threshold, ratio_ox),
        "fuel": (ratio_fuel >= threshold, ratio_fuel),
    }
