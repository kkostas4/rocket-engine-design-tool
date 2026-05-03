"""
discharge.py
============

Lichtarowicz–Duggins–Markland (1965) discharge coefficient correlation for
long, sharp-edged circular orifices in incompressible non-cavitating flow.

This module provides the correlation used to compute a Reynolds-number-
dependent Cd for the fuel orifices in the pintle injector.  It is explicitly
NOT applied to the LOX annular passage (different geometry class — Lichtarowicz
is valid for cylindrical bores, not annuli).

Source
------
    Lichtarowicz, A., Duggins, R.K. and Markland, E. (1965).
    "Discharge coefficients for incompressible non-cavitating flow through
    long orifices."
    Proc. IMechE, Journal of Mechanical Engineering Science, Vol. 7, No. 2,
    pp. 210–219.

    The formula implemented here is Equation (13) of that paper, which
    covers the full Re range in a single expression:

        1/Cd = 1/Cd_∞ + (20/Re)·(1 + 2.25·L/D)
                       − 0.005·(L/D) / [1 + 7.5·(log₁₀Re − 4.8)²]

    where  Cd_∞ = 0.827 − 0.0085·(L/D)  is the asymptotic (high-Re)
    discharge coefficient.

Validity
--------
    10 ≤ Re ≤ 2×10⁵
    2  ≤ L/D ≤ 10
    Sharp-edged inlet, fully-developed turbulent pipe flow inside the orifice.
    Incompressible, non-cavitating, single-phase liquid.

Flag
----
    Inputs outside the validity band emit a RuntimeWarning.  The correlation
    is still evaluated (extrapolation may be acceptable near the boundaries)
    but the warning flags it for review.
"""

from __future__ import annotations

import math
import warnings

# Validity bounds from Lichtarowicz et al. (1965), p. 211.
_RE_MIN = 10.0
_RE_MAX = 2.0e5
_LD_MIN = 2.0
_LD_MAX = 10.0


def cd_lichtarowicz(Re: float, LD: float) -> float:
    """
    Lichtarowicz–Duggins–Markland (1965) discharge coefficient.

    Implements Equation (13) from the paper:

    .. math::

        C_{d,\\infty} = 0.827 - 0.0085 \\cdot (L/D)

        \\frac{1}{C_d} = \\frac{1}{C_{d,\\infty}}
            + \\frac{20}{Re}\\left(1 + 2.25 \\frac{L}{D}\\right)
            - \\frac{0.005 \\cdot (L/D)}{1 + 7.5\\left(\\log_{10} Re - 4.8\\right)^2}

    Source: Lichtarowicz, A., Duggins, R.K. and Markland, E. (1965).
            Proc. IMechE, J. Mech. Eng. Sci., Vol. 7, pp. 210–219. Eq. (13).
    Validity: 10 ≤ Re ≤ 2×10⁵, 2 ≤ L/D ≤ 10.

    Parameters
    ----------
    Re : float
        Reynolds number based on orifice diameter and mean velocity.
    LD : float
        Orifice length-to-diameter ratio.

    Returns
    -------
    float
        Discharge coefficient Cd [-].

    Raises
    ------
    ValueError
        If Re ≤ 0 or LD ≤ 0.

    Warns
    -----
    RuntimeWarning
        If Re or L/D are outside the stated validity range.
    """
    if Re <= 0.0:
        raise ValueError(f"Re must be positive; got {Re}")
    if LD <= 0.0:
        raise ValueError(f"LD must be positive; got {LD}")

    if not (_RE_MIN <= Re <= _RE_MAX):
        warnings.warn(
            f"Re = {Re:.3g} is outside Lichtarowicz (1965) validity range "
            f"[{_RE_MIN:.0f}, {_RE_MAX:.2g}]. "
            "Cd may be unreliable.",
            RuntimeWarning,
            stacklevel=2,
        )
    if not (_LD_MIN <= LD <= _LD_MAX):
        warnings.warn(
            f"L/D = {LD:.3g} is outside Lichtarowicz (1965) validity range "
            f"[{_LD_MIN:.0f}, {_LD_MAX:.0f}]. "
            "Cd may be unreliable.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Asymptotic high-Re discharge coefficient — Eq. (13), first sub-expression.
    cd_inf = 0.827 - 0.0085 * LD

    # Viscous correction term — Eq. (13).
    viscous = (20.0 / Re) * (1.0 + 2.25 * LD)

    # Transition correction term — Eq. (13).
    log_re = math.log10(Re)
    transition = 0.005 * LD / (1.0 + 7.5 * (log_re - 4.8) ** 2)

    inv_cd = 1.0 / cd_inf + viscous - transition
    return 1.0 / inv_cd


def cd_iterative(
    *,
    m_dot: float,
    rho: float,
    mu: float,
    dP: float,
    LD: float,
    tol: float = 1e-6,
    max_iter: int = 20,
) -> tuple[float, float]:
    """
    Self-consistent Lichtarowicz discharge coefficient for a sized orifice.

    Solves the chicken-and-egg coupling between Cd, orifice diameter D, and
    Reynolds number Re by fixed-point iteration:

    1. Guess Cd₀ = Cd_∞(L/D)  (high-Re asymptote).
    2. Compute D from  D = sqrt(4·ṁ / (Cd·n_factor))
       where n_factor = π · Cd · sqrt(2·ρ·ΔP).
    3. Compute V = Cd · sqrt(2·ΔP/ρ).
    4. Compute Re = ρ·V·D / μ.
    5. Compute new Cd via :func:`cd_lichtarowicz`.
    6. Repeat until |ΔCd| / Cd < tol.

    Source: Lichtarowicz, A., Duggins, R.K. and Markland, E. (1965).
            Proc. IMechE, J. Mech. Eng. Sci., Vol. 7, pp. 210–219. Eq. (13).
    Validity: 10 ≤ Re ≤ 2×10⁵, 2 ≤ L/D ≤ 10 (see :func:`cd_lichtarowicz`).

    Parameters
    ----------
    m_dot : float
        Mass flow rate through a *single* orifice [kg/s].
    rho : float
        Propellant density [kg/m³].
    mu : float
        Propellant dynamic viscosity [Pa·s].
    dP : float
        Pressure drop across the orifice [Pa].
    LD : float
        Orifice length-to-diameter ratio.
    tol : float, optional
        Convergence tolerance on Cd (default 1e-6).
    max_iter : int, optional
        Maximum number of iterations (default 20).

    Returns
    -------
    tuple[float, float]
        (Cd, Re) — converged discharge coefficient and Reynolds number.

    Raises
    ------
    ValueError
        If any dimensional input is non-positive.
    RuntimeError
        If the iteration does not converge within max_iter steps.
    """
    if m_dot <= 0.0:
        raise ValueError(f"m_dot must be positive; got {m_dot}")
    if rho <= 0.0:
        raise ValueError(f"rho must be positive; got {rho}")
    if mu <= 0.0:
        raise ValueError(f"mu must be positive; got {mu}")
    if dP <= 0.0:
        raise ValueError(f"dP must be positive; got {dP}")
    if LD <= 0.0:
        raise ValueError(f"LD must be positive; got {LD}")

    # Initial guess: high-Re asymptote.
    cd = 0.827 - 0.0085 * LD

    sqrt_2rho_dP = math.sqrt(2.0 * rho * dP)

    for _ in range(max_iter):
        # Orifice area from current Cd, then diameter.
        A = m_dot / (cd * sqrt_2rho_dP)
        D = math.sqrt(4.0 * A / math.pi)

        # Mean velocity and Reynolds number.
        V = cd * math.sqrt(2.0 * dP / rho)
        Re = rho * V * D / mu

        # New Cd from Lichtarowicz (suppress validity warnings inside the loop;
        # the final Re will be checked by the caller via cd_lichtarowicz).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cd_new = cd_lichtarowicz(Re, LD)

        if abs(cd_new - cd) / cd_new < tol:
            # Final call with warnings enabled to surface any validity issues.
            cd_final = cd_lichtarowicz(Re, LD)
            return cd_final, Re

        cd = cd_new

    raise RuntimeError(
        f"cd_iterative did not converge in {max_iter} iterations "
        f"(last Cd = {cd:.6f}, last ΔCd/Cd = {abs(cd_new - cd)/cd_new:.2e}). "
        "Check inputs."
    )
