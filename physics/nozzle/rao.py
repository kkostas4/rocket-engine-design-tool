"""
rao.py
======

Rao parabolic bell nozzle contour and throat sizing for the reference engine.

Two public functions:
    - ``throat_area``   : isentropic throat sizing from c*.
    - ``rao_contour``   : Rao parabolic bell nozzle (x, r) arrays in metres.

Sources
-------
Throat sizing
    Huzel, D.K. and Huang, D.H. (1992). "Modern Engineering for Design of
    Liquid-Propellant Rocket Engines." AIAA Progress in Astronautics and
    Aeronautics, Vol. 147, p. 69.  (Equation relating At, m_dot, c*, Pc.)

Rao contour
    Rao, G.V.R. (1958). "Exhaust Nozzle Contour for Optimum Thrust."
    Jet Propulsion, Vol. 28, No. 6, pp. 377–382.
    (Parabolic approximation to the optimum bell contour.)

    Practical implementation follows the parametric description in:
    Sutton, G.P. and Biblarz, O. (2017). "Rocket Propulsion Elements,"
    9th ed., Wiley. Section 3.3 (bell nozzle geometry).

Validity
--------
    Rao parabolic approximation is valid for expansion ratios 5 ≤ ε ≤ 50
    and nozzle lengths 60 %–100 % of the equivalent 15° cone length.
    The upstream convergent section uses a circular arc of radius 1.5 * R_t
    (standard practice per Huzel & Huang §4.2).
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Throat sizing
# ---------------------------------------------------------------------------


def throat_area(m_dot: float, Pc: float, c_star: float) -> float:
    """
    Isentropic throat area from the c* definition.

    The characteristic velocity is defined as::

        c* = Pc * At / m_dot  →  At = m_dot * c* / Pc

    Source: Huzel & Huang (1992), Vol. 147, p. 69, Eq. (4-25).
    Validity: steady, 1-D isentropic flow; combustion efficiency
              captured implicitly in the empirical c* value.

    Parameters
    ----------
    m_dot : float
        Total propellant mass flow rate [kg/s].
    Pc : float
        Chamber pressure [Pa].
    c_star : float
        Characteristic velocity c* [m/s].

    Returns
    -------
    float
        Throat area At [m²].
    """
    return m_dot * c_star / Pc


# ---------------------------------------------------------------------------
# Rao parabolic bell nozzle
# ---------------------------------------------------------------------------

# Rao (1958) recommended entrance half-angles at the throat and exit
# half-angles for the parabolic approximation as a function of ε.
# For practical engineering use the following defaults are widely cited
# (Sutton & Biblarz §3.3, Table 3-2; 80 % bell length):
_THETA_N_DEG_DEFAULT = 25.0   # throat-end half-angle of bell section [°]
_THETA_E_DEG_DEFAULT = 8.0    # exit half-angle [°]
_BELL_FRAC_DEFAULT = 0.80     # fraction of equivalent 15° cone length


def rao_contour(
    R_t: float,
    eps: float,
    theta_n_deg: float = _THETA_N_DEG_DEFAULT,
    theta_e_deg: float = _THETA_E_DEG_DEFAULT,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rao parabolic bell nozzle contour.

    Returns arrays ``(x, r)`` in metres.  The coordinate origin is at the
    throat (x = 0, r = R_t).  The upstream convergent section is a circular
    arc of radius 1.5 * R_t; the divergent bell section is a parabola fitted
    through the throat exit point with slope tan(theta_n), ending exactly at
    R_e = R_t * sqrt(eps) with slope tan(theta_e).

    Derivation of bell length
    -------------------------
    For the parabola  r(x) = A·x² + B·x + C  with three free coefficients,
    the three independent conditions are::

        r(x_N) = r_N           [start position]
        r'(x_N) = tan(θ_n)    [start slope]
        r'(x_E) = tan(θ_e)    [end slope]

    Substituting into r(x_E) = R_e gives the closed-form bell length::

        d = x_E − x_N = 2·(R_e − r_N) / (tan θ_n + tan θ_e)

    This guarantees the exit radius is exact regardless of the angle choice.

    Source: Rao, G.V.R. (1958). "Exhaust Nozzle Contour for Optimum Thrust."
            Jet Propulsion, Vol. 28, No. 6, pp. 377–382.
            Parametric implementation per Sutton & Biblarz (2017) §3.3.
    Equation: Parabolic approximation, Eq. (3-27) family.
    Validity: 5 ≤ ε ≤ 50, gamma ≈ 1.1–1.4.

    Parameters
    ----------
    R_t : float
        Throat radius [m].
    eps : float
        Nozzle expansion ratio Ae/At.
    theta_n_deg : float
        Bell entrance half-angle at the parabola start (just past the
        downstream throat arc) [°].  Default 25°.
    theta_e_deg : float
        Bell exit half-angle [°].  Default 8°.
    n_points : int
        Number of (x, r) points along the full contour.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (x, r) — axial coordinate [m] and radius [m].
        x = 0 at the throat; x < 0 in the convergent section.
    """
    R_e = R_t * math.sqrt(eps)  # exit radius [m]
    theta_n = math.radians(theta_n_deg)
    theta_e = math.radians(theta_e_deg)
    tan_n = math.tan(theta_n)
    tan_e = math.tan(theta_e)

    # ------------------------------------------------------------------
    # 1.  Upstream convergent section
    #     Circular arc of radius r_u = 1.5*R_t centred at (0, R_t + r_u).
    #     Arc sweeps 30° upstream from the throat tangent point.
    # ------------------------------------------------------------------
    r_u = 1.5 * R_t
    arc_sweep_rad = math.radians(30.0)
    n_conv = max(10, n_points // 5)
    phi_conv = np.linspace(-arc_sweep_rad, 0.0, n_conv)
    x_conv = r_u * np.sin(phi_conv)           # x < 0 upstream of throat
    r_conv = R_t + r_u * (1.0 - np.cos(phi_conv))

    # ------------------------------------------------------------------
    # 2.  Downstream throat arc (Sutton §3.3)
    #     Radius r_d = 0.382*R_t sweeps from 0° at the throat to theta_n,
    #     giving the parabola start point (x_N, r_N).
    # ------------------------------------------------------------------
    r_d = 0.382 * R_t
    x_N = r_d * math.sin(theta_n)
    r_N = R_t + r_d * (1.0 - math.cos(theta_n))

    # ------------------------------------------------------------------
    # 3.  Divergent parabola
    #
    #     Bell length from closed-form derivation (see docstring):
    #         d = 2*(R_e - r_N) / (tan_n + tan_e)
    #
    #     Parabola coefficients (3 conditions: r_N, r'_N = tan_n, r'_E = tan_e):
    #         A = (tan_n - tan_e) / (2*(x_N - x_E))
    #         B = tan_n - 2*A*x_N
    #         C = r_N - A*x_N^2 - B*x_N
    # ------------------------------------------------------------------
    d = 2.0 * (R_e - r_N) / (tan_n + tan_e)
    x_E = x_N + d

    A_par = (tan_n - tan_e) / (2.0 * (x_N - x_E))
    B_par = tan_n - 2.0 * A_par * x_N
    C_par = r_N - A_par * x_N**2 - B_par * x_N

    n_bell = n_points - n_conv
    x_bell = np.linspace(x_N, x_E, n_bell)
    r_bell = A_par * x_bell**2 + B_par * x_bell + C_par

    # ------------------------------------------------------------------
    # 4.  Concatenate, dropping the duplicate throat-arc/parabola junction
    # ------------------------------------------------------------------
    x_all = np.concatenate([x_conv, x_bell[1:]])
    r_all = np.concatenate([r_conv, r_bell[1:]])

    return x_all, r_all
