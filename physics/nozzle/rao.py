"""
rao.py
======

Rao parabolic bell nozzle contour and throat sizing for the reference engine.

Public API
----------
Functions (backward-compatible):
    - ``throat_area``   : isentropic throat sizing from c*.
    - ``rao_contour``   : Rao parabolic bell nozzle (x, r) arrays in metres.

Dataclass:
    - ``RaoNozzle``     : interpolated r(x), A(x), area_ratio(x) accessors
                          built on top of rao_contour.

Sources
-------
Throat sizing
    Huzel, D.K. and Huang, D.H. (1992). "Modern Engineering for Design of
    Liquid-Propellant Rocket Engines." AIAA Progress in Astronautics and
    Aeronautics, Vol. 147, p. 69.  (Equation relating At, m_dot, c*, Pc.)

Rao contour
    Rao, G.V.R. (1958). "Exhaust Nozzle Contour for Optimum Thrust."
    Jet Propulsion, Vol. 28, No. 6, pp. 377-382.
    (Parabolic approximation to the optimum bell contour.)

    Practical implementation follows the parametric description in:
    Sutton, G.P. and Biblarz, O. (2017). "Rocket Propulsion Elements,"
    9th ed., Wiley. Section 3.3, Table 3-2 (bell nozzle geometry).

Validity
--------
    Rao parabolic approximation is valid for expansion ratios 5 <= eps <= 50
    and nozzle lengths 60%-100% of the equivalent 15 deg cone length.
    The upstream convergent section uses a circular arc of radius 1.5 * R_t
    and the downstream throat arc uses 0.382 * R_t, both per Huzel & Huang
    Sec. 4.2 / Sutton & Biblarz Sec. 3.3.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# Throat sizing
# ---------------------------------------------------------------------------


def throat_area(m_dot: float, Pc: float, c_star: float) -> float:
    """
    Isentropic throat area from the c* definition.

    The characteristic velocity is defined as::

        c* = Pc * At / m_dot  ->  At = m_dot * c* / Pc

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
        Throat area At [m2].
    """
    return m_dot * c_star / Pc


# ---------------------------------------------------------------------------
# Rao parabolic bell nozzle  (array-returning function)
# ---------------------------------------------------------------------------

# Default angles from Sutton & Biblarz (2017) Sec. 3.3, Table 3-2 for an
# approximately 80%-85% bell at expansion ratios eps ~ 10-20.  For other
# expansion ratios or length fractions consult Rao (1958) Table 1 directly.
_THETA_N_DEG_DEFAULT = 25.0  # bell entrance half-angle [deg]
_THETA_E_DEG_DEFAULT = 8.0  # bell exit half-angle [deg]
_BELL_FRAC_DEFAULT = 0.80  # nominal fraction of equivalent 15 deg cone length


def rao_contour(
    R_t: float,
    eps: float,
    theta_n_deg: float = _THETA_N_DEG_DEFAULT,
    theta_e_deg: float = _THETA_E_DEG_DEFAULT,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rao parabolic bell nozzle contour including the downstream throat arc.

    Returns arrays ``(x, r)`` in metres.  The coordinate origin is at the
    throat (x = 0, r = R_t).  The contour has three sections:

    1. Convergent arc   : circular arc R = 1.5*R_t, 30 deg sweep, x < 0.
    2. Downstream throat arc : circular arc R = 0.382*R_t from x=0 to x_N.
    3. Divergent bell parabola : from (x_N, r_N) to (x_E, R_e).

    Derivation of bell length
    -------------------------
    For the parabola r(x) = A*x^2 + B*x + C with three free coefficients,
    the three independent conditions are::

        r(x_N) = r_N           [start position]
        r'(x_N) = tan(theta_n) [start slope  -- matched to arc tangent]
        r'(x_E) = tan(theta_e) [end slope]

    Substituting r(x_E) = R_e gives the closed-form bell length::

        d = x_E - x_N = 2*(R_e - r_N) / (tan(theta_n) + tan(theta_e))

    This guarantees the exit radius is exact regardless of the angle choice,
    and that the slope is C1-continuous at both junction points (x=0 and x_N).

    Source: Rao, G.V.R. (1958). "Exhaust Nozzle Contour for Optimum Thrust."
            Jet Propulsion, Vol. 28, No. 6, pp. 377-382.
            Parametric description per Sutton & Biblarz (2017) Sec. 3.3.
    Equation: Parabolic approximation, Eq. (3-27) family.
    Validity: 5 <= eps <= 50, gamma approx 1.1-1.4.

    Parameters
    ----------
    R_t : float
        Throat radius [m].
    eps : float
        Nozzle expansion ratio Ae/At.
    theta_n_deg : float
        Bell entrance half-angle at the parabola start [deg].  Default 25 deg.
    theta_e_deg : float
        Bell exit half-angle [deg].  Default 8 deg.
    n_points : int
        Total number of (x, r) points along the full contour.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (x, r) -- axial coordinate [m] and radius [m].
        x = 0 at the throat; x < 0 in the convergent section.
    """
    R_e = R_t * math.sqrt(eps)
    theta_n = math.radians(theta_n_deg)
    theta_e = math.radians(theta_e_deg)
    tan_n = math.tan(theta_n)
    tan_e = math.tan(theta_e)

    # ------------------------------------------------------------------
    # 1.  Upstream convergent section
    #     Circular arc of radius r_u = 1.5*R_t.  Sweeps 30 deg upstream.
    # ------------------------------------------------------------------
    r_u = 1.5 * R_t
    n_conv = max(10, n_points // 5)
    phi_conv = np.linspace(-math.radians(30.0), 0.0, n_conv)
    x_conv = r_u * np.sin(phi_conv)
    r_conv = R_t + r_u * (1.0 - np.cos(phi_conv))
    # x_conv[-1] == 0.0, r_conv[-1] == R_t  (throat point, slope = 0)

    # ------------------------------------------------------------------
    # 2.  Downstream throat arc (Sutton Sec. 3.3)
    #     Circular arc of radius r_d = 0.382*R_t from throat (phi=0)
    #     to the bell tangent point at phi = theta_n.
    # ------------------------------------------------------------------
    r_d = 0.382 * R_t
    n_arc = max(8, n_points // 15)
    phi_arc = np.linspace(0.0, theta_n, n_arc)
    x_arc = r_d * np.sin(phi_arc)
    r_arc = R_t + r_d * (1.0 - np.cos(phi_arc))
    # x_arc[-1] == x_N, r_arc[-1] == r_N, slope == tan(theta_n)

    x_N = float(x_arc[-1])
    r_N = float(r_arc[-1])

    # ------------------------------------------------------------------
    # 3.  Divergent bell parabola
    #
    #     Bell length (closed-form, see docstring):
    #         d = 2*(R_e - r_N) / (tan_n + tan_e)
    #
    #     Parabola coefficients:
    #         A = (tan_n - tan_e) / (2*(x_N - x_E))
    #         B = tan_n - 2*A*x_N
    #         C = r_N - A*x_N^2 - B*x_N
    # ------------------------------------------------------------------
    d = 2.0 * (R_e - r_N) / (tan_n + tan_e)
    x_E = x_N + d

    A_par = (tan_n - tan_e) / (2.0 * (x_N - x_E))
    B_par = tan_n - 2.0 * A_par * x_N
    C_par = r_N - A_par * x_N**2 - B_par * x_N

    n_bell = n_points - n_conv - n_arc
    x_bell = np.linspace(x_N, x_E, max(n_bell, 10))
    r_bell = A_par * x_bell**2 + B_par * x_bell + C_par

    # ------------------------------------------------------------------
    # 4.  Concatenate, dropping duplicate junction points.
    #     x_conv[-1] == x_arc[0] == 0.0  (throat)
    #     x_arc[-1]  == x_bell[0] == x_N (arc/parabola junction)
    # ------------------------------------------------------------------
    x_all = np.concatenate([x_conv, x_arc[1:], x_bell[1:]])
    r_all = np.concatenate([r_conv, r_arc[1:], r_bell[1:]])

    return x_all, r_all


# ---------------------------------------------------------------------------
# RaoNozzle dataclass  -- callable r(x), A(x), area_ratio(x) API
# ---------------------------------------------------------------------------


@dataclass
class RaoNozzle:
    """
    Rao parabolic bell nozzle with interpolated r(x) and A(x) accessors.

    Wraps ``rao_contour`` and builds a scipy linear interpolant over 500
    internal points so that r(x) and A(x) can be evaluated at any axial
    station without recomputing the full contour each time.

    The coordinate origin is at the throat: x = 0, r = r_throat.
    The convergent section lies at x < 0; the divergent bell at x > 0.

    Source: Rao, G.V.R. (1958). "Exhaust Nozzle Contour for Optimum Thrust."
            Jet Propulsion, Vol. 28, No. 6, pp. 377-382.
            Sutton, G.P. and Biblarz, O. (2017). "Rocket Propulsion Elements,"
            9th ed., Wiley. Section 3.3, Table 3-2.
    Validity: 5 <= epsilon <= 50, gamma approx 1.1-1.4.

    Default angles theta_n = 25 deg, theta_e = 8 deg are from
    Sutton & Biblarz Table 3-2 and are appropriate for eps ~ 10-20 with an
    approximately 85% bell length.  For other expansion ratios consult
    Rao (1958) Table 1 and pass the appropriate angles explicitly.

    Parameters
    ----------
    r_throat : float
        Throat radius [m].
    epsilon : float
        Nozzle expansion ratio Ae/At (>= 1).
    length_fraction : float
        Nominal bell length as a fraction of the 15 deg equivalent cone
        length (stored for documentation; the actual length is determined
        by theta_n and theta_e).  Default 0.80.
    theta_n : float
        Bell entrance half-angle [deg].  Default 25 deg.
    theta_e : float
        Bell exit half-angle [deg].  Default 8 deg.
    """

    r_throat: float
    epsilon: float
    length_fraction: float = 0.80
    theta_n: float = 25.0
    theta_e: float = 8.0

    def __post_init__(self) -> None:
        if self.r_throat <= 0.0:
            raise ValueError(f"r_throat must be positive, got {self.r_throat}")
        if self.epsilon < 1.0:
            raise ValueError(f"epsilon must be >= 1, got {self.epsilon}")
        if not (5.0 <= self.epsilon <= 50.0):
            warnings.warn(
                f"epsilon = {self.epsilon} outside Rao validity range [5, 50].",
                RuntimeWarning,
                stacklevel=2,
            )

        # Build dense arrays and interpolant (500 points gives sub-micron
        # interpolation error for typical rocket nozzle dimensions).
        self._x_arr, self._r_arr = rao_contour(
            R_t=self.r_throat,
            eps=self.epsilon,
            theta_n_deg=self.theta_n,
            theta_e_deg=self.theta_e,
            n_points=500,
        )
        self._r_interp: interp1d = interp1d(
            self._x_arr,
            self._r_arr,
            kind="linear",
            bounds_error=True,
            assume_sorted=True,
        )

    # ------------------------------------------------------------------
    # Geometry accessors
    # ------------------------------------------------------------------

    @property
    def x_throat(self) -> float:
        """Axial coordinate of the throat [m].  By convention x = 0."""
        return 0.0

    @property
    def x_exit(self) -> float:
        """Axial coordinate of the nozzle exit plane [m]."""
        return float(self._x_arr[-1])

    @property
    def x_inlet(self) -> float:
        """Axial coordinate of the convergent section inlet [m].  x < 0."""
        return float(self._x_arr[0])

    @property
    def A_throat(self) -> float:
        """Throat cross-sectional area [m2]."""
        return math.pi * self.r_throat**2

    def r(self, x: float) -> float:
        """Wall radius at axial station x [m]."""
        return float(self._r_interp(x))

    def A(self, x: float) -> float:
        """Cross-sectional area at axial station x [m2]."""
        return math.pi * self.r(x) ** 2

    def area_ratio(self, x: float) -> float:
        """Local area ratio A(x) / A_throat [-]."""
        return self.A(x) / self.A_throat

    def contour(self, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """Return evenly-spaced (x, r) arrays with n points from inlet to exit."""
        x_out = np.linspace(self.x_inlet, self.x_exit, n)
        r_out = self._r_interp(x_out)
        return x_out, np.asarray(r_out)

    # ------------------------------------------------------------------
    # Geometry diagnostics
    # ------------------------------------------------------------------

    def bell_length_fraction(self) -> float:
        """Actual bell length / equivalent 15-deg cone length [-]."""
        R_e = self.r_throat * math.sqrt(self.epsilon)
        L_cone_15 = (R_e - self.r_throat) / math.tan(math.radians(15.0))
        return self.x_exit / L_cone_15
