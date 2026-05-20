"""
rao.py
======

Rao parabolic bell nozzle contour and throat sizing for the reference engine.

Public API
----------
Module-level:
    - ``DEFAULT_L_STAR`` : dict[str, float] — recommended L* by propellant class.

Functions (backward-compatible):
    - ``throat_area``   : isentropic throat sizing from c*.
    - ``rao_contour``   : Rao parabolic bell nozzle (x, r) arrays in metres.

Dataclass:
    - ``RaoNozzle``     : interpolated r(x), A(x), area_ratio(x) accessors
                          including the L*-derived cylindrical chamber section
                          upstream of the Rao convergent arc.

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

Characteristic length
    Sutton, G.P. and Biblarz, O. (2017). "Rocket Propulsion Elements,"
    9th ed., Wiley. Chapter 8, Table 8-1 (L* design guidance per propellant).
    L* = V_chamber / A_throat, where V_chamber = V_cylinder + V_convergent_arc.

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
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import interp1d  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# Default characteristic lengths by propellant class
# Source: Sutton & Biblarz (2017) Table 8-1.
# ---------------------------------------------------------------------------

DEFAULT_L_STAR: dict[str, float] = {
    "methalox": 1.1,   # m  LCH4 / LOX — similar to propane/oxygen range
    "kerolox":  1.1,   # m  RP-1 / LOX
    "hydrolox": 0.7,   # m  LH2 / LOX (shorter due to very high flame speed)
}


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
    Rao parabolic bell nozzle with L*-derived cylindrical chamber section.

    Geometry (upstream to downstream):
      1. Cylindrical chamber : x_chamber_start ≤ x < x_inlet
                               constant radius r_chamber = r_throat * sqrt(Ac_At)
      2. Convergent arc      : x_inlet ≤ x ≤ 0 (throat)
                               30-degree circular arc, R = 1.5*R_t
      3. Downstream arc      : 0 ≤ x ≤ x_N
                               circular arc R = 0.382*R_t
      4. Divergent bell      : x_N ≤ x ≤ x_exit
                               parabolic contour

    The cylindrical chamber length L_chamber is derived from the
    characteristic length L* (Sutton & Biblarz Ch. 8):

        V_chamber = L* × A_throat
        V_chamber = π·r_chamber² · L_chamber + V_convergent
        L_chamber = (L* · A_throat − V_convergent) / (π · r_chamber²)

    where V_convergent is the volume of the convergent arc numerically
    integrated from x_inlet to the throat.

    Source: Sutton, G.P. and Biblarz, O. (2017). "Rocket Propulsion Elements,"
            9th ed., Wiley. Ch. 8, Table 8-1 — L* design guidance per propellant.
            Huzel & Huang (1992) §4.2 — Ac/At and L* sizing.
    Validity: same as rao_contour (eps 5–50, gamma 1.1–1.4).

    Parameters
    ----------
    r_throat : float
        Throat radius [m].
    epsilon : float
        Nozzle expansion ratio Ae/At (>= 1).
    L_star : float
        Characteristic length [m].  Default 1.1 m — typical methalox / kerolox
        design value per Sutton & Biblarz Table 8-1.  Use DEFAULT_L_STAR to
        look up per propellant class.
    Ac_At : float
        Chamber-to-throat area ratio [-].  Default 4.0 per CLAUDE.md
        reference engine (Ac/At = 4.0).
    length_fraction : float
        Nominal bell length as a fraction of the 15 deg equivalent cone
        length (stored for documentation; actual length is determined by
        theta_n and theta_e).  Default 0.80.
    theta_n : float
        Bell entrance half-angle [deg].  Default 25 deg.
    theta_e : float
        Bell exit half-angle [deg].  Default 8 deg.
    """

    r_throat: float
    epsilon: float
    L_star: float = 1.1
    Ac_At: float = 4.0
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
        if self.L_star <= 0.0:
            raise ValueError(f"L_star must be positive, got {self.L_star}")
        if self.Ac_At < 1.0:
            raise ValueError(f"Ac_At must be >= 1, got {self.Ac_At}")

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
        # Chamber section derived from L*
        # Source: Sutton & Biblarz (2017) Ch. 8, L* definition.
        # ------------------------------------------------------------------
        self._r_chamber: float = self.r_throat * math.sqrt(self.Ac_At)
        self._A_chamber: float = math.pi * self._r_chamber ** 2

        # V_convergent = π ∫[x_inlet to 0] r(x)² dx
        # Integrate using the pre-built contour arrays (convergent section only).
        _conv_mask = self._x_arr <= 0.0
        _x_conv = self._x_arr[_conv_mask]
        _r_conv = self._r_arr[_conv_mask]
        # Ensure we include x=0 (throat) if the mask omits it.
        if len(_x_conv) == 0 or _x_conv[-1] < 0.0:
            _x_conv = np.append(_x_conv, 0.0)
            _r_conv = np.append(_r_conv, self.r_throat)
        self._V_convergent: float = float(
            math.pi * float(np.trapezoid(_r_conv ** 2, _x_conv))
        )

        # L_chamber from L* definition
        self._L_chamber: float = (
            self.L_star * self.A_throat - self._V_convergent
        ) / self._A_chamber

        if self._L_chamber <= 0.0:
            raise ValueError(
                f"L_star={self.L_star:.3f} m is too small for "
                f"Ac_At={self.Ac_At:.2f}: "
                f"computed L_chamber={self._L_chamber:.6f} m <= 0. "
                f"Increase L_star or decrease Ac_At."
            )

        self._x_chamber_start: float = float(self._x_arr[0]) - self._L_chamber

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
        """Axial coordinate of the convergent-arc inlet [m].  x < 0.

        This is the junction between the cylindrical chamber and the
        convergent arc — NOT the upstream face of the chamber.
        Use ``x_chamber_start`` for the upstream face.
        """
        return float(self._x_arr[0])

    @property
    def x_chamber_start(self) -> float:
        """Axial coordinate of the upstream face of the cylindrical chamber [m].

        x_chamber_start = x_inlet − L_chamber < x_inlet < 0.
        """
        return self._x_chamber_start

    @property
    def A_throat(self) -> float:
        """Throat cross-sectional area [m2]."""
        return math.pi * self.r_throat**2

    @property
    def r_chamber(self) -> float:
        """Cylindrical chamber inner wall radius [m].

        r_chamber = r_throat × sqrt(Ac_At).
        """
        return self._r_chamber

    @property
    def A_chamber(self) -> float:
        """Cylindrical chamber cross-sectional area [m²]."""
        return self._A_chamber

    @property
    def L_chamber(self) -> float:
        """Cylindrical chamber axial length [m].

        Derived from L*: L_chamber = (L* · At − V_convergent) / A_chamber.
        """
        return self._L_chamber

    @property
    def V_convergent(self) -> float:
        """Volume of the convergent section (x_inlet to throat) [m³]."""
        return self._V_convergent

    @property
    def V_chamber(self) -> float:
        """Total combustion chamber volume (cylindrical + convergent) [m³].

        V_chamber = L* × A_throat, by definition of L*.
        """
        return self.L_star * self.A_throat

    def r(self, x: float) -> float:
        """Wall radius at axial station x [m].

        Returns r_chamber for x in [x_chamber_start, x_inlet),
        the interpolated Rao contour value for x in [x_inlet, x_exit].
        Raises ValueError outside [x_chamber_start, x_exit].
        """
        x = float(x)
        if x < self._x_chamber_start:
            raise ValueError(
                f"x={x:.6f} m is upstream of x_chamber_start="
                f"{self._x_chamber_start:.6f} m"
            )
        if x < float(self._x_arr[0]):
            return self._r_chamber
        return float(self._r_interp(x))

    def A(self, x: float) -> float:
        """Cross-sectional area at axial station x [m2]."""
        x = float(x)
        if x < self._x_chamber_start:
            raise ValueError(
                f"x={x:.6f} m is upstream of x_chamber_start="
                f"{self._x_chamber_start:.6f} m"
            )
        if x < float(self._x_arr[0]):
            return self._A_chamber
        return math.pi * self.r(x) ** 2

    def area_ratio(self, x: float) -> float:
        """Local area ratio A(x) / A_throat [-]."""
        x = float(x)
        if x < self._x_chamber_start:
            raise ValueError(
                f"x={x:.6f} m is upstream of x_chamber_start="
                f"{self._x_chamber_start:.6f} m"
            )
        if x < float(self._x_arr[0]):
            return self._A_chamber / self.A_throat
        return self.A(x) / self.A_throat

    def contour(self, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """Return evenly-spaced (x, r) arrays with n points from x_chamber_start to exit."""
        x_out = np.linspace(self._x_chamber_start, self.x_exit, n)
        x_inlet = float(self._x_arr[0])
        # Vectorised: r_chamber in the cylindrical section, interpolant in the nozzle.
        r_out = np.where(
            x_out < x_inlet,
            self._r_chamber,
            np.asarray(
                self._r_interp(
                    np.clip(x_out, x_inlet, float(self._x_arr[-1]))
                )
            ).astype(float),
        )
        return x_out, np.asarray(r_out)

    # ------------------------------------------------------------------
    # Geometry diagnostics
    # ------------------------------------------------------------------

    def bell_length_fraction(self) -> float:
        """Actual bell length / equivalent 15-deg cone length [-]."""
        R_e = self.r_throat * math.sqrt(self.epsilon)
        L_cone_15 = (R_e - self.r_throat) / math.tan(math.radians(15.0))
        return self.x_exit / L_cone_15
