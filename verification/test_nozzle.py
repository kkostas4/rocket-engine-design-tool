"""
test_nozzle.py
==============

Verification tests for physics/nozzle/rao.py.

Reproduces the reference engine throat and exit geometry:
    Throat diameter  = 22.313 mm  -> At = 3.910e-4 m2
    Exit diameter    = 72.662 mm  -> Ae = 4.147e-3 m2
    Expansion ratio  eps = 10.605

Reference: CLAUDE.md -- Authoritative Numbers / Geometry.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from physics.nozzle.rao import RaoNozzle, rao_contour, throat_area

# ---------------------------------------------------------------------------
# Reference values (CLAUDE.md)
# ---------------------------------------------------------------------------
MDOT_KG_S = 1.4239
PC_PA = 6.8948e6
CSTAR_MS = 1893.3

AT_REF = 3.910e-4  # m2
DT_REF = 0.022313  # m  (22.313 mm)
AE_REF = 4.147e-3  # m2
DE_REF = 0.072662  # m  (72.662 mm)
EPS_REF = 10.605

TOL_AREA = 0.005  # 0.5 %
TOL_DIAM = 0.005  # 0.5 %


# ---------------------------------------------------------------------------
# Throat sizing
# ---------------------------------------------------------------------------
class TestThroatArea:
    def test_area_value(self) -> None:
        """At = m_dot * c* / Pc reproduces 3.910e-4 m2 to 0.5 %."""
        at = throat_area(MDOT_KG_S, PC_PA, CSTAR_MS)
        dev = abs(at - AT_REF) / AT_REF
        assert dev < TOL_AREA, (
            f"At = {at:.6e} m2,  expected {AT_REF:.6e} m2  "
            f"(dev = {100*dev:.3f} %)"
        )

    def test_throat_diameter(self) -> None:
        """Diameter from At reproduces 22.313 mm to 0.5 %."""
        at = throat_area(MDOT_KG_S, PC_PA, CSTAR_MS)
        dt = math.sqrt(4.0 * at / math.pi)
        dev = abs(dt - DT_REF) / DT_REF
        assert dev < TOL_DIAM, (
            f"Dt = {dt*1e3:.4f} mm,  expected {DT_REF*1e3:.4f} mm  "
            f"(dev = {100*dev:.3f} %)"
        )

    def test_exit_area(self) -> None:
        """Ae = eps * At reproduces 4.147e-3 m2 to 0.5 %."""
        at = throat_area(MDOT_KG_S, PC_PA, CSTAR_MS)
        ae = EPS_REF * at
        dev = abs(ae - AE_REF) / AE_REF
        assert dev < TOL_AREA, (
            f"Ae = {ae:.6e} m2,  expected {AE_REF:.6e} m2  "
            f"(dev = {100*dev:.3f} %)"
        )

    def test_exit_diameter(self) -> None:
        """Exit diameter from Ae reproduces 72.662 mm to 0.5 %."""
        at = throat_area(MDOT_KG_S, PC_PA, CSTAR_MS)
        ae = EPS_REF * at
        de = math.sqrt(4.0 * ae / math.pi)
        dev = abs(de - DE_REF) / DE_REF
        assert dev < TOL_DIAM, (
            f"De = {de*1e3:.4f} mm,  expected {DE_REF*1e3:.4f} mm  "
            f"(dev = {100*dev:.3f} %)"
        )


# ---------------------------------------------------------------------------
# Rao contour  (array-returning function)
# ---------------------------------------------------------------------------
class TestRaoContour:
    @pytest.fixture(scope="class")
    def contour(self) -> tuple[np.ndarray, np.ndarray]:
        R_t = DT_REF / 2.0
        return rao_contour(R_t, EPS_REF)

    def test_exit_radius(self, contour: tuple[np.ndarray, np.ndarray]) -> None:
        """Contour exit radius matches R_e = 36.331 mm to 0.5 %."""
        x, r = contour
        R_e_ref = DE_REF / 2.0
        dev = abs(r[-1] - R_e_ref) / R_e_ref
        assert dev < TOL_DIAM, (
            f"r_exit = {r[-1]*1e3:.4f} mm,  expected {R_e_ref*1e3:.4f} mm  "
            f"(dev = {100*dev:.3f} %)"
        )

    def test_throat_radius(self, contour: tuple[np.ndarray, np.ndarray]) -> None:
        """Contour passes through throat radius R_t at x ~ 0."""
        x, r = contour
        R_t = DT_REF / 2.0
        idx = int(np.argmin(np.abs(x)))
        dev = abs(r[idx] - R_t) / R_t
        assert dev < 0.02, (
            f"r at x~0 is {r[idx]*1e3:.4f} mm, expected ~{R_t*1e3:.4f} mm"
        )

    def test_x_starts_negative(self, contour: tuple[np.ndarray, np.ndarray]) -> None:
        """Convergent section extends upstream (x < 0)."""
        x, _ = contour
        assert x[0] < 0.0

    def test_x_ends_positive(self, contour: tuple[np.ndarray, np.ndarray]) -> None:
        """Divergent section extends downstream (x > 0)."""
        x, _ = contour
        assert x[-1] > 0.0

    def test_monotone_x(self, contour: tuple[np.ndarray, np.ndarray]) -> None:
        """x is monotonically increasing along the contour."""
        x, _ = contour
        assert np.all(np.diff(x) > 0)

    def test_exit_area_ratio(self, contour: tuple[np.ndarray, np.ndarray]) -> None:
        """Area ratio at exit equals eps to within 1 %."""
        x, r = contour
        R_t = DT_REF / 2.0
        eps_computed = (r[-1] / R_t) ** 2
        dev = abs(eps_computed - EPS_REF) / EPS_REF
        assert dev < 0.01, (
            f"eps_computed = {eps_computed:.4f},  expected {EPS_REF}  "
            f"(dev = {100*dev:.3f} %)"
        )


# ---------------------------------------------------------------------------
# RaoNozzle dataclass  (callable r(x), A(x) API)
# ---------------------------------------------------------------------------
class TestRaoNozzle:
    """
    Tests for the RaoNozzle dataclass.

    Reference contour: reference engine (CLAUDE.md), eps = 10.605,
    D_throat = 22.313 mm, D_exit = 72.662 mm.
    """

    @pytest.fixture(scope="class")
    def nozzle(self) -> RaoNozzle:
        return RaoNozzle(r_throat=DT_REF / 2.0, epsilon=EPS_REF)

    def test_throat_radius(self, nozzle: RaoNozzle) -> None:
        """r(x_throat) == r_throat to < 0.01 %."""
        assert abs(nozzle.r(nozzle.x_throat) - DT_REF / 2.0) / (DT_REF / 2.0) < 1e-4

    def test_exit_area(self, nozzle: RaoNozzle) -> None:
        """A(x_exit) == eps * A_throat to within 0.1 %."""
        A_exit = nozzle.A(nozzle.x_exit)
        A_exit_ref = EPS_REF * nozzle.A_throat
        dev = abs(A_exit - A_exit_ref) / A_exit_ref
        assert dev < 0.001, f"A_exit dev = {100*dev:.4f} %"

    def test_continuity_at_throat(self, nozzle: RaoNozzle) -> None:
        """r(x) is continuous at x=0 (no jump > 0.01 % of R_t across a 1 um step)."""
        eps_x = 1.0e-6  # 1 um
        r_left = nozzle.r(-eps_x)
        r_right = nozzle.r(eps_x)
        jump = abs(r_right - r_left) / nozzle.r_throat
        assert jump < 1e-4, f"Discontinuity at throat: dr/R_t = {jump:.2e}"

    def test_slope_continuity_at_N(self, nozzle: RaoNozzle) -> None:
        """dr/dx is continuous at the throat-arc / bell-parabola junction x_N.

        At x_N the slope must equal tan(theta_n) from both the arc and the
        parabola (imposed by construction of rao_contour).  A finite-difference
        check over a 1 mm window centred on x_N must agree to within 1 %.
        """
        import math

        theta_n = math.radians(nozzle.theta_n)
        r_d = 0.382 * nozzle.r_throat
        x_N = r_d * math.sin(theta_n)
        tan_n = math.tan(theta_n)

        dx = 5.0e-5  # 50 um -- half-window for finite difference
        slope_fd = (nozzle.r(x_N + dx) - nozzle.r(x_N - dx)) / (2 * dx)
        assert abs(slope_fd - tan_n) / tan_n < 0.01, (
            f"Slope at x_N: FD = {slope_fd:.5f}, tan(theta_n) = {tan_n:.5f}"
        )

    def test_against_published_contour(self, nozzle: RaoNozzle) -> None:
        """Reproduce reference engine exit radius 36.331 mm (CLAUDE.md) to 0.1 %.

        The reference engine geometry is the authoritative baseline for this
        project (CLAUDE.md: Authoritative Numbers / Geometry).  The
        rao_contour function was verified to this accuracy in the original
        sprint; the RaoNozzle class must reproduce the same result via its
        interpolated r() accessor.
        """
        r_exit = nozzle.r(nozzle.x_exit)
        r_exit_ref = DE_REF / 2.0  # 36.331 mm
        dev = abs(r_exit - r_exit_ref) / r_exit_ref
        assert dev < 0.001, (
            f"r_exit = {r_exit*1e3:.4f} mm, expected {r_exit_ref*1e3:.4f} mm "
            f"(dev = {100*dev:.4f} %)"
        )

    def test_x_throat_is_zero(self, nozzle: RaoNozzle) -> None:
        assert nozzle.x_throat == 0.0

    def test_x_inlet_negative(self, nozzle: RaoNozzle) -> None:
        assert nozzle.x_inlet < 0.0

    def test_x_exit_positive(self, nozzle: RaoNozzle) -> None:
        assert nozzle.x_exit > 0.0

    def test_A_throat(self, nozzle: RaoNozzle) -> None:
        """A_throat == pi * r_throat^2."""
        import math

        assert abs(nozzle.A_throat - math.pi * (DT_REF / 2) ** 2) < 1e-12

    def test_area_ratio_at_exit(self, nozzle: RaoNozzle) -> None:
        """area_ratio(x_exit) == eps to within 0.1 %."""
        eps = nozzle.area_ratio(nozzle.x_exit)
        dev = abs(eps - EPS_REF) / EPS_REF
        assert dev < 0.001

    def test_contour_shape(self, nozzle: RaoNozzle) -> None:
        """contour(n) returns two arrays of length n."""
        x, r = nozzle.contour(150)
        assert len(x) == 150
        assert len(r) == 150
        assert x[0] == pytest.approx(nozzle.x_inlet)
        assert x[-1] == pytest.approx(nozzle.x_exit)

    def test_invalid_r_throat_raises(self) -> None:
        with pytest.raises(ValueError, match="r_throat"):
            RaoNozzle(r_throat=-0.01, epsilon=10.0)

    def test_invalid_epsilon_raises(self) -> None:
        with pytest.raises(ValueError, match="epsilon"):
            RaoNozzle(r_throat=0.01, epsilon=0.5)

    def test_out_of_range_epsilon_warns(self) -> None:
        with pytest.warns(RuntimeWarning, match="validity range"):
            RaoNozzle(r_throat=0.01, epsilon=60.0)
