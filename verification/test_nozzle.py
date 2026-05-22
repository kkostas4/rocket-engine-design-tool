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

from noyron.rp.nozzle.rao import RaoNozzle, rao_contour, throat_area

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
        """contour(n) returns two arrays of length n, spanning x_chamber_start to x_exit."""
        x, r = nozzle.contour(150)
        assert len(x) == 150
        assert len(r) == 150
        assert x[0] == pytest.approx(nozzle.x_chamber_start)
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


# ---------------------------------------------------------------------------
# Chamber geometry (L*-derived barrel section)
# ---------------------------------------------------------------------------
class TestChamberGeometry:
    """
    Tests for the L*-derived cylindrical chamber section on RaoNozzle.

    Reference: Sutton & Biblarz (2017) Ch. 8, Table 8-1.
    CLAUDE.md frozen baseline: Ac/At = 4.0, D_chamber = 44.625 mm,
    L_chamber = 125 mm.
    """

    @pytest.fixture(scope="class")
    def nozzle(self) -> RaoNozzle:
        return RaoNozzle(r_throat=DT_REF / 2.0, epsilon=EPS_REF)

    def test_r_chamber_matches_contraction_ratio(self, nozzle: RaoNozzle) -> None:
        """r_chamber = r_throat * sqrt(Ac_At) → D_chamber = 44.625 mm (CLAUDE.md)."""
        r_ch_ref = (DT_REF / 2.0) * math.sqrt(4.0)  # r_throat * 2
        dev = abs(nozzle.r_chamber - r_ch_ref) / r_ch_ref
        assert dev < 1e-9, (
            f"r_chamber = {nozzle.r_chamber*1e3:.6f} mm, "
            f"expected {r_ch_ref*1e3:.6f} mm"
        )

    def test_L_star_volume_identity(self, nozzle: RaoNozzle) -> None:
        """L* identity: L_chamber*A_chamber + V_convergent == L_star*A_throat within 2 %."""
        V_reconstructed = nozzle.L_chamber * nozzle.A_chamber + nozzle.V_convergent
        V_ref = nozzle.L_star * nozzle.A_throat
        dev = abs(V_reconstructed - V_ref) / V_ref
        assert dev < 0.02, (
            f"L* volume identity error {100*dev:.4f} %: "
            f"got {V_reconstructed*1e6:.4f} cm³, "
            f"expected {V_ref*1e6:.4f} cm³"
        )

    def test_V_chamber_equals_L_star_times_A_throat(self, nozzle: RaoNozzle) -> None:
        """V_chamber property returns L_star * A_throat."""
        assert nozzle.V_chamber == pytest.approx(nozzle.L_star * nozzle.A_throat)

    def test_chamber_radius_constant_in_barrel(self, nozzle: RaoNozzle) -> None:
        """r(x) == r_chamber for all x in the cylindrical section."""
        x_vals = np.linspace(nozzle.x_chamber_start, nozzle.x_inlet - 1e-6, 5)
        for x in x_vals:
            r = nozzle.r(float(x))
            assert abs(r - nozzle.r_chamber) / nozzle.r_chamber < 1e-9, (
                f"r({x:.6f}) = {r:.9f}, expected r_chamber = {nozzle.r_chamber:.9f}"
            )

    def test_x_chamber_start_upstream_of_convergent(self, nozzle: RaoNozzle) -> None:
        """x_chamber_start < x_inlet < 0 < x_exit."""
        assert nozzle.x_chamber_start < nozzle.x_inlet < 0.0 < nozzle.x_exit

    def test_V_chamber_positive(self, nozzle: RaoNozzle) -> None:
        """Total combustion chamber volume is positive."""
        assert nozzle.V_chamber > 0.0

    def test_L_star_zero_raises(self) -> None:
        """L_star <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="L_star"):
            RaoNozzle(r_throat=DT_REF / 2.0, epsilon=EPS_REF, L_star=0.0)

    def test_Ac_At_below_one_raises(self) -> None:
        """Ac_At < 1 raises ValueError."""
        with pytest.raises(ValueError, match="Ac_At"):
            RaoNozzle(r_throat=DT_REF / 2.0, epsilon=EPS_REF, Ac_At=0.5)
