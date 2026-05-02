"""
test_nozzle.py
==============

Verification tests for physics/nozzle/rao.py.

Reproduces the reference engine throat and exit geometry:
    Throat diameter  = 22.313 mm  → At = 3.910e-4 m²
    Exit diameter    = 72.662 mm  → Ae = 4.147e-3 m²
    Expansion ratio  ε = 10.605

Reference: CLAUDE.md — Authoritative Numbers / Geometry.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from physics.nozzle.rao import rao_contour, throat_area

# ---------------------------------------------------------------------------
# Reference values (CLAUDE.md)
# ---------------------------------------------------------------------------
MDOT_KG_S = 1.4239
PC_PA = 6.8948e6
CSTAR_MS = 1893.3

AT_REF = 3.910e-4       # m²
DT_REF = 0.022313       # m  (22.313 mm)
AE_REF = 4.147e-3       # m²
DE_REF = 0.072662       # m  (72.662 mm)
EPS_REF = 10.605

TOL_AREA = 0.005        # 0.5 %
TOL_DIAM = 0.005        # 0.5 %


# ---------------------------------------------------------------------------
# Throat sizing
# ---------------------------------------------------------------------------
class TestThroatArea:
    def test_area_value(self) -> None:
        """At = m_dot * c* / Pc reproduces 3.910e-4 m² to 0.5 %."""
        at = throat_area(MDOT_KG_S, PC_PA, CSTAR_MS)
        dev = abs(at - AT_REF) / AT_REF
        assert dev < TOL_AREA, (
            f"At = {at:.6e} m²,  expected {AT_REF:.6e} m²  "
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
        """Ae = ε * At reproduces 4.147e-3 m² to 0.5 %."""
        at = throat_area(MDOT_KG_S, PC_PA, CSTAR_MS)
        ae = EPS_REF * at
        dev = abs(ae - AE_REF) / AE_REF
        assert dev < TOL_AREA, (
            f"Ae = {ae:.6e} m²,  expected {AE_REF:.6e} m²  "
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
# Rao contour
# ---------------------------------------------------------------------------
class TestRaoContour:
    @pytest.fixture(scope="class")
    def contour(self) -> tuple[np.ndarray, np.ndarray]:
        R_t = DT_REF / 2.0
        return rao_contour(R_t, EPS_REF)

    def test_exit_radius(
        self, contour: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Contour exit radius matches R_e = 36.331 mm to 0.5 %."""
        x, r = contour
        R_e_ref = DE_REF / 2.0
        dev = abs(r[-1] - R_e_ref) / R_e_ref
        assert dev < TOL_DIAM, (
            f"r_exit = {r[-1]*1e3:.4f} mm,  expected {R_e_ref*1e3:.4f} mm  "
            f"(dev = {100*dev:.3f} %)"
        )

    def test_throat_radius(
        self, contour: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Contour passes through throat radius R_t at x ~ 0."""
        x, r = contour
        R_t = DT_REF / 2.0
        # Find point closest to x = 0
        idx = int(np.argmin(np.abs(x)))
        dev = abs(r[idx] - R_t) / R_t
        assert dev < 0.02, (  # 2 % — throat arc offset is expected
            f"r at x≈0 is {r[idx]*1e3:.4f} mm, expected ~{R_t*1e3:.4f} mm"
        )

    def test_x_starts_negative(
        self, contour: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Convergent section extends upstream (x < 0)."""
        x, _ = contour
        assert x[0] < 0.0, "First point should be upstream of throat"

    def test_x_ends_positive(
        self, contour: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Divergent section extends downstream (x > 0)."""
        x, _ = contour
        assert x[-1] > 0.0, "Last point should be downstream of throat"

    def test_monotone_x(
        self, contour: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """x is monotonically increasing along the contour."""
        x, _ = contour
        assert np.all(np.diff(x) > 0), "x must be strictly increasing"

    def test_exit_area_ratio(
        self, contour: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Area ratio at exit equals ε to within 1 %."""
        x, r = contour
        R_t = DT_REF / 2.0
        eps_computed = (r[-1] / R_t) ** 2
        dev = abs(eps_computed - EPS_REF) / EPS_REF
        assert dev < 0.01, (
            f"ε_computed = {eps_computed:.4f},  expected {EPS_REF}  "
            f"(dev = {100*dev:.3f} %)"
        )
