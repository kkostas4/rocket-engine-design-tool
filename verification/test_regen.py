"""
test_regen.py
=============

Verification tests for the simplified Bartz function ``bartz_h_simple``
in physics/regen/bartz.py.

For tests of the full Bartz equation with sigma correction see test_bartz.py.

Reference
---------
    Bartz, D.R. (1957). "A Simple Equation for Rapid Estimation of Rocket
    Nozzle Convective Heat Transfer Coefficients."
    Jet Propulsion, Vol. 27, No. 1, pp. 49-51. Eq. (5).
"""

from __future__ import annotations

import pytest

from physics.regen.bartz import bartz_h_simple

# ---------------------------------------------------------------------------
# Reference engine conditions
# ---------------------------------------------------------------------------
D_T = 0.022313  # throat diameter [m]
PC = 6.8948e6  # chamber pressure [Pa]
CSTAR = 1893.3  # c* [m/s]
EPS = 10.605  # expansion ratio

# Representative gas properties at throat for LCH4/LOX at Pc~1000 psia:
# mu, cp, Pr, gamma are approximations -- not from live CEA.
MU = 8.0e-5  # dynamic viscosity [Pa.s]
CP = 2000.0  # cp [J/(kg.K)]
PR = 0.60  # Prandtl number
GAMMA = 1.15  # ratio of specific heats
R_C = D_T / 2.0  # throat curvature radius [m]  (Bartz 1957, p.50)


class TestBartzThroat:
    def test_h_physical_range(self) -> None:
        """h at throat is in physically plausible range 5,000-200,000 W/(m2.K)."""
        h = bartz_h_simple(
            D_t=D_T,
            Pc=PC,
            c_star=CSTAR,
            A_ratio=1.0,
            gamma=GAMMA,
            Pr=PR,
            mu=MU,
            cp=CP,
            r_c=R_C,
        )
        assert 5_000 < h < 200_000, f"h_throat = {h:.1f} outside expected range"

    def test_h_throat_gt_h_exit(self) -> None:
        """h at throat > h at exit (A_ratio scaling: h ~ (1/A_ratio)^0.9)."""
        h_throat = bartz_h_simple(D_T, PC, CSTAR, 1.0, GAMMA, PR, MU, CP, R_C)
        h_exit = bartz_h_simple(D_T, PC, CSTAR, EPS, GAMMA, PR, MU, CP, R_C)
        assert h_throat > h_exit

    def test_area_ratio_scaling(self) -> None:
        """h scales as (1/A_ratio)^0.9 -- spot-check at A=1 vs A=2."""
        h1 = bartz_h_simple(D_T, PC, CSTAR, 1.0, GAMMA, PR, MU, CP, R_C)
        h2 = bartz_h_simple(D_T, PC, CSTAR, 2.0, GAMMA, PR, MU, CP, R_C)
        ratio_expected = 2.0**0.9
        ratio_actual = h1 / h2
        assert abs(ratio_actual - ratio_expected) / ratio_expected < 1e-6


class TestBartzSensitivity:
    def test_h_increases_with_pc(self) -> None:
        """Higher Pc -> higher h (h ~ Pc^0.8 in simplified Bartz)."""
        h_lo = bartz_h_simple(D_T, PC * 0.5, CSTAR, 1.0, GAMMA, PR, MU, CP, R_C)
        h_hi = bartz_h_simple(D_T, PC * 1.5, CSTAR, 1.0, GAMMA, PR, MU, CP, R_C)
        assert h_hi > h_lo

    def test_h_increases_with_mu(self) -> None:
        """Higher mu -> higher h (h ~ mu^0.2 in simplified Bartz)."""
        h_lo = bartz_h_simple(D_T, PC, CSTAR, 1.0, GAMMA, PR, MU * 0.5, CP, R_C)
        h_hi = bartz_h_simple(D_T, PC, CSTAR, 1.0, GAMMA, PR, MU * 2.0, CP, R_C)
        assert h_hi > h_lo

    def test_invalid_a_ratio_raises(self) -> None:
        with pytest.raises(ValueError):
            bartz_h_simple(D_T, PC, CSTAR, -1.0, GAMMA, PR, MU, CP, R_C)

    def test_invalid_r_c_raises(self) -> None:
        with pytest.raises(ValueError):
            bartz_h_simple(D_T, PC, CSTAR, 1.0, GAMMA, PR, MU, CP, 0.0)
