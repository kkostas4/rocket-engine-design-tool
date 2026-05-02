"""
test_regen.py
=============

Verification tests for physics/regen/bartz.py.

Tests the Bartz (1957) simplified heat-transfer coefficient using
reference-engine conditions at the throat and a spot-check at the exit
to confirm the A_ratio scaling.

Gas property values used
------------------------
These are representative combustion-gas values at the throat for LCH4/LOX
at Pc ≈ 1,000 psia, O/F ≈ 3.6.  They are not from a live CEA call; a proper
regen analysis will substitute accurate CEA-derived properties.

    mu  = 8.0e-5 Pa·s   (dynamic viscosity — typical for methalox ~3500 K)
    cp  = 2000 J/(kg·K) (specific heat — approximate equilibrium value)
    Pr  = 0.60          (Prandtl number — typical hot combustion gas)
    gamma = 1.15        (specific heat ratio — approximate)
    r_c = D_t / 2       (throat wall curvature radius — Bartz 1957, p.50)

Reference
---------
    Bartz, D.R. (1957). "A Simple Equation for Rapid Estimation of Rocket
    Nozzle Convective Heat Transfer Coefficients."
    Jet Propulsion, Vol. 27, No. 1, pp. 49–51. Eq. (5).
"""

from __future__ import annotations

import pytest

from physics.regen.bartz import bartz_h

# ---------------------------------------------------------------------------
# Reference engine conditions
# ---------------------------------------------------------------------------
D_T = 0.022313          # throat diameter [m]
PC = 6.8948e6           # chamber pressure [Pa]
CSTAR = 1893.3          # c* [m/s]
EPS = 10.605            # expansion ratio

# Representative gas properties at throat (see module docstring)
MU = 8.0e-5             # dynamic viscosity [Pa·s]
CP = 2000.0             # cp [J/(kg·K)]
PR = 0.60               # Prandtl number
GAMMA = 1.15            # ratio of specific heats
R_C = D_T / 2.0         # throat curvature radius [m]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestBartzThroat:
    def test_h_physical_range(self) -> None:
        """h at throat is in physically plausible range 5,000–200,000 W/(m²·K)."""
        h = bartz_h(
            D_t=D_T, Pc=PC, c_star=CSTAR,
            A_ratio=1.0, gamma=GAMMA,
            Pr=PR, mu=MU, cp=CP, r_c=R_C,
        )
        assert 5_000 < h < 200_000, (
            f"h_throat = {h:.1f} W/(m²·K) outside expected range 5,000–200,000"
        )

    def test_h_throat_gt_h_exit(self) -> None:
        """h at throat > h at exit (A_ratio scaling: h ∝ (1/A_ratio)^0.9)."""
        h_throat = bartz_h(
            D_t=D_T, Pc=PC, c_star=CSTAR,
            A_ratio=1.0, gamma=GAMMA,
            Pr=PR, mu=MU, cp=CP, r_c=R_C,
        )
        h_exit = bartz_h(
            D_t=D_T, Pc=PC, c_star=CSTAR,
            A_ratio=EPS, gamma=GAMMA,
            Pr=PR, mu=MU, cp=CP, r_c=R_C,
        )
        assert h_throat > h_exit, (
            f"h_throat = {h_throat:.1f}, h_exit = {h_exit:.1f}: "
            "throat h must exceed exit h"
        )

    def test_area_ratio_scaling(self) -> None:
        """h scales as (1/A_ratio)^0.9 — test at two stations."""
        h1 = bartz_h(D_T, PC, CSTAR, 1.0, GAMMA, PR, MU, CP, R_C)
        h2 = bartz_h(D_T, PC, CSTAR, 2.0, GAMMA, PR, MU, CP, R_C)
        # Expected ratio: (1/1)^0.9 / (1/2)^0.9 = 2^0.9 ≈ 1.866
        ratio_expected = 2.0**0.9
        ratio_actual = h1 / h2
        dev = abs(ratio_actual - ratio_expected) / ratio_expected
        assert dev < 1e-6, (
            f"h(A=1)/h(A=2) = {ratio_actual:.6f}, expected {ratio_expected:.6f}"
        )


class TestBartzSensitivity:
    def test_h_increases_with_pc(self) -> None:
        """Higher Pc → higher h (at fixed c*, h ∝ Pc^0.8)."""
        h_lo = bartz_h(D_T, PC * 0.5, CSTAR, 1.0, GAMMA, PR, MU, CP, R_C)
        h_hi = bartz_h(D_T, PC * 1.5, CSTAR, 1.0, GAMMA, PR, MU, CP, R_C)
        assert h_hi > h_lo

    def test_h_increases_with_mu(self) -> None:
        """Higher mu → higher h (h ∝ mu^0.2 in simplified Bartz)."""
        h_lo = bartz_h(D_T, PC, CSTAR, 1.0, GAMMA, PR, MU * 0.5, CP, R_C)
        h_hi = bartz_h(D_T, PC, CSTAR, 1.0, GAMMA, PR, MU * 2.0, CP, R_C)
        assert h_hi > h_lo

    def test_invalid_a_ratio_raises(self) -> None:
        with pytest.raises(ValueError):
            bartz_h(D_T, PC, CSTAR, -1.0, GAMMA, PR, MU, CP, R_C)

    def test_invalid_r_c_raises(self) -> None:
        with pytest.raises(ValueError):
            bartz_h(D_T, PC, CSTAR, 1.0, GAMMA, PR, MU, CP, 0.0)
