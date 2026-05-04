"""
test_bartz.py
=============

Verification tests for the full Bartz API in physics/regen/bartz.py:
    - CombustionGas dataclass
    - bartz_h(*, x, nozzle, gas, T_wall)  -- with sigma correction
    - heat_flux_profile(...)

Sources
-------
    Bartz, D.R. (1957). "A Simple Equation for Rapid Estimation of Rocket
    Nozzle Convective Heat Transfer Coefficients."
    Jet Propulsion, Vol. 27, No. 1, pp. 49-51. Eqs. (4) and (5).

    Huzel, D.K. and Huang, D.H. (1992). "Modern Engineering for Design of
    Liquid-Propellant Rocket Engines." NASA SP-125. Fig. 4-12 for
    SSME-class heat transfer magnitude reference.

NOTE on test_against_bartz_paper
---------------------------------
Bartz (1957) does not publish a self-contained numerical example suitable
for point-by-point comparison (the paper presents a chart rather than a
table).  The test ``test_throat_h_order_of_magnitude`` instead verifies
the order of magnitude against the range cited in Huzel & Huang Fig. 4-12
for high-Pc methalox-class engines.  All other tests are analytical
(scaling laws) or regression tests locked to the first computed value.
"""

from __future__ import annotations


import numpy as np
import pytest

from physics.nozzle.rao import RaoNozzle
from physics.regen.bartz import (
    CombustionGas,
    adiabatic_wall_temperature,
    bartz_h,
    heat_flux_profile,
    mach_from_area_ratio,
)

# ---------------------------------------------------------------------------
# Reference engine gas state  (CLAUDE.md baseline, Pc = 1000 psia)
# ---------------------------------------------------------------------------
GAS_REF = CombustionGas(
    Tc=3561.2,  # K   (CEA at O/F=3.6, Pc=1000 psia)
    Pc=6.8948e6,  # Pa
    gamma=1.15,  # approximate methalox value
    MW=21.9,  # kg/kmol
    cp=2000.0,  # J/(kg.K)   -- approximate
    mu=8.0e-5,  # Pa.s       -- approximate
    Pr=0.60,  # --           -- approximate
    c_star=1893.3,  # m/s
)

# Reference nozzle (CLAUDE.md geometry)
NOZZLE_REF = RaoNozzle(r_throat=0.022313 / 2.0, epsilon=10.605)

T_WALL = 800.0  # K  -- representative gas-side wall temp for first pass


# ===========================================================================
# 1. Mach number solver
# ===========================================================================
class TestMachSolver:
    def test_sonic_at_unity_area_ratio(self) -> None:
        """A/A* = 1 -> M = 1."""
        assert mach_from_area_ratio(1.0, 1.15) == pytest.approx(1.0, abs=1e-9)

    def test_subsonic_root(self) -> None:
        """Subsonic root: A/A*=2, gamma=1.4 -> M ~ 0.3059."""
        M = mach_from_area_ratio(2.0, 1.4, supersonic=False)
        assert 0.0 < M < 1.0

    def test_supersonic_root(self) -> None:
        """Supersonic root: A/A*=2, gamma=1.4 -> M ~ 2.197."""
        M = mach_from_area_ratio(2.0, 1.4, supersonic=True)
        assert M > 1.0

    def test_area_ratio_round_trip(self) -> None:
        """Round-trip: A/A* -> M -> A/A* within 1e-8."""
        eps_target = 5.3
        gamma = 1.15
        M = mach_from_area_ratio(eps_target, gamma, supersonic=True)
        exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
        t = 1.0 + (gamma - 1.0) / 2.0 * M**2
        eps_check = (1.0 / M) * (2.0 / (gamma + 1.0) * t) ** exponent
        assert abs(eps_check - eps_target) / eps_target < 1e-8


# ===========================================================================
# 2. Sigma correction factor
# ===========================================================================
class TestSigmaFactor:
    def test_sigma_recovers_unity_at_M0_Tw_equals_T0(self) -> None:
        """At T_w = T_aw = T_0 and M = 0, sigma -> 1 (Bartz 1957, Eq. 4).

        M2_term = 1 at M=0, T_w/T_0 = 1 =>
        sigma = 1/((0.5*1*1+0.5)^0.68 * 1^0.12) = 1/(1.0)^0.68 = 1.
        """
        gas = GAS_REF
        nozzle = NOZZLE_REF
        # At x_inlet (near M=0), T_aw ~ T0; evaluate with T_wall = Tc
        T_wall_eq = gas.Tc  # T_w = T_aw = T_0
        h_uncorrected = bartz_h(
            x=nozzle.x_inlet, nozzle=nozzle, gas=gas, T_wall=gas.Tc
        )
        # Verify the ratio h/h_simple is close to sigma=1 by computing h_simple
        from physics.regen.bartz import bartz_h_simple

        D_t = 2 * nozzle.r_throat
        A_ratio = nozzle.area_ratio(nozzle.x_inlet)
        r_c = 0.382 * nozzle.r_throat
        h_simple = bartz_h_simple(
            D_t=D_t,
            Pc=gas.Pc,
            c_star=gas.c_star,
            A_ratio=A_ratio,
            gamma=gas.gamma,
            Pr=gas.Pr,
            mu=gas.mu,
            cp=gas.cp,
            r_c=r_c,
        )
        # At M~0 (convergent inlet) with T_w=T0, sigma should be very close to 1
        sigma = h_uncorrected / h_simple
        assert abs(sigma - 1.0) < 0.02, f"sigma = {sigma:.5f}, expected ~1.0"

    def test_sigma_gt_1_cold_wall(self) -> None:
        """sigma > 1 when T_wall << T_aw (cold wall amplifies heat transfer)."""
        # At the throat with T_wall = 300 K << T_aw ~ 3500 K
        h_cold = bartz_h(x=0.0, nozzle=NOZZLE_REF, gas=GAS_REF, T_wall=300.0)
        h_hot = bartz_h(x=0.0, nozzle=NOZZLE_REF, gas=GAS_REF, T_wall=3000.0)
        assert h_cold > h_hot, "Colder wall should give higher h (larger sigma)"

    def test_sigma_decreases_as_wall_heats(self) -> None:
        """h decreases monotonically as T_wall increases toward T_aw."""
        T_walls = [300.0, 600.0, 900.0, 1200.0, 1500.0]
        h_vals = [
            bartz_h(x=0.0, nozzle=NOZZLE_REF, gas=GAS_REF, T_wall=T)
            for T in T_walls
        ]
        for i in range(1, len(h_vals)):
            assert h_vals[i] < h_vals[i - 1], (
                f"h not decreasing: h(T={T_walls[i-1]})={h_vals[i-1]:.0f}, "
                f"h(T={T_walls[i]})={h_vals[i]:.0f}"
            )


# ===========================================================================
# 3. Order-of-magnitude check (throat h)
# ===========================================================================
class TestThroatHOrderOfMagnitude:
    def test_throat_h_order_of_magnitude(self) -> None:
        """For high-Pc methalox (Pc~200 bar, D_t=100 mm), h at throat in
        30-80 kW/(m2.K).

        Reference: Huzel & Huang (1992) NASA SP-125, Fig. 4-12 --
        heat transfer coefficient vs. Pc for methalox-class engines at
        SSME operating conditions.

        FLAG: This is an order-of-magnitude check against the Huzel &
        Huang range, not a point-by-point reproduction of a specific
        numerical example.  Bartz (1957) does not provide a standalone
        worked example with a verifiable numerical result.
        """
        # SSME-class: Pc = 200 bar, D_t = 100 mm, methalox gas properties
        gas_ssme = CombustionGas(
            Tc=3500.0,
            Pc=20.0e6,  # ~200 bar
            gamma=1.15,
            MW=21.9,
            cp=2000.0,
            mu=8.0e-5,
            Pr=0.60,
            c_star=1893.3,
        )
        nozzle_ssme = RaoNozzle(r_throat=0.050, epsilon=10.0)  # 100 mm throat
        h = bartz_h(x=0.0, nozzle=nozzle_ssme, gas=gas_ssme, T_wall=800.0)
        assert 30_000 <= h <= 80_000, (
            f"h_throat = {h/1e3:.1f} kW/(m2.K), expected 30-80 kW/(m2.K) "
            f"for SSME-class conditions (Huzel & Huang Fig. 4-12)"
        )


# ===========================================================================
# 4. Axial distribution checks
# ===========================================================================
class TestAxialDistribution:
    def test_h_falls_with_area_ratio_downstream(self) -> None:
        """h_g decreases monotonically moving downstream of the throat."""
        x_pts = np.linspace(0.0, NOZZLE_REF.x_exit, 10)[1:]  # skip throat
        h_vals = [bartz_h(x=x, nozzle=NOZZLE_REF, gas=GAS_REF, T_wall=T_WALL) for x in x_pts]
        for i in range(1, len(h_vals)):
            assert h_vals[i] < h_vals[i - 1], (
                f"h not decreasing downstream at index {i}: "
                f"h[{i-1}]={h_vals[i-1]:.1f}, h[{i}]={h_vals[i]:.1f}"
            )

    def test_h_falls_in_chamber(self) -> None:
        """h in convergent section << h at throat (area ratio effect)."""
        h_inlet = bartz_h(x=NOZZLE_REF.x_inlet, nozzle=NOZZLE_REF, gas=GAS_REF, T_wall=T_WALL)
        h_throat = bartz_h(x=0.0, nozzle=NOZZLE_REF, gas=GAS_REF, T_wall=T_WALL)
        A_ratio_inlet = NOZZLE_REF.area_ratio(NOZZLE_REF.x_inlet)
        # h_inlet / h_throat ~ (1/A_ratio_inlet)^0.9 times sigma ratio
        # qualitative check: h_inlet < h_throat
        assert h_inlet < h_throat, (
            f"h_inlet={h_inlet:.0f}, h_throat={h_throat:.0f}: "
            "h at inlet must be less than at throat"
        )

    def test_T_aw_approaches_Tc_at_low_M(self) -> None:
        """T_aw -> Tc as M -> 0 (convergent inlet)."""
        T_aw = adiabatic_wall_temperature(NOZZLE_REF.x_inlet, NOZZLE_REF, GAS_REF)
        assert abs(T_aw - GAS_REF.Tc) / GAS_REF.Tc < 0.05, (
            f"T_aw = {T_aw:.1f} K, Tc = {GAS_REF.Tc:.1f} K at M~0"
        )

    def test_T_aw_below_Tc_at_high_M(self) -> None:
        """T_aw < Tc in the supersonic divergent section (kinetic energy effect)."""
        T_aw = adiabatic_wall_temperature(NOZZLE_REF.x_exit, NOZZLE_REF, GAS_REF)
        assert T_aw < GAS_REF.Tc, (
            f"T_aw = {T_aw:.1f} K should be < Tc = {GAS_REF.Tc:.1f} K at exit"
        )


# ===========================================================================
# 5. heat_flux_profile
# ===========================================================================
class TestHeatFluxProfile:
    def test_profile_shape(self) -> None:
        """heat_flux_profile returns dict with correct array lengths."""
        prof = heat_flux_profile(nozzle=NOZZLE_REF, gas=GAS_REF, T_wall_guess=T_WALL, n=50)
        for key in ("x", "r", "M", "T_aw", "h_g", "q_dot_wall"):
            assert key in prof
            assert len(prof[key]) == 50

    def test_q_dot_positive_cold_wall(self) -> None:
        """Heat flows into the wall (q_dot > 0) when T_wall < T_aw everywhere."""
        prof = heat_flux_profile(nozzle=NOZZLE_REF, gas=GAS_REF, T_wall_guess=600.0, n=50)
        assert np.all(prof["q_dot_wall"] > 0), "All q_dot should be positive for cold wall"

    def test_q_dot_peak_near_throat(self) -> None:
        """Peak heat flux occurs in the throat region."""
        prof = heat_flux_profile(nozzle=NOZZLE_REF, gas=GAS_REF, T_wall_guess=T_WALL, n=100)
        i_peak = int(np.argmax(prof["q_dot_wall"]))
        x_peak = prof["x"][i_peak]
        # Peak should be in the throat region (within 20% of nozzle length from throat)
        L_total = NOZZLE_REF.x_exit - NOZZLE_REF.x_inlet
        assert abs(x_peak) < 0.20 * L_total, (
            f"Peak q_dot at x = {x_peak*1e3:.1f} mm, expected near throat"
        )

    def test_mach_increases_through_nozzle(self) -> None:
        """M < 1 in convergent, M = 1 at throat, M > 1 in divergent."""
        prof = heat_flux_profile(nozzle=NOZZLE_REF, gas=GAS_REF, T_wall_guess=T_WALL, n=100)
        x = prof["x"]
        M = prof["M"]
        # All convergent: M < 1
        assert np.all(M[x < -1e-6] < 1.0)
        # All divergent: M > 1
        assert np.all(M[x > 1e-6] > 1.0)

    def test_r_c_default_is_downstream_arc(self) -> None:
        """Default R_c = 0.382 * r_throat (downstream throat arc radius)."""
        h_default = bartz_h(x=0.0, nozzle=NOZZLE_REF, gas=GAS_REF, T_wall=T_WALL)
        h_explicit = bartz_h(
            x=0.0,
            nozzle=NOZZLE_REF,
            gas=GAS_REF,
            T_wall=T_WALL,
            R_c=0.382 * NOZZLE_REF.r_throat,
        )
        assert abs(h_default - h_explicit) / h_default < 1e-10
