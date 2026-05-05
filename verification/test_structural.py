"""
test_structural.py
==================

Verification tests for physics/structural/ -- first-pass analytical liner
stress and LCF check for the regeneratively-cooled chamber.

Sources
-------
    Timoshenko & Goodier (1951), Theory of Elasticity -- hoop stress, thermal
    stress linear-gradient formulas.
    Ellis (2005) NASA/TM-2005-213566 -- GRCop-84 yield strength at temperature.
    Quentmeyer (1977) NASA TM-X-73665 -- LCF framework for copper liners.
    NASA SP-8087 (1977) Sec. 4 -- combined stress analysis for regen liners.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from physics.structural.liner_stress import (
    hoop_stress,
    liner_stress_profile,
    through_wall_thermal_stress,
    von_mises,
)
from physics.structural.materials import GRCOP84_PROXY
from physics.structural.fatigue import lcf_check, manson_coffin_life

# ---------------------------------------------------------------------------
# Reference engine fixtures (same as other test modules)
# ---------------------------------------------------------------------------
from physics.nozzle.rao import RaoNozzle
from physics.regen.bartz import CombustionGas
from physics.regen.coolant import GRCOP42, CoolingChannelGeometry, CoolantInlet
from physics.regen.solver import solve_regen

GAS = CombustionGas(
    Tc=3561.2,
    Pc=6.8948e6,
    gamma=1.15,
    MW=21.9,
    cp=2000.0,
    mu=8.0e-5,
    Pr=0.60,
    c_star=1893.3,
)
NOZZLE = RaoNozzle(r_throat=0.022313 / 2.0, epsilon=10.605)
COOLANT = CoolantInlet(T=150.0, P=12.0e6, m_dot=0.3096, fluid="Methane")

# Sized geometry (76 ch, w=0.6 mm, h_throat=0.3 mm, h_div=3.2 mm)
_X2 = 2.0 * NOZZLE.r_throat
SIZED_GEOM = CoolingChannelGeometry(
    n_channels=76,
    width=lambda x: 0.6e-3,
    height=lambda x: 0.3e-3 if x < _X2 else 3.2e-3,
    wall_thickness=lambda x: 0.5e-3,
    land_thickness=lambda x: 0.3e-3,
)

# Cache the regen solution so the suite only runs solve_regen once.
_CACHED_REGEN: dict | None = None


def _get_regen() -> dict:
    global _CACHED_REGEN
    if _CACHED_REGEN is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _CACHED_REGEN = solve_regen(
                nozzle=NOZZLE,
                gas=GAS,
                channel_geom=SIZED_GEOM,
                coolant_inlet=COOLANT,
                wall_material=GRCOP42,
                direction="counterflow",
                n_stations=100,
                tol=1.0,
            )
    return _CACHED_REGEN


# ---------------------------------------------------------------------------
# 1. Thin-shell hoop stress recovers P*r/t
# ---------------------------------------------------------------------------
class TestThinShell:
    def test_thin_shell_recovers_pr_over_t(self) -> None:
        """hoop_stress returns P*r/t exactly for any valid inputs.

        Source: Timoshenko & Goodier (1951) §131.
        """
        P = 6.0e6   # Pa  (chamber pressure)
        r = 0.015   # m   (15 mm radius)
        t = 0.5e-3  # m   (0.5 mm wall)
        expected = P * r / t
        result = hoop_stress(P_internal=P, r_inner=r, t_wall=t)
        assert abs(result - expected) < 1.0, (
            f"hoop_stress = {result:.0f} Pa, expected {expected:.0f} Pa"
        )

    def test_zero_pressure_gives_zero_stress(self) -> None:
        """Zero pressure -> zero hoop stress."""
        assert hoop_stress(P_internal=0.0, r_inner=0.01, t_wall=0.5e-3) == 0.0

    def test_invalid_t_wall_raises(self) -> None:
        with pytest.raises(ValueError, match="t_wall"):
            hoop_stress(P_internal=1e6, r_inner=0.01, t_wall=0.0)

    def test_invalid_r_inner_raises(self) -> None:
        with pytest.raises(ValueError, match="r_inner"):
            hoop_stress(P_internal=1e6, r_inner=0.0, t_wall=1e-3)


# ---------------------------------------------------------------------------
# 2. Thermal stress is zero when gradient is zero
# ---------------------------------------------------------------------------
class TestThermalStressZeroGradient:
    def test_zero_at_zero_gradient(self) -> None:
        """T_gas = T_cool → both thermal stresses are zero.

        Source: Timoshenko & Goodier (1951) §143 -- sigma = 0 when T(z) = const.
        """
        T = 500.0
        sig_g, sig_c = through_wall_thermal_stress(
            T_gas_side=T, T_cool_side=T, E=100e9, alpha=17e-6, nu=0.34
        )
        assert abs(sig_g) < 1.0, f"sigma_th_gas = {sig_g:.1f} Pa (expected 0)"
        assert abs(sig_c) < 1.0, f"sigma_th_cool = {sig_c:.1f} Pa (expected 0)"


# ---------------------------------------------------------------------------
# 3. Thermal stress sign convention
# ---------------------------------------------------------------------------
class TestThermalStressSign:
    def test_gas_side_compressive_hot_wall(self) -> None:
        """Hot gas face is in compression when T_gas > T_cool.

        Physical justification: the hot face wants to expand relative to the
        constrained bulk -> compressive (negative) stress.
        Source: Timoshenko & Goodier (1951) §143; NASA SP-8087 Sec. 4.3.
        """
        sig_g, sig_c = through_wall_thermal_stress(
            T_gas_side=900.0, T_cool_side=300.0,
            E=100e9, alpha=17e-6, nu=0.34,
        )
        assert sig_g < 0.0, (
            f"Gas-side thermal stress should be compressive (< 0), got {sig_g/1e6:.1f} MPa"
        )

    def test_cool_side_tensile_hot_wall(self) -> None:
        """Cool face is in tension when T_gas > T_cool.

        Source: Timoshenko & Goodier (1951) §143.
        """
        sig_g, sig_c = through_wall_thermal_stress(
            T_gas_side=900.0, T_cool_side=300.0,
            E=100e9, alpha=17e-6, nu=0.34,
        )
        assert sig_c > 0.0, (
            f"Cool-side thermal stress should be tensile (> 0), got {sig_c/1e6:.1f} MPa"
        )

    def test_gas_cool_are_equal_and_opposite(self) -> None:
        """sigma_th_gas = -sigma_th_cool (symmetric about the midplane).

        Follows directly from the linear gradient formula.
        """
        sig_g, sig_c = through_wall_thermal_stress(
            T_gas_side=800.0, T_cool_side=200.0,
            E=110e9, alpha=17e-6, nu=0.34,
        )
        assert abs(sig_g + sig_c) < 1.0, (
            f"|sigma_gas + sigma_cool| = {abs(sig_g+sig_c):.1f} Pa (expected 0)"
        )


# ---------------------------------------------------------------------------
# 4. Von Mises stress -- textbook checks
# ---------------------------------------------------------------------------
class TestVonMises:
    def test_uniaxial_recovers_sigma(self) -> None:
        """For pure uniaxial stress sigma_1 (sigma_2 = sigma_3 = 0):
        sigma_vm = |sigma_1|.

        Source: Timoshenko & Goodier (1951) §151.
        """
        sigma = 200.0e6
        result = von_mises(sigma, 0.0, 0.0)
        assert abs(result - sigma) < 1.0, (
            f"von_mises({sigma/1e6} MPa, 0, 0) = {result/1e6:.2f} MPa, "
            f"expected {sigma/1e6:.2f} MPa"
        )

    def test_equal_biaxial_recovers_sigma(self) -> None:
        """For equal biaxial tension sigma_1 = sigma_2, sigma_3 = 0:
        sigma_vm = |sigma_1|.  (The shear terms cancel.)
        """
        sigma = 150.0e6
        result = von_mises(sigma, sigma, 0.0)
        assert abs(result - sigma) < 1.0, (
            f"von_mises({sigma/1e6}, {sigma/1e6}, 0) = {result/1e6:.2f} MPa"
        )

    def test_pure_shear(self) -> None:
        """For pure shear (sigma_1 = tau, sigma_2 = -tau, sigma_3 = 0):
        sigma_vm = sqrt(3) * tau.

        Source: von Mises criterion, standard result.
        """
        tau = 100.0e6
        result = von_mises(tau, -tau, 0.0)
        expected = math.sqrt(3.0) * tau
        assert abs(result - expected) / expected < 1e-9, (
            f"pure-shear: sigma_vm = {result/1e6:.3f}, expected {expected/1e6:.3f} MPa"
        )

    def test_always_nonnegative(self) -> None:
        """Von Mises stress is always non-negative."""
        for s1, s2, s3 in [(-500e6, 200e6, 0), (0, 0, 0), (-100e6, -200e6, -300e6)]:
            assert von_mises(s1, s2, s3) >= 0.0


# ---------------------------------------------------------------------------
# 5. GRCop-84 yield at temperature matches Ellis (2005) data
# ---------------------------------------------------------------------------
class TestGrcopYieldAtTemperature:
    def test_yield_at_295K(self) -> None:
        """sigma_y(295 K) ≈ 266 MPa per Ellis (2005) NASA/TM-2005-213566, Table 2."""
        result = GRCOP84_PROXY.sigma_y(295.0)
        expected = 266.0e6
        assert abs(result - expected) / expected < 0.05, (
            f"sigma_y(295 K) = {result/1e6:.1f} MPa, expected {expected/1e6:.0f} MPa"
        )

    def test_yield_at_811K(self) -> None:
        """sigma_y(811 K) ≈ 168 MPa per Ellis (2005) Table 2."""
        result = GRCOP84_PROXY.sigma_y(811.0)
        expected = 168.0e6
        assert abs(result - expected) / expected < 0.05, (
            f"sigma_y(811 K) = {result/1e6:.1f} MPa, expected {expected/1e6:.0f} MPa"
        )

    def test_yield_decreases_with_temperature(self) -> None:
        """Yield strength decreases monotonically with temperature."""
        temps = [300.0, 500.0, 700.0, 900.0]
        sy = [GRCOP84_PROXY.sigma_y(T) for T in temps]
        for i in range(len(sy) - 1):
            assert sy[i] > sy[i + 1], (
                f"sigma_y not monotone: sy({temps[i]}) = {sy[i]/1e6:.1f}, "
                f"sy({temps[i+1]}) = {sy[i+1]/1e6:.1f} MPa"
            )

    def test_E_at_295K(self) -> None:
        """E(295 K) ≈ 117 GPa per Ellis (2005) Table 2."""
        result = GRCOP84_PROXY.E(295.0)
        expected = 117.0e9
        assert abs(result - expected) / expected < 0.05, (
            f"E(295 K) = {result/1e9:.1f} GPa, expected {expected/1e9:.0f} GPa"
        )


# ---------------------------------------------------------------------------
# 6. Manson-Coffin: unit checks
# ---------------------------------------------------------------------------
class TestMansonCoffin:
    def test_zero_strain_returns_inf(self) -> None:
        """Zero strain amplitude -> infinite life."""
        Nf = manson_coffin_life(
            strain_range_plastic=0.0,
            strain_range_elastic=0.0,
            material=GRCOP84_PROXY,
        )
        assert Nf == math.inf

    def test_high_strain_short_life(self) -> None:
        """5 % total plastic strain -> life well under 1000 cycles."""
        Nf = manson_coffin_life(
            strain_range_plastic=0.05,
            strain_range_elastic=1e-4,
            material=GRCOP84_PROXY,
        )
        assert Nf < 1000, f"Expected N_f < 1000 for 5 % plastic strain, got {Nf:.0f}"

    def test_low_strain_long_life(self) -> None:
        """0.01 % plastic strain -> life > 10,000 cycles."""
        Nf = manson_coffin_life(
            strain_range_plastic=1e-4,
            strain_range_elastic=1e-4,
            material=GRCOP84_PROXY,
        )
        assert Nf > 10_000, f"Expected N_f > 10,000 for tiny strain, got {Nf:.0f}"

    def test_life_decreases_with_strain(self) -> None:
        """Higher strain range -> shorter life."""
        Nf_low = manson_coffin_life(
            strain_range_plastic=0.002, strain_range_elastic=1e-3, material=GRCOP84_PROXY
        )
        Nf_high = manson_coffin_life(
            strain_range_plastic=0.020, strain_range_elastic=1e-3, material=GRCOP84_PROXY
        )
        assert Nf_low > Nf_high, (
            f"N_f(low strain) = {Nf_low:.0f} should be > N_f(high strain) = {Nf_high:.0f}"
        )


# ---------------------------------------------------------------------------
# 7. Full chamber stress profile runs without error
# ---------------------------------------------------------------------------
class TestFullChamberCheck:
    def test_profile_runs_and_returns_sane_values(self) -> None:
        """liner_stress_profile completes on the sized regen design and
        returns physically plausible values.

        Checks:
          - All required keys present
          - Arrays have correct length
          - sigma_hoop > 0 everywhere (tensile from coolant pressure)
          - sigma_thermal_gas < 0 everywhere (compressive, hot side)
          - sigma_thermal_cool > 0 everywhere (tensile, cool side)
          - sigma_vm_gas > 0 everywhere
          - Peak VM in a physically plausible range [10 MPa, 2000 MPa]
        """
        regen = _get_regen()
        sol = liner_stress_profile(
            regen_solution=regen,
            geometry=SIZED_GEOM,
            material=GRCOP84_PROXY,
        )

        required_keys = [
            "x", "T_gas", "T_cool", "t_wall",
            "sigma_hoop", "sigma_thermal_gas", "sigma_thermal_cool",
            "sigma_vm_gas", "sigma_vm_cool", "margin_yield", "margin_ultimate",
        ]
        for k in required_keys:
            assert k in sol, f"Missing key: {k}"

        n = len(regen["x"])
        for k in required_keys:
            assert len(sol[k]) == n, f"{k}: expected length {n}, got {len(sol[k])}"

        assert np.all(sol["sigma_hoop"] > 0), "sigma_hoop should be positive everywhere"
        assert np.all(sol["sigma_thermal_gas"] < 0), (
            "sigma_thermal_gas should be compressive (< 0) everywhere"
        )
        assert np.all(sol["sigma_thermal_cool"] > 0), (
            "sigma_thermal_cool should be tensile (> 0) everywhere"
        )
        assert np.all(sol["sigma_vm_gas"] > 0), "sigma_vm_gas must be positive"

        vm_peak = float(sol["sigma_vm_gas"].max())
        assert 10e6 < vm_peak < 2000e6, (
            f"Peak VM = {vm_peak/1e6:.1f} MPa outside plausible range [10, 2000] MPa"
        )

    def test_lcf_check_runs(self) -> None:
        """lcf_check completes and returns a positive predicted life."""
        regen = _get_regen()
        sol = liner_stress_profile(
            regen_solution=regen,
            geometry=SIZED_GEOM,
            material=GRCOP84_PROXY,
        )
        result = lcf_check(
            regen_solution=regen,
            structural_solution=sol,
            n_cycles_required=100,
            material=GRCOP84_PROXY,
        )
        assert result["N_f"] > 0, "Predicted life must be positive"
        assert result["N_f"] < math.inf, "Predicted life must be finite"
        assert "margin_lcf" in result
        assert "passes" in result
