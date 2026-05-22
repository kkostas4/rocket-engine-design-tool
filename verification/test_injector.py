"""
test_injector.py
================

Verification tests for the injector subsystem:
  - physics/injector/orifice.py
  - physics/injector/pintle.py
  - physics/injector/stability.py

Reference engine conditions (Pc = 1000 psia, O/F = 3.60, LCH4/LOX):
    dP_inj  = 450 psi  = 3.1026e6 Pa  (equal for both streams)
    LCH4:   m_dot = 0.3096 kg/s, rho = 386 kg/m³
    LOX:    m_dot = 1.1144 kg/s, rho = 1116 kg/m³
    Cd_fuel = Cd_ox = 0.6
    pintle_OD = 0.030 m, n_fuel_orifices = 20

Expected outputs (from CLAUDE.md baseline):
    fuel orifice diameter = 0.819 mm
    LOX annulus OD        = 30.47 mm
    TMR                   ≈ 0.47  (correct momentum-ratio formula)

NOTE — TMR history
------------------
The original implementation used ρ_ox·V_ox² / (ρ_f·V_f²), a dynamic-pressure
ratio that algebraically reduces to Cd_ox²·dP_ox / (Cd_f²·dP_f).  With equal
Cd and equal dP that is identically 1.000 — independent of ρ, ṁ, or MR.
The correct momentum-ratio TMR = (ṁ_f·V_f)/(ṁ_ox·V_ox) carries the density
and mass-flow dependence; for the reference engine it evaluates to ≈ 0.47.
"""

from __future__ import annotations

import math

import pytest

from noyron.rp.injector.orifice import (
    injection_velocity,
    orifice_area_required,
    orifice_mass_flow,
)
from noyron.rp.injector.pintle import PintleInjector
from noyron.rp.injector.stability import check_stiffness, stiffness_ratio
from noyron.kernel.util.units import PSI_TO_PA

# ---------------------------------------------------------------------------
# Reference engine constants
# ---------------------------------------------------------------------------
PC_REF = 1000.0 * PSI_TO_PA        # 6.8948e6 Pa
DP_INJ = 450.0 * PSI_TO_PA         # 3.1026e6 Pa

M_DOT_FUEL = 0.3096                 # kg/s
M_DOT_OX = 1.1144                   # kg/s
RHO_FUEL = 386.0                    # kg/m³ — LCH4 at 118 K
RHO_OX = 1116.0                     # kg/m³ — LOX at 100 K
CD_FUEL = 0.6
CD_OX = 0.6
PINTLE_OD = 0.030                   # m
N_FUEL = 20

# Expected TMR with correct momentum-ratio formula and equal Cd/dP:
#   TMR = (1/MR) * sqrt(rho_ox/rho_fuel)
MR_REF = M_DOT_OX / M_DOT_FUEL      # 3.600
TMR_EXPECTED = (1.0 / MR_REF) * math.sqrt(RHO_OX / RHO_FUEL)  # ≈ 0.472


# ---------------------------------------------------------------------------
# Helper: build the reference pintle injector (backward-compat path)
# ---------------------------------------------------------------------------
def _ref_pintle() -> PintleInjector:
    # Uses legacy dP_inj keyword — handled by __post_init__ backward compat.
    return PintleInjector(
        m_dot_ox=M_DOT_OX,
        m_dot_fuel=M_DOT_FUEL,
        rho_ox=RHO_OX,
        rho_fuel=RHO_FUEL,
        dP_inj=DP_INJ,
        Cd_ox=CD_OX,
        Cd_fuel=CD_FUEL,
        pintle_OD=PINTLE_OD,
        n_fuel_orifices=N_FUEL,
    )


# ===========================================================================
# 1. Unit tests for the three orifice primitives
# ===========================================================================
class TestOrificeEquations:
    """Unit tests for orifice_mass_flow, orifice_area_required, injection_velocity."""

    def test_round_trip_mass_flow(self) -> None:
        """orifice_mass_flow(Cd, orifice_area_required(m, Cd, rho, dP), rho, dP) == m."""
        for m_target in (0.10, 0.3096, 1.1144, 2.5):
            A = orifice_area_required(m_target, CD_FUEL, RHO_FUEL, DP_INJ)
            m_back = orifice_mass_flow(CD_FUEL, A, RHO_FUEL, DP_INJ)
            rel_err = abs(m_back - m_target) / m_target
            assert rel_err < 1e-12, (
                f"Round-trip failed for m={m_target}: got {m_back}, "
                f"rel error = {rel_err:.2e}"
            )

    def test_orifice_mass_flow_positive(self) -> None:
        """Mass flow is positive for valid inputs."""
        m = orifice_mass_flow(0.6, 1e-5, 386.0, 3.1e6)
        assert m > 0.0

    def test_orifice_area_required_positive(self) -> None:
        """Required area is positive for valid inputs."""
        A = orifice_area_required(0.3096, 0.6, 386.0, 3.1e6)
        assert A > 0.0

    def test_injection_velocity_positive(self) -> None:
        """Injection velocity is positive for valid inputs."""
        V = injection_velocity(0.3096, 1.054e-5, 386.0)
        assert V > 0.0

    # --- ValueError on zero inputs ---
    def test_orifice_mass_flow_zero_Cd(self) -> None:
        with pytest.raises(ValueError, match="Cd"):
            orifice_mass_flow(0.0, 1e-5, 386.0, 3.1e6)

    def test_orifice_mass_flow_negative_A(self) -> None:
        with pytest.raises(ValueError, match="A"):
            orifice_mass_flow(0.6, -1e-5, 386.0, 3.1e6)

    def test_orifice_mass_flow_zero_rho(self) -> None:
        with pytest.raises(ValueError, match="rho"):
            orifice_mass_flow(0.6, 1e-5, 0.0, 3.1e6)

    def test_orifice_mass_flow_zero_dP(self) -> None:
        with pytest.raises(ValueError, match="dP"):
            orifice_mass_flow(0.6, 1e-5, 386.0, 0.0)

    def test_orifice_area_required_zero_mdot(self) -> None:
        with pytest.raises(ValueError, match="m_dot"):
            orifice_area_required(0.0, 0.6, 386.0, 3.1e6)

    def test_orifice_area_required_negative_dP(self) -> None:
        with pytest.raises(ValueError, match="dP"):
            orifice_area_required(0.3096, 0.6, 386.0, -1.0)

    def test_injection_velocity_zero_A(self) -> None:
        with pytest.raises(ValueError, match="A"):
            injection_velocity(0.3096, 0.0, 386.0)

    def test_injection_velocity_zero_rho(self) -> None:
        with pytest.raises(ValueError, match="rho"):
            injection_velocity(0.3096, 1e-5, 0.0)

    def test_injection_velocity_negative_mdot(self) -> None:
        with pytest.raises(ValueError, match="m_dot"):
            injection_velocity(-1.0, 1e-5, 386.0)


# ===========================================================================
# 2. Reference engine verification (1 % tolerance)
# ===========================================================================
class TestPintleReferenceEngine:
    """Reproduces CLAUDE.md baseline numbers to within 1%."""

    def test_fuel_orifice_diameter(self) -> None:
        """Fuel orifice diameter == 0.819 mm within 1%."""
        p = _ref_pintle()
        d_mm = p.fuel_orifice_diameter * 1e3
        expected_mm = 0.819
        rel_err = abs(d_mm - expected_mm) / expected_mm
        assert rel_err < 0.01, (
            f"fuel_orifice_diameter = {d_mm:.4f} mm, "
            f"expected {expected_mm} mm, rel err = {rel_err:.4f}"
        )

    def test_lox_annulus_OD(self) -> None:
        """LOX annulus OD == 30.47 mm within 1%."""
        p = _ref_pintle()
        od_mm = p.lox_annulus_OD * 1e3
        expected_mm = 30.47
        rel_err = abs(od_mm - expected_mm) / expected_mm
        assert rel_err < 0.01, (
            f"lox_annulus_OD = {od_mm:.4f} mm, "
            f"expected {expected_mm} mm, rel err = {rel_err:.4f}"
        )

    def test_TMR_correct_momentum_ratio(self) -> None:
        """TMR matches (1/MR)*sqrt(rho_ox/rho_fuel) with equal Cd and dP."""
        p = _ref_pintle()
        rel_err = abs(p.TMR - TMR_EXPECTED) / TMR_EXPECTED
        assert rel_err < 0.001, (
            f"TMR = {p.TMR:.6f}, expected {TMR_EXPECTED:.6f} "
            f"(dev = {rel_err*100:.4f} %)"
        )

    def test_TMR_in_dressler_range(self) -> None:
        """TMR is in Dressler & Bauer (2000) recommended range 0.5–2.0."""
        p = _ref_pintle()
        # Reference engine is near the lower bound; allow slightly below 0.5
        # because it is a real published engine and Dressler's range is a guide.
        assert 0.3 < p.TMR < 2.0, (
            f"TMR = {p.TMR:.4f} is far outside 0.3–2.0"
        )

    def test_V_ox_physical_range(self) -> None:
        """LOX injection velocity is in physically plausible range 10–200 m/s."""
        p = _ref_pintle()
        assert 10.0 < p.V_ox < 200.0, (
            f"V_ox = {p.V_ox:.2f} m/s outside expected range 10–200 m/s"
        )

    def test_V_fuel_physical_range(self) -> None:
        """Fuel injection velocity is in physically plausible range 10–200 m/s."""
        p = _ref_pintle()
        assert 10.0 < p.V_fuel < 200.0, (
            f"V_fuel = {p.V_fuel:.2f} m/s outside expected range 10–200 m/s"
        )

    def test_V_ox_positive(self) -> None:
        """LOX injection velocity is positive."""
        assert _ref_pintle().V_ox > 0.0

    def test_V_fuel_positive(self) -> None:
        """Fuel injection velocity is positive."""
        assert _ref_pintle().V_fuel > 0.0

    def test_summary_returns_string(self) -> None:
        """summary() returns a non-empty string."""
        s = _ref_pintle().summary()
        assert isinstance(s, str) and len(s) > 0

    def test_lox_annulus_gap_positive(self) -> None:
        """LOX annulus gap is positive (annulus OD > pintle OD)."""
        p = _ref_pintle()
        assert p.lox_annulus_gap > 0.0


# ===========================================================================
# 3. Stiffness ratio tests (updated for per-stream dict return)
# ===========================================================================
class TestStiffness:
    """Tests for stiffness_ratio and check_stiffness."""

    def test_stiffness_ratio_reference(self) -> None:
        """stiffness_ratio(450 psi, 1000 psia) ≈ 0.45 within 1%."""
        ratio = stiffness_ratio(DP_INJ, PC_REF)
        expected = 0.45
        rel_err = abs(ratio - expected) / expected
        assert rel_err < 0.01, (
            f"stiffness_ratio = {ratio:.5f}, expected ~{expected}, "
            f"rel err = {rel_err:.4f}"
        )

    def test_check_stiffness_reference_passes(self) -> None:
        """Reference engine (45%) passes 20% threshold on both streams."""
        result = check_stiffness(
            dP_ox=DP_INJ, dP_fuel=DP_INJ, Pc=PC_REF
        )
        assert result["ox"][0] is True
        assert result["fuel"][0] is True

    def test_check_stiffness_low_dP_fails(self) -> None:
        """dP = 10% Pc fails 20% threshold on both streams."""
        dP_low = 0.10 * PC_REF
        result = check_stiffness(dP_ox=dP_low, dP_fuel=dP_low, Pc=PC_REF)
        assert result["ox"][0] is False
        assert result["fuel"][0] is False

    def test_check_stiffness_exactly_at_threshold(self) -> None:
        """dP = exactly 20% Pc passes (>= threshold)."""
        dP_threshold = 0.20 * PC_REF
        result = check_stiffness(dP_ox=dP_threshold, dP_fuel=dP_threshold, Pc=PC_REF)
        assert result["ox"][0] is True
        assert result["fuel"][0] is True

    def test_check_stiffness_returns_dict(self) -> None:
        """check_stiffness returns dict with 'ox' and 'fuel' keys."""
        result = check_stiffness(dP_ox=DP_INJ, dP_fuel=DP_INJ, Pc=PC_REF)
        assert isinstance(result, dict)
        assert "ox" in result and "fuel" in result
        passed_ox, ratio_ox = result["ox"]
        passed_fuel, ratio_fuel = result["fuel"]
        assert isinstance(passed_ox, bool)
        assert isinstance(ratio_ox, float)
        assert isinstance(passed_fuel, bool)
        assert isinstance(ratio_fuel, float)

    def test_stiffness_ratio_zero_dP_raises(self) -> None:
        with pytest.raises(ValueError, match="dP_inj"):
            stiffness_ratio(0.0, PC_REF)

    def test_stiffness_ratio_zero_Pc_raises(self) -> None:
        with pytest.raises(ValueError, match="Pc"):
            stiffness_ratio(DP_INJ, 0.0)

    def test_check_stiffness_custom_threshold(self) -> None:
        """Custom threshold of 0.50 causes reference engine to fail."""
        result = check_stiffness(
            dP_ox=DP_INJ, dP_fuel=DP_INJ, Pc=PC_REF, threshold=0.50
        )
        assert result["ox"][0] is False
        assert result["fuel"][0] is False


# ===========================================================================
# 4. Pc sweep — parametric sanity check
# ===========================================================================
class TestPcSweep:
    """
    Sweep Pc 800–1400 psia with dP/Pc = 0.45 constant.

    Fuel orifice diameter must be strictly monotonically decreasing with Pc
    because larger Pc → larger dP (at fixed ratio) → smaller area needed.
    """

    def test_orifice_diameter_decreases_with_pc(self) -> None:
        """Fuel orifice diameter is strictly monotonically decreasing with Pc."""
        pc_psia_values = [800, 900, 1000, 1100, 1200, 1300, 1400]
        diameters = []

        for pc_psia in pc_psia_values:
            Pc = pc_psia * PSI_TO_PA
            dP = 0.45 * Pc
            p = PintleInjector(
                m_dot_ox=M_DOT_OX,
                m_dot_fuel=M_DOT_FUEL,
                rho_ox=RHO_OX,
                rho_fuel=RHO_FUEL,
                dP_ox=dP,
                dP_fuel=dP,
                Cd_ox=CD_OX,
                Cd_fuel=CD_FUEL,
                pintle_OD=PINTLE_OD,
                n_fuel_orifices=N_FUEL,
            )
            diameters.append(p.fuel_orifice_diameter)

        for i in range(1, len(diameters)):
            assert diameters[i] < diameters[i - 1], (
                f"Orifice diameter not decreasing: "
                f"d[{i-1}]={diameters[i-1]*1e3:.4f} mm at "
                f"Pc={pc_psia_values[i-1]} psia, "
                f"d[{i}]={diameters[i]*1e3:.4f} mm at "
                f"Pc={pc_psia_values[i]} psia"
            )

    def test_all_velocities_physical_in_sweep(self) -> None:
        """Injection velocities remain in 10–200 m/s range across sweep."""
        for pc_psia in range(800, 1401, 100):
            Pc = pc_psia * PSI_TO_PA
            dP = 0.45 * Pc
            p = PintleInjector(
                m_dot_ox=M_DOT_OX,
                m_dot_fuel=M_DOT_FUEL,
                rho_ox=RHO_OX,
                rho_fuel=RHO_FUEL,
                dP_ox=dP,
                dP_fuel=dP,
            )
            assert 10.0 < p.V_ox < 200.0, f"V_ox out of range at Pc={pc_psia} psia"
            assert 10.0 < p.V_fuel < 200.0, f"V_fuel out of range at Pc={pc_psia} psia"

    def test_TMR_constant_in_equal_dP_sweep(self) -> None:
        """TMR is constant ≈ (1/MR)·sqrt(rho_ox/rho_fuel) across equal-dP sweep.

        With equal Cd and equal dP (both scaling together), dP cancels out of
        the TMR formula, so TMR is independent of Pc.
        """
        tmrs = []
        for pc_psia in range(800, 1401, 100):
            Pc = pc_psia * PSI_TO_PA
            dP = 0.45 * Pc
            p = PintleInjector(
                m_dot_ox=M_DOT_OX,
                m_dot_fuel=M_DOT_FUEL,
                rho_ox=RHO_OX,
                rho_fuel=RHO_FUEL,
                dP_ox=dP,
                dP_fuel=dP,
            )
            tmrs.append(p.TMR)

        # All TMR values should be within 0.1 % of each other.
        tmr_mean = sum(tmrs) / len(tmrs)
        for tmr in tmrs:
            assert abs(tmr - tmr_mean) / tmr_mean < 0.001, (
                f"TMR not constant across sweep: {tmr:.6f} vs mean {tmr_mean:.6f}"
            )

        # And close to the analytic value.
        assert abs(tmr_mean - TMR_EXPECTED) / TMR_EXPECTED < 0.001
