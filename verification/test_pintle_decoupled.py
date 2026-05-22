"""
test_pintle_decoupled.py
========================

Tests for the decoupled dP_ox / dP_fuel path introduced after commit 04a0cb6.

Covers:
  1. Backward-compat via from_uniform_dP and legacy dP_inj keyword.
  2. TMR monotonicity when dP_fuel / dP_ox varies.
  3. Per-stream stiffness check (ox passes / fuel fails, and vice versa).
"""

from __future__ import annotations

import math

import pytest

from physics.injector.pintle import PintleInjector
from physics.injector.stability import check_stiffness
from noyron.kernel.util.units import PSI_TO_PA

# ---------------------------------------------------------------------------
# Reference engine constants
# ---------------------------------------------------------------------------
PC_REF = 1000.0 * PSI_TO_PA
DP_REF = 450.0 * PSI_TO_PA

M_DOT_FUEL = 0.3096
M_DOT_OX = 1.1144
RHO_FUEL = 386.0
RHO_OX = 1116.0
CD_FUEL = 0.6
CD_OX = 0.6
PINTLE_OD = 0.030
N_FUEL = 20

MR = M_DOT_OX / M_DOT_FUEL  # 3.600


def _uniform(dP: float = DP_REF) -> PintleInjector:
    """Convenience: equal dP on both streams via new keyword form."""
    return PintleInjector(
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


def _decoupled(dP_ox: float, dP_fuel: float) -> PintleInjector:
    return PintleInjector(
        m_dot_ox=M_DOT_OX,
        m_dot_fuel=M_DOT_FUEL,
        rho_ox=RHO_OX,
        rho_fuel=RHO_FUEL,
        dP_ox=dP_ox,
        dP_fuel=dP_fuel,
        Cd_ox=CD_OX,
        Cd_fuel=CD_FUEL,
        pintle_OD=PINTLE_OD,
        n_fuel_orifices=N_FUEL,
    )


# ===========================================================================
# 1. Backward compatibility
# ===========================================================================
class TestBackwardCompat:
    """from_uniform_dP and legacy dP_inj keyword must match the direct path."""

    def test_from_uniform_dP_matches_direct(self) -> None:
        """from_uniform_dP(dP_inj=X) reproduces every output of dP_ox=X,dP_fuel=X."""
        direct = _uniform(DP_REF)
        via_classmethod = PintleInjector.from_uniform_dP(
            dP_inj=DP_REF,
            m_dot_ox=M_DOT_OX,
            m_dot_fuel=M_DOT_FUEL,
            rho_ox=RHO_OX,
            rho_fuel=RHO_FUEL,
            Cd_ox=CD_OX,
            Cd_fuel=CD_FUEL,
            pintle_OD=PINTLE_OD,
            n_fuel_orifices=N_FUEL,
        )
        assert direct.fuel_orifice_diameter == pytest.approx(
            via_classmethod.fuel_orifice_diameter, rel=1e-12
        )
        assert direct.lox_annulus_OD == pytest.approx(
            via_classmethod.lox_annulus_OD, rel=1e-12
        )
        assert direct.V_ox == pytest.approx(via_classmethod.V_ox, rel=1e-12)
        assert direct.V_fuel == pytest.approx(via_classmethod.V_fuel, rel=1e-12)
        assert direct.TMR == pytest.approx(via_classmethod.TMR, rel=1e-12)

    def test_legacy_dP_inj_keyword_matches_direct(self) -> None:
        """PintleInjector(dP_inj=X) backward-compat path matches dP_ox=X,dP_fuel=X."""
        legacy = PintleInjector(
            m_dot_ox=M_DOT_OX,
            m_dot_fuel=M_DOT_FUEL,
            rho_ox=RHO_OX,
            rho_fuel=RHO_FUEL,
            dP_inj=DP_REF,
            Cd_ox=CD_OX,
            Cd_fuel=CD_FUEL,
            pintle_OD=PINTLE_OD,
            n_fuel_orifices=N_FUEL,
        )
        direct = _uniform(DP_REF)
        assert legacy.fuel_orifice_diameter == pytest.approx(
            direct.fuel_orifice_diameter, rel=1e-12
        )
        assert legacy.lox_annulus_OD == pytest.approx(
            direct.lox_annulus_OD, rel=1e-12
        )
        assert legacy.TMR == pytest.approx(direct.TMR, rel=1e-12)

    def test_missing_both_dP_raises(self) -> None:
        """Constructing without any dP specification raises ValueError."""
        with pytest.raises(ValueError, match="dP_ox"):
            PintleInjector(
                m_dot_ox=M_DOT_OX,
                m_dot_fuel=M_DOT_FUEL,
                rho_ox=RHO_OX,
                rho_fuel=RHO_FUEL,
            )


# ===========================================================================
# 2. TMR monotonicity in dP_fuel / dP_ox ratio
# ===========================================================================
class TestTMRDecoupled:
    """TMR = (1/MR)·(Cd_f/Cd_ox)·sqrt((rho_ox/rho_f)·(dP_f/dP_ox)).

    With fixed dP_ox and Cd_f = Cd_ox, TMR is monotonically increasing
    in dP_fuel / dP_ox.
    """

    def test_TMR_monotonic_in_dP_ratio(self) -> None:
        """TMR increases strictly as dP_fuel / dP_ox rises from 0.5 → 3.0."""
        dP_ox_fixed = DP_REF
        ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
        tmrs = [
            _decoupled(dP_ox_fixed, r * dP_ox_fixed).TMR for r in ratios
        ]
        for i in range(1, len(tmrs)):
            assert tmrs[i] > tmrs[i - 1], (
                f"TMR not increasing: TMR({ratios[i-1]:.2f}) = {tmrs[i-1]:.5f}, "
                f"TMR({ratios[i]:.2f}) = {tmrs[i]:.5f}"
            )

    def test_TMR_formula_analytic(self) -> None:
        """TMR matches analytic formula (1/MR)·(Cd_f/Cd_ox)·sqrt(rho_ox/rho_f·dPf/dPox)."""
        dP_ox = DP_REF
        dP_fuel = 1.5 * DP_REF
        p = _decoupled(dP_ox, dP_fuel)
        expected = (1.0 / MR) * (CD_FUEL / CD_OX) * math.sqrt(
            (RHO_OX / RHO_FUEL) * (dP_fuel / dP_ox)
        )
        assert abs(p.TMR - expected) / expected < 1e-9

    def test_TMR_at_unity_ratio(self) -> None:
        """With equal dP, TMR = (1/MR)·sqrt(rho_ox/rho_fuel) regardless of Pc."""
        expected = (1.0 / MR) * math.sqrt(RHO_OX / RHO_FUEL)
        for scale in (0.5, 1.0, 2.0):
            p = _decoupled(scale * DP_REF, scale * DP_REF)
            assert abs(p.TMR - expected) / expected < 1e-9

    def test_higher_fuel_dP_raises_TMR(self) -> None:
        """dP_fuel > dP_ox gives TMR > equal-dP baseline (all else fixed)."""
        baseline = _decoupled(DP_REF, DP_REF).TMR
        higher = _decoupled(DP_REF, 1.5 * DP_REF).TMR
        assert higher > baseline

    def test_lower_fuel_dP_lowers_TMR(self) -> None:
        """dP_fuel < dP_ox gives TMR < equal-dP baseline."""
        baseline = _decoupled(DP_REF, DP_REF).TMR
        lower = _decoupled(DP_REF, 0.7 * DP_REF).TMR
        assert lower < baseline


# ===========================================================================
# 3. Per-stream stiffness checks
# ===========================================================================
class TestStiffnessPerSide:
    """Ox can pass while fuel fails, and vice versa."""

    def test_both_pass_reference(self) -> None:
        """Reference engine: both streams pass at dP/Pc = 0.45."""
        result = check_stiffness(dP_ox=DP_REF, dP_fuel=DP_REF, Pc=PC_REF)
        assert result["ox"][0] is True
        assert result["fuel"][0] is True

    def test_ox_passes_fuel_fails(self) -> None:
        """High dP_ox (passes), very low dP_fuel (fails)."""
        dP_ox_high = 0.45 * PC_REF      # 45% — passes
        dP_fuel_low = 0.05 * PC_REF     # 5% — fails (< 20% threshold)
        result = check_stiffness(dP_ox=dP_ox_high, dP_fuel=dP_fuel_low, Pc=PC_REF)
        assert result["ox"][0] is True, (
            f"ox should PASS (ratio={result['ox'][1]:.3f})"
        )
        assert result["fuel"][0] is False, (
            f"fuel should FAIL (ratio={result['fuel'][1]:.3f})"
        )

    def test_fuel_passes_ox_fails(self) -> None:
        """High dP_fuel (passes), very low dP_ox (fails)."""
        dP_ox_low = 0.05 * PC_REF       # 5% — fails
        dP_fuel_high = 0.45 * PC_REF    # 45% — passes
        result = check_stiffness(dP_ox=dP_ox_low, dP_fuel=dP_fuel_high, Pc=PC_REF)
        assert result["ox"][0] is False, (
            f"ox should FAIL (ratio={result['ox'][1]:.3f})"
        )
        assert result["fuel"][0] is True, (
            f"fuel should PASS (ratio={result['fuel'][1]:.3f})"
        )

    def test_both_fail_low_dP(self) -> None:
        """Very low dP on both streams: both fail."""
        dP_low = 0.05 * PC_REF
        result = check_stiffness(dP_ox=dP_low, dP_fuel=dP_low, Pc=PC_REF)
        assert result["ox"][0] is False
        assert result["fuel"][0] is False

    def test_ratios_are_correct(self) -> None:
        """check_stiffness reports the correct numerical ratios."""
        dP_ox = 0.30 * PC_REF
        dP_fuel = 0.40 * PC_REF
        result = check_stiffness(dP_ox=dP_ox, dP_fuel=dP_fuel, Pc=PC_REF)
        assert abs(result["ox"][1] - 0.30) / 0.30 < 1e-9
        assert abs(result["fuel"][1] - 0.40) / 0.40 < 1e-9
