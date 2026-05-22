"""
test_discharge.py
=================

Verification tests for physics/injector/discharge.py.

Tests the Lichtarowicz–Duggins–Markland (1965) discharge coefficient
correlation and its self-consistent iterative solver.

Reference
---------
    Lichtarowicz, A., Duggins, R.K. and Markland, E. (1965).
    "Discharge coefficients for incompressible non-cavitating flow through
    long orifices."
    Proc. IMechE, J. Mech. Eng. Sci., Vol. 7, pp. 210–219.
"""

from __future__ import annotations

import math
import warnings

import pytest

from noyron.rp.injector.discharge import cd_iterative, cd_lichtarowicz
from noyron.rp.injector.pintle import PintleInjector
from noyron.kernel.util.units import PSI_TO_PA

# ---------------------------------------------------------------------------
# Reference engine fuel conditions (used for iteration tests)
# ---------------------------------------------------------------------------
M_DOT_FUEL_PER = 0.3096 / 20        # kg/s per orifice
RHO_FUEL = 386.0                    # kg/m³ — LCH4 at 118 K
MU_FUEL = 1.4e-4                    # Pa·s — LCH4 at ~118 K (approx.)
DP_FUEL = 450.0 * PSI_TO_PA         # 3.1026e6 Pa


# ===========================================================================
# 1. Asymptotic behaviour
# ===========================================================================
class TestCdLichtarowicz:
    """Direct tests of the cd_lichtarowicz function."""

    def test_cd_asymptote_LD4(self) -> None:
        """At Re=1e6, L/D=4: Cd → Cd_inf = 0.827 - 0.0085*4 = 0.793 within 1%."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # Re > 2e5
            cd = cd_lichtarowicz(Re=1e6, LD=4.0)
        cd_inf = 0.827 - 0.0085 * 4.0  # 0.793
        assert abs(cd - cd_inf) / cd_inf < 0.01, (
            f"Cd = {cd:.5f}, Cd_inf = {cd_inf:.5f} (dev = {abs(cd-cd_inf)/cd_inf*100:.3f} %)"
        )

    def test_cd_asymptote_LD6(self) -> None:
        """At Re=1e6, L/D=6: Cd → 0.827 - 0.0085*6 = 0.776 within 1%."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cd = cd_lichtarowicz(Re=1e6, LD=6.0)
        cd_inf = 0.827 - 0.0085 * 6.0  # 0.776
        assert abs(cd - cd_inf) / cd_inf < 0.01

    def test_cd_low_re_below_asymptote(self) -> None:
        """At Re=100, Cd is materially below the high-Re asymptote."""
        cd_low = cd_lichtarowicz(Re=100.0, LD=4.0)
        cd_inf = 0.827 - 0.0085 * 4.0
        assert cd_low < cd_inf * 0.90, (
            f"Expected Cd << Cd_inf at Re=100; got Cd={cd_low:.4f}, Cd_inf={cd_inf:.4f}"
        )

    def test_cd_increases_with_re(self) -> None:
        """Cd is monotonically increasing with Re (viscous correction decreases)."""
        re_values = [50, 200, 1000, 5000, 2e4, 1e5]
        cds = [cd_lichtarowicz(Re=re, LD=4.0) for re in re_values]
        for i in range(1, len(cds)):
            assert cds[i] > cds[i - 1], (
                f"Cd not increasing: Cd(Re={re_values[i-1]:.0f})={cds[i-1]:.5f}, "
                f"Cd(Re={re_values[i]:.0f})={cds[i]:.5f}"
            )

    def test_cd_decreases_with_LD(self) -> None:
        """Cd_inf (and therefore Cd at fixed high Re) decreases as L/D increases."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cds = [cd_lichtarowicz(Re=1e6, LD=ld) for ld in [2, 4, 6, 8, 10]]
        for i in range(1, len(cds)):
            assert cds[i] < cds[i - 1], "Cd_inf should decrease with L/D"

    def test_validity_warning_high_re(self) -> None:
        """Re > 2e5 triggers a RuntimeWarning."""
        with pytest.warns(RuntimeWarning, match="validity range"):
            cd_lichtarowicz(Re=5e5, LD=4.0)

    def test_validity_warning_low_ld(self) -> None:
        """L/D < 2 triggers a RuntimeWarning."""
        with pytest.warns(RuntimeWarning, match="validity range"):
            cd_lichtarowicz(Re=1000.0, LD=1.0)

    def test_zero_re_raises(self) -> None:
        with pytest.raises(ValueError, match="Re"):
            cd_lichtarowicz(0.0, 4.0)

    def test_zero_ld_raises(self) -> None:
        with pytest.raises(ValueError, match="LD"):
            cd_lichtarowicz(1000.0, 0.0)


# ===========================================================================
# 2. Iterative solver
# ===========================================================================
class TestCdIterative:
    """Tests for cd_iterative fixed-point solver."""

    def test_converges_reference_conditions(self) -> None:
        """Solver converges for the reference engine per-orifice flow."""
        cd, re = cd_iterative(
            m_dot=M_DOT_FUEL_PER,
            rho=RHO_FUEL,
            mu=MU_FUEL,
            dP=DP_FUEL,
            LD=4.0,
        )
        # Cd should be in a physically plausible range.
        assert 0.5 < cd < 0.95, f"Unexpected Cd = {cd:.4f}"
        assert re > 0.0

    def test_self_consistency(self) -> None:
        """Returned (Cd, Re) satisfy the orifice and Re definitions."""
        cd, re = cd_iterative(
            m_dot=M_DOT_FUEL_PER,
            rho=RHO_FUEL,
            mu=MU_FUEL,
            dP=DP_FUEL,
            LD=4.0,
        )
        # Reconstruct diameter and check Re consistency.
        sqrt_2rho_dP = math.sqrt(2.0 * RHO_FUEL * DP_FUEL)
        A = M_DOT_FUEL_PER / (cd * sqrt_2rho_dP)
        D = math.sqrt(4.0 * A / math.pi)
        V = cd * math.sqrt(2.0 * DP_FUEL / RHO_FUEL)
        re_check = RHO_FUEL * V * D / MU_FUEL
        assert abs(re_check - re) / re < 1e-4, (
            f"Re consistency failed: solver Re={re:.1f}, recomputed={re_check:.1f}"
        )

    def test_cd_lichtarowicz_consistent_at_solution(self) -> None:
        """At the fixed point, cd_lichtarowicz(Re, LD) reproduces Cd."""
        cd, re = cd_iterative(
            m_dot=M_DOT_FUEL_PER,
            rho=RHO_FUEL,
            mu=MU_FUEL,
            dP=DP_FUEL,
            LD=4.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cd_check = cd_lichtarowicz(re, 4.0)
        assert abs(cd_check - cd) / cd < 1e-5, (
            f"Fixed-point not satisfied: Cd={cd:.6f}, Cd_licht(Re)={cd_check:.6f}"
        )

    def test_explicit_Cd_skips_iteration(self) -> None:
        """Passing Cd_fuel=0.75 in PintleInjector returns exactly 0.75 (no iteration)."""
        p = PintleInjector(
            m_dot_ox=1.1144,
            m_dot_fuel=0.3096,
            rho_ox=1116.0,
            rho_fuel=RHO_FUEL,
            dP_ox=DP_FUEL,
            dP_fuel=DP_FUEL,
            Cd_ox=0.6,
            Cd_fuel=0.75,   # explicit
        )
        assert p.cd_fuel_effective == pytest.approx(0.75, rel=1e-12)

    def test_different_LD_changes_result(self) -> None:
        """Different L/D values give different converged Cd."""
        cd_ld4, _ = cd_iterative(
            m_dot=M_DOT_FUEL_PER, rho=RHO_FUEL,
            mu=MU_FUEL, dP=DP_FUEL, LD=4.0,
        )
        cd_ld8, _ = cd_iterative(
            m_dot=M_DOT_FUEL_PER, rho=RHO_FUEL,
            mu=MU_FUEL, dP=DP_FUEL, LD=8.0,
        )
        # Higher L/D → lower Cd_inf → lower Cd.
        assert cd_ld8 < cd_ld4, (
            f"Expected Cd(L/D=8) < Cd(L/D=4); got {cd_ld8:.5f} vs {cd_ld4:.5f}"
        )

    def test_invalid_inputs_raise(self) -> None:
        """Non-positive inputs raise ValueError."""
        with pytest.raises(ValueError, match="m_dot"):
            cd_iterative(m_dot=0.0, rho=386.0, mu=1.4e-4, dP=3.1e6, LD=4.0)
        with pytest.raises(ValueError, match="mu"):
            cd_iterative(m_dot=0.01, rho=386.0, mu=0.0, dP=3.1e6, LD=4.0)


# ===========================================================================
# 3. LOX annulus is NOT touched by Lichtarowicz
# ===========================================================================
class TestLOXAnnulusUntouched:
    """Lichtarowicz must NOT be applied to the LOX annular passage."""

    def test_lox_cd_is_user_input(self) -> None:
        """Cd_ox is always the user-supplied value; iteration is for fuel only."""
        p = PintleInjector(
            m_dot_ox=1.1144,
            m_dot_fuel=0.3096,
            rho_ox=1116.0,
            rho_fuel=386.0,
            dP_ox=DP_FUEL,
            dP_fuel=DP_FUEL,
            Cd_ox=0.65,
            Cd_fuel=0.6,   # explicit
        )
        # LOX annulus area must use Cd_ox=0.65, not Lichtarowicz.
        from noyron.rp.injector.orifice import orifice_area_required
        A_expected = orifice_area_required(1.1144, 0.65, 1116.0, DP_FUEL)
        assert p.lox_annulus_area == pytest.approx(A_expected, rel=1e-12)

    def test_lox_cd_unchanged_when_fuel_lichtarowicz(self) -> None:
        """Even when fuel uses Lichtarowicz, Cd_ox stays at the explicit value."""
        p = PintleInjector(
            m_dot_ox=1.1144,
            m_dot_fuel=0.3096,
            rho_ox=1116.0,
            rho_fuel=386.0,
            dP_ox=DP_FUEL,
            dP_fuel=DP_FUEL,
            Cd_ox=0.65,
            Cd_fuel=None,     # triggers Lichtarowicz for fuel
            mu_fuel=MU_FUEL,
        )
        # V_ox must match what Cd_ox=0.65 would give.
        from noyron.rp.injector.orifice import orifice_area_required, injection_velocity
        A_ox_expected = orifice_area_required(1.1144, 0.65, 1116.0, DP_FUEL)
        V_ox_expected = injection_velocity(1.1144, A_ox_expected, 1116.0)
        assert p.V_ox == pytest.approx(V_ox_expected, rel=1e-12)
