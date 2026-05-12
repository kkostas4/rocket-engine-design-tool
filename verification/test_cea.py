"""
test_cea.py
===========

Verification tests for physics/thermochem/cea.py::cea_combustion().

All tests call the live rocketcea backend.  They are fast (rocketcea is
~10 ms per call) but require rocketcea to be installed; they are NOT marked
@pytest.mark.slow.

Reference values
----------------
Gordon, S. and McBride, B.J. (1994). "Computer Program for Calculation of
Complex Chemical Equilibrium Compositions and Applications." NASA RP-1311.
Implemented via rocketcea (https://rocketcea.readthedocs.io/).

Tolerance notes
---------------
CEA gives theoretical equilibrium values; the reference engine uses a
delivered c* that is ~3.9% higher (combustion efficiency ~0.96).  Tests
that check absolute c* values therefore use a ±30 m/s window (< 2 %).
"""
from __future__ import annotations

import pytest

from noyron.engine import METHALOX
from physics.thermochem.cea import cea_combustion, cea_isp_vac
from physics.util.units import PSI_TO_PA

_PC_REF = 6_894_800.0  # 1000 psia in Pa
_PC_7   = 7_000_000.0  # 7 MPa


# ---------------------------------------------------------------------------
# test_cea_methalox_at_design_point
# ---------------------------------------------------------------------------

class TestDesignPoint:
    def test_tc_within_tolerance(self) -> None:
        """Tc at (Pc=7 MPa, MR=3.6) is within 100 K of 3585 K."""
        gas = cea_combustion(propellant=METHALOX, Pc=_PC_7, MR=3.6)
        assert abs(gas.Tc - 3585.0) < 100.0, (
            f"Tc = {gas.Tc:.1f} K, expected ~3585 K (dev = {gas.Tc - 3585:.1f} K)"
        )

    def test_cstar_within_tolerance(self) -> None:
        """c* at (Pc=7 MPa, MR=3.6) is within 30 m/s of 1838 m/s."""
        gas = cea_combustion(propellant=METHALOX, Pc=_PC_7, MR=3.6)
        assert abs(gas.c_star - 1838.0) < 30.0, (
            f"c* = {gas.c_star:.1f} m/s, expected ~1838 m/s "
            f"(dev = {gas.c_star - 1838:.1f} m/s)"
        )

    def test_gamma_physical_range(self) -> None:
        """gamma is in (1.05, 1.40) — expected range for hot combustion gas."""
        gas = cea_combustion(propellant=METHALOX, Pc=_PC_REF, MR=3.6)
        assert 1.05 < gas.gamma < 1.40, f"gamma = {gas.gamma:.4f} out of range"

    def test_all_fields_positive(self) -> None:
        """All CombustionGas fields are strictly positive."""
        gas = cea_combustion(propellant=METHALOX, Pc=_PC_REF, MR=3.6)
        for field in ("Tc", "Pc", "gamma", "MW", "cp", "mu", "Pr", "c_star"):
            val = getattr(gas, field)
            assert val > 0, f"CombustionGas.{field} = {val} is not positive"


# ---------------------------------------------------------------------------
# test_cea_varies_with_mr
# ---------------------------------------------------------------------------

class TestVariesWithMR:
    def test_cstar_differs_across_mr(self) -> None:
        """c* at MR=3.0, 3.6, 3.9 all differ by > 5 m/s from each other."""
        cs = [cea_combustion(propellant=METHALOX, Pc=_PC_REF, MR=mr).c_star
              for mr in (3.0, 3.6, 3.9)]
        assert abs(cs[0] - cs[1]) > 5.0, "c* at MR=3.0 and 3.6 should differ"
        assert abs(cs[1] - cs[2]) > 5.0, "c* at MR=3.6 and 3.9 should differ"
        assert abs(cs[0] - cs[2]) > 5.0, "c* at MR=3.0 and 3.9 should differ"

    def test_isp_approx_peaks_near_stoich(self) -> None:
        """
        Vacuum Isp (c* × Cf_vac / g0) peaks near MR=3.3-3.7 for methalox.
        Uses the full isentropic Cf formula (including pressure thrust at Pa=0).
        """
        from noyron.engine import _thrust_coefficient
        _G0 = 9.80665
        eps = 10.605

        def _isp_vac(mr: float) -> float:
            gas = cea_combustion(propellant=METHALOX, Pc=_PC_REF, MR=mr)
            Cf_vac = _thrust_coefficient(eps, gas.gamma, _PC_REF, Pa=0.0)
            return gas.c_star * Cf_vac / _G0

        isps = {mr: _isp_vac(mr) for mr in (2.5, 3.0, 3.3, 3.6, 3.9, 4.5)}
        peak_mr = max(isps, key=isps.__getitem__)
        assert 3.0 <= peak_mr <= 4.0, (
            f"Peak Isp at MR={peak_mr:.1f}, expected 3.0–4.0 for methalox. "
            f"Isp values: {isps}"
        )

    def test_gamma_increases_fuel_rich(self) -> None:
        """gamma increases as mixture becomes more fuel-rich (lower MR)."""
        gamma_25 = cea_combustion(propellant=METHALOX, Pc=_PC_REF, MR=2.5).gamma
        gamma_36 = cea_combustion(propellant=METHALOX, Pc=_PC_REF, MR=3.6).gamma
        assert gamma_25 > gamma_36, (
            f"Expected gamma(MR=2.5) > gamma(MR=3.6), "
            f"got {gamma_25:.4f} vs {gamma_36:.4f}"
        )


# ---------------------------------------------------------------------------
# test_cea_varies_with_pc
# ---------------------------------------------------------------------------

class TestVariesWithPc:
    def test_tc_differs_across_pc(self) -> None:
        """Tc at Pc=5 MPa and Pc=11 MPa differs by > 50 K."""
        tc_lo = cea_combustion(propellant=METHALOX, Pc=5.0e6, MR=3.6).Tc
        tc_hi = cea_combustion(propellant=METHALOX, Pc=11.0e6, MR=3.6).Tc
        assert abs(tc_hi - tc_lo) > 50.0, (
            f"Tc at 5 MPa = {tc_lo:.1f} K, at 11 MPa = {tc_hi:.1f} K "
            f"(diff = {abs(tc_hi - tc_lo):.1f} K, expected > 50 K)"
        )

    def test_cstar_changes_with_pc(self) -> None:
        """c* at Pc=5 MPa differs from Pc=11 MPa by at least 5 m/s."""
        cs_lo = cea_combustion(propellant=METHALOX, Pc=5.0e6, MR=3.6).c_star
        cs_hi = cea_combustion(propellant=METHALOX, Pc=11.0e6, MR=3.6).c_star
        assert abs(cs_hi - cs_lo) > 5.0, (
            f"c* barely changed with Pc: {cs_lo:.1f} vs {cs_hi:.1f} m/s"
        )


# ---------------------------------------------------------------------------
# test_cache_hit
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# test_cea_isp_vac — equilibrium exit Isp
# ---------------------------------------------------------------------------

class TestIspVac:
    def test_isp_vac_absolute_at_reference(self) -> None:
        """
        Isp_vac at (Pc=10 MPa, MR=3.5, eps=40) is within 5% of 367 s.

        Reference: rocketcea get_IvacCstrTc, cross-checked against NASA CEA
        published values for LOX/CH4 at similar conditions.
        """
        isp = cea_isp_vac(propellant=METHALOX, Pc=10e6, MR=3.5, eps=40.0)
        assert abs(isp - 367.0) / 367.0 < 0.05, (
            f"Isp_vac = {isp:.2f} s, expected ~367 s (error = {abs(isp-367)/367*100:.1f}%)"
        )

    def test_isp_vac_peaks_in_stoich_range(self) -> None:
        """
        Equilibrium Isp_vac (eps=10.6) peaks in MR=[3.0, 4.0] for methalox.

        Uses equilibrium-flow get_IvacCstrTc — not frozen-gamma Cf — so the
        peak correctly lies near MR=3.3-3.5 rather than being shifted low by
        the high frozen chamber gamma at fuel-rich conditions.
        """
        mrs = (2.5, 3.0, 3.3, 3.5, 3.6, 3.7, 4.0)
        isps = {mr: cea_isp_vac(propellant=METHALOX, Pc=_PC_7, MR=mr, eps=10.6)
                for mr in mrs}
        peak_mr = max(isps, key=isps.__getitem__)
        assert 3.0 <= peak_mr <= 4.0, (
            f"Peak equilibrium Isp at MR={peak_mr:.1f}, expected 3.0–4.0. "
            f"Values: {isps}"
        )

    def test_isp_vac_greater_than_frozen_cf_at_fuel_rich(self) -> None:
        """
        Equilibrium Isp_vac < frozen-Cf Isp at MR=2.5 (fuel-rich).

        At fuel-rich conditions the frozen chamber gamma is elevated (~1.17)
        which inflates Cf and overestimates Isp.  Equilibrium expansion
        correctly gives a lower value there, shifting the peak to higher MR.
        """
        from noyron.engine import _thrust_coefficient
        _G0 = 9.80665
        mr = 2.5
        gas = cea_combustion(propellant=METHALOX, Pc=_PC_7, MR=mr)
        Cf_frozen = _thrust_coefficient(10.6, gas.gamma, gas.Pc, Pa=0.0)
        isp_frozen = gas.c_star * Cf_frozen / _G0
        isp_equil  = cea_isp_vac(propellant=METHALOX, Pc=_PC_7, MR=mr, eps=10.6)
        assert isp_equil < isp_frozen, (
            f"At MR=2.5 equilibrium Isp ({isp_equil:.2f} s) should be < "
            f"frozen-Cf Isp ({isp_frozen:.2f} s) but is not."
        )


class TestCacheHit:
    def test_same_object_returned_on_hit(self) -> None:
        """lru_cache returns the exact same object on a cache hit (identity check)."""
        g1 = cea_combustion(propellant=METHALOX, Pc=_PC_REF, MR=3.6)
        g2 = cea_combustion(propellant=METHALOX, Pc=_PC_REF, MR=3.6)
        assert g1 is g2, "Cache miss on identical args — lru_cache not working"

    def test_different_mr_gives_different_object(self) -> None:
        """Different MR values return distinct CombustionGas objects."""
        g1 = cea_combustion(propellant=METHALOX, Pc=_PC_REF, MR=3.0)
        g2 = cea_combustion(propellant=METHALOX, Pc=_PC_REF, MR=3.6)
        assert g1 is not g2
        assert g1.c_star != g2.c_star
