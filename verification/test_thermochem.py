"""
test_thermochem.py
==================

Verification tests for physics/thermochem/cea.py.

Reproduces the O/F = 3.60 row from the reference CSV sweep
(Pc = 1,000 psia, ε = 68.027) to within 1 %.

Reference values (from cea_sweep_pc1000.csv, row O/F = 3.60):
    isp_vac  = 377.97 s
    cstar    = 5979.08 ft/s → 1,822.2 m/s
    tc       = 3,561.2 K
"""

from __future__ import annotations

import pytest

from physics.thermochem.cea import CEA
from physics.util.units import PSI_TO_PA

# ---------------------------------------------------------------------------
# Reference point
# ---------------------------------------------------------------------------
PC_PA = 1_000.0 * PSI_TO_PA   # 6.8948 MPa
OF_REF = 3.60
EPS_REF = 68.027

ISP_VAC_REF = 377.97   # s
CSTAR_REF_MS = 5979.08 * 0.3048  # ft/s → m/s ≈ 1822.2 m/s
TC_REF_K = 3561.2      # K

TOL = 0.01   # 1 %


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cea() -> CEA:
    return CEA()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestCEARefPoint:
    def test_isp_vac(self, cea: CEA) -> None:
        """isp_vac at O/F=3.6 matches CSV row to within 1 %."""
        val = cea.isp_vac(PC_PA, OF_REF, EPS_REF)
        assert abs(val - ISP_VAC_REF) / ISP_VAC_REF < TOL, (
            f"isp_vac = {val:.3f} s, expected {ISP_VAC_REF} s "
            f"(dev = {100*(val-ISP_VAC_REF)/ISP_VAC_REF:.3f} %)"
        )

    def test_cstar(self, cea: CEA) -> None:
        """cstar at O/F=3.6 matches CSV row (converted to m/s) to within 1 %."""
        val = cea.cstar(PC_PA, OF_REF)
        assert abs(val - CSTAR_REF_MS) / CSTAR_REF_MS < TOL, (
            f"cstar = {val:.2f} m/s, expected {CSTAR_REF_MS:.2f} m/s "
            f"(dev = {100*(val-CSTAR_REF_MS)/CSTAR_REF_MS:.3f} %)"
        )

    def test_tc(self, cea: CEA) -> None:
        """tc at O/F=3.6 matches CSV row to within 1 %."""
        val = cea.tc(PC_PA, OF_REF)
        assert abs(val - TC_REF_K) / TC_REF_K < TOL, (
            f"tc = {val:.1f} K, expected {TC_REF_K} K "
            f"(dev = {100*(val-TC_REF_K)/TC_REF_K:.3f} %)"
        )

    def test_tc_physical_range(self, cea: CEA) -> None:
        """Chamber temperature is in physically plausible range."""
        val = cea.tc(PC_PA, OF_REF)
        assert 2_000.0 < val < 4_500.0, f"tc = {val:.1f} K is outside 2000–4500 K"


class TestCEAInterpolation:
    def test_peak_isp_near_of_36(self, cea: CEA) -> None:
        """Peak Isp_vac occurs near O/F = 3.4–3.6 (methalox stoich range)."""
        isp_34 = cea.isp_vac(PC_PA, 3.4, EPS_REF)
        isp_36 = cea.isp_vac(PC_PA, 3.6, EPS_REF)
        isp_40 = cea.isp_vac(PC_PA, 4.0, EPS_REF)
        assert isp_36 >= isp_40, "Isp should decrease beyond O/F = 3.6"
        assert isp_36 >= isp_34 or abs(isp_36 - isp_34) < 1.0, (
            "Peak Isp should be near O/F = 3.4–3.6"
        )

    def test_cstar_monotone_below_peak(self, cea: CEA) -> None:
        """c* increases with O/F from 1 → ~3 (fuel-rich to stoich)."""
        c1 = cea.cstar(PC_PA, 1.0)
        c2 = cea.cstar(PC_PA, 2.0)
        c3 = cea.cstar(PC_PA, 3.0)
        assert c1 < c2 < c3, "c* should increase from O/F=1 to 3"


class TestCEAGuardRails:
    def test_wrong_pc_raises(self, cea: CEA) -> None:
        with pytest.raises(NotImplementedError):
            cea.isp_vac(500.0 * PSI_TO_PA, OF_REF, EPS_REF)

    def test_wrong_eps_raises(self, cea: CEA) -> None:
        with pytest.raises(NotImplementedError):
            cea.isp_vac(PC_PA, OF_REF, 10.605)

    def test_out_of_range_of_raises(self, cea: CEA) -> None:
        with pytest.raises(ValueError):
            cea.isp_vac(PC_PA, 10.0, EPS_REF)

    def test_isp_sl_not_implemented(self, cea: CEA) -> None:
        with pytest.raises(NotImplementedError):
            cea.isp_sl(PC_PA, OF_REF, EPS_REF)
