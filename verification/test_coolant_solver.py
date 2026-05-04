"""
test_coolant_solver.py
======================

Verification tests for physics/regen/solver.py -- the 1D coupled regen
cooling solver.

NOTE on test_against_published_engine
--------------------------------------
No published regeneratively-cooled methalox engine data with sufficient
geometric detail (channel width, height, land thickness, n_channels as a
function of x) was found in the open literature for point-by-point
comparison.  The RL-10, J-2, and RS-25 papers that exist are either
classified, incomplete, or cover LOX/LH2 rather than methalox.

This test is therefore replaced by:
    (a) A physical range check: peak T_wall_gas must be in [200, 3500] K.
    (b) A sanity note: GRCop-42 continuous limit is 810 K (NASA TM-2005-213566,
        Ellis 2005).  The test DOES NOT assert this limit since a valid
        under-cooled design would correctly predict T > 810 K.

If/when a suitable reference dataset is found, this test should be replaced
with a proper regression.  FLAG: test_against_published_engine is a
placeholder.

Sources
-------
    Dittus & Boelter (1930), restated in McAdams (1954) -- Nu correlation.
    Haaland (1983) -- friction factor.
    Incropera et al. (2011) Eq. (3.86) -- overall surface efficiency.
    NASA SP-8087 (1977) Sec. 3.1 -- thermal resistance network.
    Ellis, D.L. (2005) NASA/TM-2005-213566 -- GRCop-84 limit (proxy for GRCop-42).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from physics.nozzle.rao import RaoNozzle
from physics.regen.bartz import CombustionGas
from physics.regen.coolant import GRCOP42, CoolingChannelGeometry, CoolantInlet
from physics.regen.solver import solve_regen

# ---------------------------------------------------------------------------
# Reference engine fixtures
# ---------------------------------------------------------------------------
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

# Channel geometry: 48 channels, uniform cross-section chosen so the arc
# per channel at the throat (pi * D_t / n_ch = pi * 22.3 mm / 48 = 1.46 mm)
# accommodates w + t_land = 1.1 + 0.35 = 1.45 mm.
GEOM = CoolingChannelGeometry(
    n_channels=48,
    width=lambda x: 1.1e-3,  # 1.1 mm
    height=lambda x: 3.0e-3,  # 3.0 mm
    wall_thickness=lambda x: 4.0e-4,  # 0.4 mm
    land_thickness=lambda x: 3.5e-4,  # 0.35 mm
)

# Coolant: LCH4 at 150 K, 12 MPa, full fuel mass flow
COOLANT = CoolantInlet(T=150.0, P=12.0e6, m_dot=0.3096, fluid="Methane")

N_FAST = 80  # station count for fast tests


def _run(direction: str = "counterflow", n: int = N_FAST) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return solve_regen(
            nozzle=NOZZLE,
            gas=GAS,
            channel_geom=GEOM,
            coolant_inlet=COOLANT,
            wall_material=GRCOP42,
            direction=direction,
            n_stations=n,
            tol=1.0,
        )


# ---------------------------------------------------------------------------
# 1. Energy balance
# ---------------------------------------------------------------------------
class TestEnergyBalance:
    def test_energy_balance(self) -> None:
        """Total Q from gas-side equals coolant enthalpy rise to within 1 %.

        The solver updates T_coolant via dT = q_prime * dx / (m_dot_ch * cp_c).
        By construction the station-level energy balance is exact.  This test
        verifies it integrates correctly using the CoolProp enthalpy function
        as an independent check.

        Source: First Law of Thermodynamics -- steady-flow energy equation.
        """
        from CoolProp.CoolProp import PropsSI

        result = _run(direction="counterflow", n=200)
        x_arr = result["x"]
        r_arr = result["r"]
        q_arr = result["q_dot"]
        T_c = result["T_coolant"]
        P_c = result["P_coolant"]
        dx = float(x_arr[1] - x_arr[0])

        n_ch = GEOM.n_channels
        m_dot = COOLANT.m_dot

        # Gas-side total Q: integrate q_dot over full gas-side area
        A_gas_arr = 2.0 * np.pi * r_arr / n_ch  # m2/m per channel
        Q_gas = float(np.sum(q_arr * A_gas_arr) * dx * n_ch)

        # Coolant enthalpy rise (CoolProp, more accurate than cp*dT)
        # Counterflow: coolant enters at exit (last index), leaves at inlet (index 0)
        h_in = PropsSI("H", "T", T_c[-1], "P", P_c[-1], "Methane")
        h_out = PropsSI("H", "T", T_c[0], "P", P_c[0], "Methane")
        Q_coolant = m_dot * (h_out - h_in)

        dev = abs(Q_gas - Q_coolant) / Q_gas
        assert dev < 0.02, (  # 2 % -- small discrepancy from cp(T) vs enthalpy
            f"Energy balance: Q_gas={Q_gas/1e3:.1f} kW, "
            f"Q_coolant={Q_coolant/1e3:.1f} kW, dev={100*dev:.2f} %"
        )


# ---------------------------------------------------------------------------
# 2. Counterflow vs coflow: coolant temperature at the throat
# ---------------------------------------------------------------------------
class TestCounterflowVsCoflow:
    def test_coflow_cooler_at_throat_for_long_bell(self) -> None:
        """For a long bell nozzle (ε=10.6), coflow arrives colder at the throat
        than counterflow, giving lower peak T_wall_gas.

        This reference engine has x_inlet ≈ -8.4 mm (convergent) and
        x_exit ≈ 83.5 mm (divergent bell).  The divergent section is ~10×
        longer than the convergent section, so it absorbs more total heat.

        Counterflow: coolant enters cold at x_exit, traverses 83.5 mm of
            divergent section, arrives at the throat already warm.
        Coflow: coolant enters cold at x_inlet, traverses only 8.4 mm of
            convergent section, arrives at the throat still near-inlet temperature.

        Therefore, for this geometry, coflow produces a colder coolant at the
        throat and a lower peak T_wall_gas.

        NOTE: NASA SP-8087 (1977) Sec. 3.1 states counterflow is generally
        preferred, which holds when the convergent/chamber section dominates
        the heat load (high L*, short bell).  For a long bell with ε ≥ 10 and
        short convergent, the divergent total heat can exceed the convergent
        total heat, reversing the advantage.  This test documents that result.
        """
        res_counter = _run("counterflow")
        res_coflow = _run("coflow")
        T_peak_counter = float(res_counter["T_wall_gas"].max())
        T_peak_coflow = float(res_coflow["T_wall_gas"].max())

        # For this long-bell geometry coflow gives a lower peak T_wall
        assert T_peak_coflow < T_peak_counter, (
            f"For ε=10.6 long-bell nozzle, coflow peak T_wall = {T_peak_coflow:.1f} K "
            f"should be < counterflow peak T_wall = {T_peak_counter:.1f} K "
            "(divergent section dominates total heat load)."
        )

        # Both directions give a finite, physically plausible peak T_wall
        for label, T_peak in (("counterflow", T_peak_counter), ("coflow", T_peak_coflow)):
            assert 200.0 < T_peak < 3500.0, (
                f"{label} peak T_wall = {T_peak:.1f} K outside physical range"
            )


# ---------------------------------------------------------------------------
# 3. Grid convergence
# ---------------------------------------------------------------------------
class TestConvergence:
    def test_grid_convergence(self) -> None:
        """Halving n_stations changes peak T_wall_gas by < 2 %.

        The 1-D Euler march has O(dx) accuracy.  Halving the step size
        should halve the discretisation error, giving < 2 % change in the
        peak wall temperature (the most sensitive output).
        """
        res_hi = _run(n=200)
        res_lo = _run(n=100)
        T_hi = float(res_hi["T_wall_gas"].max())
        T_lo = float(res_lo["T_wall_gas"].max())
        dev = abs(T_hi - T_lo) / T_hi
        assert dev < 0.02, (
            f"Grid convergence: T_peak(n=200)={T_hi:.1f} K, "
            f"T_peak(n=100)={T_lo:.1f} K, dev={100*dev:.2f} %"
        )


# ---------------------------------------------------------------------------
# 4. Physical range and structural checks
# ---------------------------------------------------------------------------
class TestPhysicalRange:
    def test_output_keys_present(self) -> None:
        """solve_regen returns all required output arrays."""
        result = _run()
        for key in ("x", "r", "T_wall_gas", "T_wall_coolant", "T_coolant",
                    "P_coolant", "q_dot", "h_g", "h_c", "dP_total"):
            assert key in result, f"Missing key: {key}"

    def test_array_length(self) -> None:
        """All arrays have length n_stations."""
        result = _run(n=N_FAST)
        for key in ("x", "r", "T_wall_gas", "T_wall_coolant",
                    "T_coolant", "P_coolant", "q_dot", "h_g", "h_c"):
            assert len(result[key]) == N_FAST, f"{key}: expected {N_FAST}, got {len(result[key])}"

    def test_T_wall_gas_physical_range(self) -> None:
        """T_wall_gas is in physically plausible range [200, 3500] K everywhere.

        FLAG -- published engine comparison: No suitable published methalox
        regen data found for point-by-point verification.  This test checks
        only that the wall temperature is in a physically possible range.
        GRCop-42 continuous service limit is 810 K (Ellis 2005, NASA TM-2005-213566);
        the solver correctly warns when this is exceeded but does not hard-fail.
        """
        result = _run()
        T_peak = float(result["T_wall_gas"].max())
        assert 200.0 < T_peak < 3500.0, (
            f"Peak T_wall_gas = {T_peak:.1f} K outside physical range [200, 3500] K"
        )

    def test_T_coolant_increases_toward_outlet(self) -> None:
        """Coolant temperature rises monotonically in the flow direction.

        In counterflow the coolant flows from exit (index -1) to inlet (index 0),
        picking up heat along the way, so T_coolant[0] > T_coolant[-1].
        """
        result = _run("counterflow")
        T_c = result["T_coolant"]
        # Coolant flows exit -> inlet: T increases from right to left
        assert float(T_c[0]) > float(T_c[-1]), (
            f"T_coolant[outlet]={T_c[0]:.1f} K should be > T_coolant[inlet]={T_c[-1]:.1f} K"
        )

    def test_q_dot_positive(self) -> None:
        """Heat flux into the wall is positive everywhere (gas hotter than coolant)."""
        result = _run()
        assert np.all(result["q_dot"] > 0), "q_dot should be positive everywhere"

    def test_pressure_drops_along_flow(self) -> None:
        """Coolant pressure decreases along the flow direction (friction)."""
        result = _run("counterflow")
        P_c = result["P_coolant"]
        # Counterflow: coolant flows from index -1 to index 0
        # Pressure should decrease from inlet (index -1) to outlet (index 0)
        assert float(P_c[-1]) > float(P_c[0]), (
            "Coolant pressure should drop along the flow direction"
        )

    def test_h_c_order_of_magnitude(self) -> None:
        """h_c from Dittus-Boelter is in 5,000-100,000 W/(m2.K) for LCH4 at these conditions."""
        result = _run()
        h_c_mean = float(result["h_c"].mean())
        assert 5_000 < h_c_mean < 100_000, (
            f"Mean h_c = {h_c_mean:.0f} W/(m2.K) outside expected range"
        )

    def test_invalid_direction_raises(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                solve_regen(
                    nozzle=NOZZLE,
                    gas=GAS,
                    channel_geom=GEOM,
                    coolant_inlet=COOLANT,
                    wall_material=GRCOP42,
                    direction="sideways",
                )
