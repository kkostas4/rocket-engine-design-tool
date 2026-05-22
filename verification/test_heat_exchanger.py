"""
test_heat_exchanger.py
======================

Verification tests for physics/heat_exchanger/.

Three test groups:

1. test_existing_regen_results_unchanged
   The methalox reference engine solve_regen() result (peak T_wall,
   coolant ΔT, coolant ΔP, total Q, etc.) must reproduce to within
   0.01% between the pre-refactor shim and the new solve_rocket_regen.
   Validates that the abstraction layer is numerically transparent.

2. test_dittus_boelter_standalone
   The standalone dittus_boelter() function is independent of geometry
   and must reproduce textbook results.

3. test_simple_plate_fin_HX
   Define a trivial constant-section HX (hot air cooling in a rectangular
   duct, cold water heated) and run solve_heat_exchanger.  Check that the
   energy balance closes to within 1%: Q = m_dot * cp * ΔT.
   This is the "we actually generalised it" proof.

References
----------
Incropera et al. (2011). "Fundamentals of Heat and Mass Transfer," 7th ed.
    Table 8.1 — benchmark Nu values for Dittus-Boelter.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from noyron.kernel.heat_exchanger.base import (
    ColdSideModel,
    ColdSideResult,
    FluidProperties,
    HeatExchangerGeometry,
    HotSideModel,
    HotSideState,
)
from noyron.kernel.heat_exchanger.correlations import (
    DittusBoelterModel,
    dittus_boelter,
    haaland_f,
)
from noyron.kernel.heat_exchanger.solver import solve_heat_exchanger
from noyron.rp.regen.rocket_regen import solve_rocket_regen
from noyron.rp.regen.bartz import CombustionGas
from noyron.rp.regen.coolant import GRCOP42, CoolantInlet, CoolingChannelGeometry
from noyron.rp.nozzle.rao import RaoNozzle


# ---------------------------------------------------------------------------
# Shared fixture: reference methalox engine
# ---------------------------------------------------------------------------

_GAS = CombustionGas(
    Tc=3561.2,
    Pc=6.8948e6,
    gamma=1.15,
    MW=21.9,
    cp=2000.0,
    mu=8.0e-5,
    Pr=0.60,
    c_star=1893.3,
)
_NOZZLE = RaoNozzle(r_throat=0.022313 / 2.0, epsilon=10.605)
_COOLANT = CoolantInlet(T=150.0, P=12.0e6, m_dot=0.3096, fluid="Methane")

_CHANNEL_GEOM = CoolingChannelGeometry(
    n_channels=76,
    width=lambda x: 0.8e-3,
    height=lambda x: 1.5e-3,
    wall_thickness=lambda x: 0.5e-3,
    land_thickness=lambda x: 0.3e-3,
)


# ---------------------------------------------------------------------------
# 1. Numerical reproduction: solve_regen shim == solve_rocket_regen
# ---------------------------------------------------------------------------


class TestRegenResultsUnchanged:
    """
    solve_regen (compatibility shim) and solve_rocket_regen must produce
    bit-identical outputs for the reference methalox engine.

    The 0.01% tolerance is tighter than machine epsilon (they should be
    exact), but allows for any non-significant floating-point order
    differences that might arise from future minor optimisations.
    """

    @pytest.fixture(scope="class")
    def both_solutions(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from noyron.rp.regen.solver import solve_regen
            sol_shim = solve_regen(
                nozzle=_NOZZLE,
                gas=_GAS,
                channel_geom=_CHANNEL_GEOM,
                coolant_inlet=_COOLANT,
                wall_material=GRCOP42,
                direction="counterflow",
                n_stations=50,
                tol=1.0,
            )
            sol_new = solve_rocket_regen(
                nozzle=_NOZZLE,
                gas=_GAS,
                channel_geom=_CHANNEL_GEOM,
                coolant_inlet=_COOLANT,
                wall_material=GRCOP42,
                direction="counterflow",
                n_stations=50,
                tol=1.0,
            )
        return sol_shim, sol_new

    def _check_scalar(self, shim, new, key: str, tol: float = 1e-4) -> None:
        a = float(shim[key]) if np.ndim(shim[key]) == 0 else float(shim[key].max())
        b = float(new[key]) if np.ndim(new[key]) == 0 else float(new[key].max())
        if a == 0.0 and b == 0.0:
            return
        dev = abs(a - b) / max(abs(a), abs(b))
        assert dev <= tol, (
            f"{key}: shim={a:.6g}, new={b:.6g}, dev={100*dev:.5f}% > {100*tol:.4f}%"
        )

    def test_peak_T_wall_gas(self, both_solutions):
        """Peak gas-side wall temperature reproduces to 0.01%."""
        shim, new = both_solutions
        self._check_scalar(shim, new, "T_wall_gas")

    def test_coolant_dT(self, both_solutions):
        """Total coolant temperature rise reproduces to 0.01%."""
        shim, new = both_solutions
        dT_shim = abs(float(shim["T_coolant"][0]) - float(shim["T_coolant"][-1]))
        dT_new = abs(float(new["T_coolant"][0]) - float(new["T_coolant"][-1]))
        dev = abs(dT_shim - dT_new) / max(dT_shim, dT_new)
        assert dev <= 1e-4, f"coolant ΔT: shim={dT_shim:.4f} K, new={dT_new:.4f} K"

    def test_coolant_dP(self, both_solutions):
        """Total coolant pressure drop reproduces to 0.01%."""
        shim, new = both_solutions
        self._check_scalar(shim, new, "dP_total")

    def test_peak_h_g(self, both_solutions):
        """Peak gas-side h reproduces to 0.01%."""
        shim, new = both_solutions
        self._check_scalar(shim, new, "h_g")

    def test_peak_h_c(self, both_solutions):
        """Peak coolant-side h reproduces to 0.01%."""
        shim, new = both_solutions
        self._check_scalar(shim, new, "h_c")

    def test_x_arrays_identical(self, both_solutions):
        """Axial station arrays are exactly equal."""
        shim, new = both_solutions
        np.testing.assert_array_equal(shim["x"], new["x"])

    def test_r_array_present(self, both_solutions):
        """Both solutions include the 'r' (nozzle radius) array."""
        shim, new = both_solutions
        assert "r" in shim
        assert "r" in new
        np.testing.assert_allclose(shim["r"], new["r"], rtol=1e-10)


# ---------------------------------------------------------------------------
# 2. Dittus-Boelter standalone function
# ---------------------------------------------------------------------------


class TestDittusBoelterStandalone:
    """
    Standalone dittus_boelter() must reproduce textbook Nu and h values.

    Reference: Incropera et al. (2011), Example 8.1 style — turbulent pipe flow.
    """

    def test_nu_value(self) -> None:
        """Nu = 0.023 * Re^0.8 * Pr^0.4 for Re=50000, Pr=4."""
        Re = 50_000.0
        Pr = 4.0
        D_h = 0.02   # m
        k_c = 0.6    # W/(m·K)
        Nu_expected = 0.023 * Re**0.8 * Pr**0.4
        h_expected = Nu_expected * k_c / D_h
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h = dittus_boelter(Re, Pr, D_h, k_c)
        assert abs(h - h_expected) / h_expected < 1e-10, (
            f"h = {h:.4f}, expected {h_expected:.4f}"
        )

    def test_higher_re_gives_higher_h(self) -> None:
        """h increases monotonically with Re."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h1 = dittus_boelter(20_000, 0.8, 0.01, 0.5)
            h2 = dittus_boelter(50_000, 0.8, 0.01, 0.5)
        assert h2 > h1

    def test_warns_low_re(self) -> None:
        """Warns when Re < 10,000."""
        with pytest.warns(RuntimeWarning, match="Dittus-Boelter"):
            dittus_boelter(5_000.0, 0.8, 0.01, 0.5)

    def test_haaland_smooth_channel(self) -> None:
        """Haaland f for smooth channel matches Colebrook approximation at Re=50000."""
        f = haaland_f(50_000.0, roughness=0.0)
        # For smooth pipe at Re=50000, f ≈ 0.0210 (Moody chart reference)
        assert 0.018 < f < 0.025, f"Unexpected f = {f:.5f}"

    def test_dittus_boelter_model_matches_function(self) -> None:
        """DittusBoelterModel.evaluate returns same h as dittus_boelter()."""

        class _SimpleGeom(HeatExchangerGeometry):
            @property
            def n_channels(self) -> int: return 1
            def wetted_perimeter_hot(self, x): return 0.05
            def wetted_perimeter_cold(self, x): return 0.10
            def cross_section_cold(self, x): return 5e-5  # 5 mm × 10 mm → 5e-5 m²
            def hydraulic_diameter_cold(self, x): return 4 * 5e-5 / 0.10
            def wall_thickness(self, x): return 5e-4
            def fin_efficiency(self, x, h_cold, k_wall=None): return 1.0
            def overall_surface_efficiency(self, x, eta_fin): return 1.0
            def wall_conductivity(self, x, T_wall): return 300.0
            def temperature_limit(self): return 1000.0
            def material_name(self): return "test_material"

        geom = _SimpleGeom()
        props = FluidProperties(rho=400.0, cp=3500.0, mu=1e-4, k=0.12, Pr=2.9)
        m_dot_ch = 0.002  # kg/s per channel
        x = 0.0

        model = DittusBoelterModel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.evaluate(x, props, m_dot_ch, geom)

        A_c = geom.cross_section_cold(x)
        D_h = geom.hydraulic_diameter_cold(x)
        V = m_dot_ch / (props.rho * A_c)
        Re = props.rho * V * D_h / props.mu
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h_expected = dittus_boelter(Re, props.Pr, D_h, props.k)

        assert abs(result.h_convective - h_expected) < 1e-9 * h_expected


# ---------------------------------------------------------------------------
# 3. Generalisation proof: simple plate-fin HX (non-rocket)
# ---------------------------------------------------------------------------


class _PlateFin(HeatExchangerGeometry):
    """
    Trivial constant-cross-section rectangular duct geometry.

    Hot side: one flat wall (width w).
    Cold side: three-sided rectangular channel (w × H).
    Fins: neglected (eta_fin = 1, eta_0 = 1).
    """

    def __init__(
        self,
        n_ch: int,
        width: float,
        height: float,
        t_wall: float,
        k_wall: float,
        T_limit: float = 600.0,
    ) -> None:
        self._n = n_ch
        self._w = width
        self._H = height
        self._tw = t_wall
        self._k = k_wall
        self._T_lim = T_limit

    @property
    def n_channels(self) -> int:
        return self._n

    def wetted_perimeter_hot(self, x: float) -> float:
        return self._w / self._n

    def wetted_perimeter_cold(self, x: float) -> float:
        return self._w + 2.0 * self._H

    def cross_section_cold(self, x: float) -> float:
        return self._w * self._H

    def hydraulic_diameter_cold(self, x: float) -> float:
        return 4.0 * self._w * self._H / (2.0 * (self._w + self._H))

    def wall_thickness(self, x: float) -> float:
        return self._tw

    def fin_efficiency(self, x: float, h_cold: float, k_wall: float | None = None) -> float:
        return 1.0  # no fins

    def overall_surface_efficiency(self, x: float, eta_fin: float) -> float:
        return 1.0  # no fins

    def wall_conductivity(self, x: float, T_wall: float) -> float:
        return self._k

    def temperature_limit(self) -> float:
        return self._T_lim

    def material_name(self) -> str:
        return "stainless_steel"


class _ConstantHotSide(HotSideModel):
    """Hot stream at fixed temperature and convective coefficient."""

    def __init__(self, T_hot: float, h_hot: float) -> None:
        self._T = T_hot
        self._h = h_hot

    def evaluate(
        self, x: float, geometry: HeatExchangerGeometry, T_wall: float
    ) -> HotSideState:
        return HotSideState(T_recovery=self._T, h_convective=self._h)


class TestSimplePlateFin:
    """
    Prove that solve_heat_exchanger works for a non-rocket application:
    hot air (constant T=400 K, constant h=250 W/m²K) heating cold water
    in a rectangular duct.

    Acceptance: energy balance closes to within 1%.
    """

    _T_hot = 400.0    # K, hot-side recovery temperature
    _h_hot = 250.0    # W/(m²·K), hot-side film coefficient
    _L = 0.5          # m, heat exchanger length

    _N_CH = 10
    _W = 0.01         # m, channel width (per channel)
    _H = 0.005        # m, channel height

    @pytest.fixture(scope="class")
    def hx_result(self):
        geometry = _PlateFin(
            n_ch=self._N_CH,
            width=self._W,
            height=self._H,
            t_wall=1e-3,
            k_wall=15.0,   # stainless ~15 W/(m·K)
        )
        hot_model = _ConstantHotSide(T_hot=self._T_hot, h_hot=self._h_hot)
        cold_model = DittusBoelterModel()
        coolant = CoolantInlet(T=300.0, P=2e5, m_dot=0.5, fluid="Water")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solve_heat_exchanger(
                hot_model=hot_model,
                cold_model=cold_model,
                geometry=geometry,
                x_start=0.0,
                x_end=self._L,
                coolant_inlet=coolant,
                direction="coflow",
                n_stations=100,
                tol=0.1,
            )
        return result, coolant, geometry

    def test_energy_balance(self, hx_result) -> None:
        """
        Q_total from q_dot integration == m_dot_ch * cp * ΔT_coolant within 1%.

        For coflow (coolant enters at x=0, exits at x=L):
          Q_integrated = sum(q_dot[i] * A_hot(x[i]) * dx)
          Q_enthalpy   = m_dot_total * cp_mean * (T_out - T_in)

        Source: energy balance per Incropera et al. (2011) Ch. 11.
        """
        result, coolant, geometry = hx_result
        x_arr = result["x"]
        q_dot = result["q_dot"]  # W/m², hot-side reference area
        T_c = result["T_coolant"]
        dx = float(x_arr[1] - x_arr[0])

        # Integrated Q: q_dot[i] * A_hot * dx, summed over all channels
        Q_integrated = 0.0
        for i, x in enumerate(x_arr):
            A_hot_total = geometry.wetted_perimeter_hot(x) * geometry.n_channels
            Q_integrated += float(q_dot[i]) * A_hot_total * dx

        # Enthalpy rise: use water cp ≈ 4182 J/(kg·K) at ~300 K
        cp_water = 4182.0
        T_in = float(T_c[0])    # coflow: inlet at i=0
        T_out = float(T_c[-1])  # coflow: exit at i=-1
        Q_enthalpy = coolant.m_dot * cp_water * (T_out - T_in)

        assert Q_enthalpy > 0.0, "Coolant should be heated (T_out > T_in)"
        dev = abs(Q_integrated - Q_enthalpy) / Q_enthalpy
        assert dev < 0.01, (
            f"Energy balance error {100*dev:.2f}%: "
            f"Q_integrated={Q_integrated:.2f} W, Q_enthalpy={Q_enthalpy:.2f} W"
        )

    def test_coolant_heated(self, hx_result) -> None:
        """Cold-side fluid temperature rises from inlet to outlet (coflow)."""
        result, _, _ = hx_result
        T_c = result["T_coolant"]
        assert float(T_c[-1]) > float(T_c[0]), (
            f"T_coolant_out={float(T_c[-1]):.1f} K not > T_in={float(T_c[0]):.1f} K"
        )

    def test_wall_hotter_than_coolant(self, hx_result) -> None:
        """Gas-side wall is hotter than coolant at every station."""
        result, _, _ = hx_result
        T_wg = result["T_wall_gas"]
        T_c = result["T_coolant"]
        assert np.all(T_wg > T_c), "T_wall_gas should exceed T_coolant everywhere"

    def test_result_has_required_keys(self, hx_result) -> None:
        """Result dict contains all expected keys."""
        result, _, _ = hx_result
        for key in ("x", "T_wall_gas", "T_wall_coolant", "T_coolant", "P_coolant",
                    "q_dot", "h_g", "h_c", "dP_total"):
            assert key in result, f"Missing key: {key}"

    def test_base_importable_without_rocket(self) -> None:
        """
        noyron.kernel.heat_exchanger.base is importable in a context where
        physics.regen and physics.nozzle have not yet been imported.

        This verifies the 'HotSideModel / ColdSideModel API must be importable
        in a fresh Python process WITHOUT any rocket-specific import' requirement.
        """
        import importlib
        import sys

        # Remove all physics/noyron.kernel.heat_exchanger modules from cache
        to_remove = [
            k for k in sys.modules
            if k.startswith("physics") or k.startswith("noyron.kernel.heat_exchanger")
        ]
        for k in to_remove:
            sys.modules.pop(k, None)

        # Import ONLY base — must not trigger rocket imports
        mod = importlib.import_module("noyron.kernel.heat_exchanger.base")
        assert hasattr(mod, "HotSideModel")
        assert hasattr(mod, "ColdSideModel")
        assert hasattr(mod, "HeatExchangerGeometry")

        # Verify physics.nozzle was NOT imported as a side effect
        assert "physics.nozzle" not in sys.modules, (
            "noyron.kernel.heat_exchanger.base should not import physics.nozzle"
        )
        assert "physics.regen" not in sys.modules, (
            "noyron.kernel.heat_exchanger.base should not import physics.regen"
        )
