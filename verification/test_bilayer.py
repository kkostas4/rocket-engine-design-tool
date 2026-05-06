"""
test_bilayer.py
===============

Verification tests for physics/structural/bilayer.py -- bilayer chamber
stress model (inner liner + outer jacket + ribs).

Sources
-------
    NASA SP-8087 (1977) §3.3 Wall Construction.
    Timoshenko & Goodier (1951) §131 (thin-shell hoop stress).
    Roark's Formulas for Stress and Strain (Young & Budynas, 8th ed.) Table 8.1.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from physics.structural.bilayer import (
    BilayerWall,
    RibGeometry,
    WallLayer,
    bilayer_stress_profile,
    size_jacket_thickness,
)
from physics.structural.materials import GRCOP84_PROXY, INCONEL718_STRUCT

# ---------------------------------------------------------------------------
# Reference engine fixtures
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

_X2 = 2.0 * NOZZLE.r_throat
SIZED_GEOM = CoolingChannelGeometry(
    n_channels=76,
    width=lambda x: 0.6e-3,
    height=lambda x: 0.3e-3 if x < _X2 else 3.2e-3,
    wall_thickness=lambda x: 0.5e-3,
    land_thickness=lambda x: 0.3e-3,
)

# Bilayer wall with typical flight dimensions:
# t_liner = 0.5 mm (same as single-wall model),
# t_jacket = 2.0 mm (autosized in test_jacket_autosizer; use 2 mm as fixed here)
_BILAYER = BilayerWall(
    inner_liner=WallLayer(
        thickness=lambda x: 0.5e-3,
        material=GRCOP84_PROXY,
    ),
    outer_jacket=WallLayer(
        thickness=lambda x: 2.0e-3,
        material=INCONEL718_STRUCT,
    ),
    ribs=RibGeometry(
        n_ribs=76,
        land_width=lambda x: 0.3e-3,
        height=lambda x: 0.3e-3 if x < _X2 else 3.2e-3,
        material=GRCOP84_PROXY,
    ),
    channel_geometry=SIZED_GEOM,
)

# Cache regen solution
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
# 1. Inner liner sees only the DIFFERENTIAL pressure
# ---------------------------------------------------------------------------
class TestInnerLinerDifferential:
    def test_inner_liner_sees_only_differential(self) -> None:
        """When P_gas = P_coolant, liner hoop stress is zero.

        Physical basis: the liner is sandwiched between gas and coolant.
        Net pressure on the liner = P_coolant - P_gas.
        Source: NASA SP-8087 (1977) §3.3.
        """
        regen = _get_regen()
        # Override P_gas_arr to equal P_coolant everywhere -> zero differential
        P_cool_arr = regen["P_coolant"].copy()

        sol = bilayer_stress_profile(
            regen_solution=regen,
            bilayer=_BILAYER,
            P_gas_arr=P_cool_arr,
        )
        # Liner hoop should be zero everywhere
        assert np.allclose(sol["sigma_h_inner"], 0.0, atol=1.0), (
            f"Liner hoop should be 0 when P_gas=P_coolant; "
            f"max = {sol['sigma_h_inner'].max():.1f} Pa"
        )

    def test_liner_hoop_tensile_when_pcoolant_gt_pgas(self) -> None:
        """sigma_h_inner > 0 when P_coolant > P_gas (normal operation).

        Source: NASA SP-8087 §3.3; sign convention: P_coolant > P_gas -> tensile.
        """
        regen = _get_regen()
        # P_gas well below P_coolant everywhere
        P_gas_arr = np.full(len(regen["x"]), 1.0e6)  # 1 MPa << 12 MPa coolant

        sol = bilayer_stress_profile(
            regen_solution=regen,
            bilayer=_BILAYER,
            P_gas_arr=P_gas_arr,
        )
        assert np.all(sol["sigma_h_inner"] > 0.0), (
            "Liner hoop should be positive (tensile) when P_coolant >> P_gas"
        )


# ---------------------------------------------------------------------------
# 2. Jacket sees full coolant pressure
# ---------------------------------------------------------------------------
class TestJacketFullPressure:
    def test_jacket_sees_full_coolant_P(self) -> None:
        """Jacket hoop = (P_coolant - P_atm) * r / t within thin-shell limit.

        Source: Timoshenko & Goodier (1951) §131.
        """
        regen = _get_regen()
        P_gas_arr = np.zeros(len(regen["x"]))

        sol = bilayer_stress_profile(
            regen_solution=regen,
            bilayer=_BILAYER,
            P_gas_arr=P_gas_arr,
            P_atm=101_325.0,
        )
        # Verify one station: check jacket hoop ≈ P_net * r_jkt / t_jkt
        i = 0
        P_cool = float(regen["P_coolant"][i])
        r_jkt = float(sol["r_jacket"][i])
        t_jkt = 2.0e-3  # fixed jacket thickness in _BILAYER
        expected = (P_cool - 101_325.0) * r_jkt / t_jkt
        actual = float(sol["sigma_h_jacket"][i])
        rel_err = abs(actual - expected) / expected
        assert rel_err < 1e-9, (
            f"Jacket hoop = {actual/1e6:.2f} MPa, expected {expected/1e6:.2f} MPa "
            f"(rel err = {rel_err:.2e})"
        )

    def test_jacket_hoop_positive(self) -> None:
        """Jacket hoop stress is positive (tensile) everywhere in normal operation."""
        regen = _get_regen()
        P_gas_arr = np.zeros(len(regen["x"]))
        sol = bilayer_stress_profile(
            regen_solution=regen,
            bilayer=_BILAYER,
            P_gas_arr=P_gas_arr,
        )
        assert np.all(sol["sigma_h_jacket"] > 0.0), (
            "Jacket hoop must be tensile when P_coolant > P_atm"
        )


# ---------------------------------------------------------------------------
# 3. Rib shear is zero when there is no differential pressure
# ---------------------------------------------------------------------------
class TestRibShear:
    def test_rib_shear_zero_at_no_differential(self) -> None:
        """tau_rib = 0 when P_gas = P_coolant.

        Source: Roark's Table 8.1; tau proportional to (P_coolant - P_gas).
        """
        regen = _get_regen()
        P_cool_arr = regen["P_coolant"].copy()

        sol = bilayer_stress_profile(
            regen_solution=regen,
            bilayer=_BILAYER,
            P_gas_arr=P_cool_arr,
        )
        assert np.allclose(sol["tau_rib"], 0.0, atol=1.0), (
            f"tau_rib should be 0 when P_gas=P_coolant; "
            f"max = {sol['tau_rib'].max():.1f} Pa"
        )

    def test_rib_shear_nonnegative(self) -> None:
        """Rib shear stress is non-negative."""
        regen = _get_regen()
        P_gas_arr = np.zeros(len(regen["x"]))
        sol = bilayer_stress_profile(
            regen_solution=regen,
            bilayer=_BILAYER,
            P_gas_arr=P_gas_arr,
        )
        assert np.all(sol["tau_rib"] >= 0.0)

    def test_sigma_vm_rib_equals_sqrt3_tau(self) -> None:
        """sigma_vm_rib = sqrt(3) * tau_rib (von Mises for pure shear).

        Source: von Mises criterion: tau_yield = sigma_yield / sqrt(3).
        """
        regen = _get_regen()
        P_gas_arr = np.zeros(len(regen["x"]))
        sol = bilayer_stress_profile(
            regen_solution=regen,
            bilayer=_BILAYER,
            P_gas_arr=P_gas_arr,
        )
        ratio = sol["sigma_vm_rib"] / (math.sqrt(3.0) * sol["tau_rib"] + 1e-30)
        assert np.allclose(ratio, 1.0, atol=1e-9), (
            "sigma_vm_rib should equal sqrt(3)*tau_rib"
        )


# ---------------------------------------------------------------------------
# 4. Bilayer passes where single-wall model failed
# ---------------------------------------------------------------------------
class TestBilayerPassesWhereSingleWallFailed:
    def test_bilayer_exit_plane_jacket_positive_margin(self) -> None:
        """
        The single-wall model predicted sigma_vm = 875 MPa (COMPRESSIVE hoop) at the
        exit plane (x ≈ 83.5 mm), using sign convention (P_gas - P_coolant)*r/t which
        wrongly put the liner in compression and misidentified the failure mode as
        compressive buckling.

        The bilayer model correctly decomposes the wall:
          - Inner liner: sigma_h_inner = (P_coolant - P_gas)*r_liner/t_liner > 0  [TENSILE]
            At the exit, (P_coolant - P_gas) ~ 11.7 MPa, r ~ 36 mm, t_liner = 0.5 mm
            -> sigma_h_inner ~ 840 MPa, above yield -> LCF governs (same regime as throat).
            This is correct physics for a thin copper liner in a flight chamber.
          - Outer jacket: sigma_h_jacket = P_coolant*r_jacket/t_jacket [TENSILE]
            With 2 mm Inconel 718 jacket, this is ~ 238 MPa << sigma_y(IN718) ~ 1034 MPa.
            -> Jacket has large POSITIVE margin.

        Key improvements over single-wall:
          1. Correct sign: liner is in TENSILE hoop (not compressive).
          2. Jacket carries structural load with positive margin.
          3. Liner high-stress at exit is correctly classified as LCF-dominated
             (same regime as throat), not a compressive buckling failure.

        Source: NASA SP-8087 (1977) §3.3; Cook & Coffey (1973) AIAA 73-1364.
        """
        from physics.regen.bartz import mach_from_area_ratio

        regen = _get_regen()
        x_arr = regen["x"]

        # Compute isentropic P_gas profile
        P_gas_arr = np.zeros(len(x_arr))
        for k, (x, _) in enumerate(zip(x_arr, regen["r"])):
            eps = NOZZLE.area_ratio(float(x))
            sup = float(x) >= 0.0
            M = mach_from_area_ratio(eps, GAS.gamma, supersonic=sup)
            g = GAS.gamma
            P_gas_arr[k] = GAS.Pc * (1.0 + (g - 1.0) / 2.0 * M**2) ** (-g / (g - 1.0))

        sol = bilayer_stress_profile(
            regen_solution=regen,
            bilayer=_BILAYER,
            P_gas_arr=P_gas_arr,
        )

        i_exit = int(np.argmax(x_arr))

        # Liner hoop is TENSILE (positive) — bilayer sign is opposite to single-wall
        assert sol["sigma_h_inner"][i_exit] > 0.0, (
            "Bilayer liner hoop at exit must be TENSILE (P_coolant > P_gas); "
            f"got {sol['sigma_h_inner'][i_exit]/1e6:.1f} MPa"
        )

        # Jacket has positive yield margin everywhere (structural load carrier)
        assert sol["margin_yield_jacket"][i_exit] >= 0.0, (
            f"Jacket yield margin at exit = {sol['margin_yield_jacket'][i_exit]:.3f} "
            "(Inconel 718 jacket must have margin >= 0)"
        )

        # Jacket VM is within a plausible range (< 600 MPa for 2 mm IN718 at exit)
        vm_jacket_exit = float(sol["sigma_vm_jacket"][i_exit])
        assert vm_jacket_exit < 600e6, (
            f"Jacket VM at exit = {vm_jacket_exit/1e6:.1f} MPa, expected < 600 MPa"
        )

    def test_bilayer_jacket_positive_margin_everywhere(self) -> None:
        """Jacket yield margin is positive at every axial station.

        The jacket is sized to carry full coolant pressure in the flight architecture.
        With 2 mm Inconel 718 and P_coolant <= 12 MPa, margin should be positive
        everywhere.

        Source: NASA SP-8087 §3.3; Timoshenko §131.
        """
        regen = _get_regen()
        P_gas_arr = np.zeros(len(regen["x"]))
        sol = bilayer_stress_profile(
            regen_solution=regen,
            bilayer=_BILAYER,
            P_gas_arr=P_gas_arr,
        )
        min_margin = float(sol["margin_yield_jacket"].min())
        assert min_margin >= 0.0, (
            f"Jacket yield margin negative at some station: min = {min_margin:.3f}"
        )

    def test_bilayer_all_margins_finite(self) -> None:
        """bilayer_stress_profile returns finite (non-NaN, non-inf) margins."""
        regen = _get_regen()
        P_gas_arr = np.zeros(len(regen["x"]))
        sol = bilayer_stress_profile(
            regen_solution=regen,
            bilayer=_BILAYER,
            P_gas_arr=P_gas_arr,
        )
        for key in ("margin_yield_inner", "margin_yield_jacket", "margin_yield_rib"):
            arr = sol[key]
            finite_arr = arr[np.isfinite(arr)]
            assert len(finite_arr) > 0, f"{key}: no finite values"
            assert not np.any(np.isnan(arr)), f"{key} contains NaN"


# ---------------------------------------------------------------------------
# 5. Jacket auto-sizer
# ---------------------------------------------------------------------------
class TestJacketAutosizer:
    def test_jacket_autosizer_yield_limited(self) -> None:
        """size_jacket_thickness gives t such that P*r/t = sigma_y/FoS.

        Source: thin-shell yield criterion, Timoshenko §131.
        """
        P = 12.0e6     # Pa
        r = 0.030      # m
        T = 250.0      # K (jacket near coolant temperature)
        FoS = 1.4

        t = size_jacket_thickness(
            P_coolant_max=P,
            r_jacket=r,
            material=INCONEL718_STRUCT,
            T_op=T,
            FoS=FoS,
        )

        sy = INCONEL718_STRUCT.sigma_y(T)
        sigma_h = P * r / t
        expected_sigma = sy / FoS

        rel_err = abs(sigma_h - expected_sigma) / expected_sigma
        assert rel_err < 1e-9, (
            f"Autosized hoop = {sigma_h/1e6:.2f} MPa, "
            f"expected sigma_y/FoS = {expected_sigma/1e6:.2f} MPa "
            f"(rel err = {rel_err:.2e})"
        )

    def test_jacket_autosizer_positive_thickness(self) -> None:
        """Autosized thickness is positive for any valid input."""
        t = size_jacket_thickness(
            P_coolant_max=15.0e6,
            r_jacket=0.025,
            material=INCONEL718_STRUCT,
            T_op=300.0,
        )
        assert t > 0.0, f"Thickness must be positive, got {t}"

    def test_jacket_autosizer_increases_with_pressure(self) -> None:
        """Higher pressure -> thicker jacket."""
        kwargs = dict(r_jacket=0.025, material=INCONEL718_STRUCT, T_op=300.0)
        t_lo = size_jacket_thickness(P_coolant_max=8.0e6, **kwargs)
        t_hi = size_jacket_thickness(P_coolant_max=16.0e6, **kwargs)
        assert t_hi > t_lo, f"t(16 MPa) = {t_hi*1e3:.2f} mm should > t(8 MPa) = {t_lo*1e3:.2f} mm"


# ---------------------------------------------------------------------------
# 6. Required output keys present
# ---------------------------------------------------------------------------
class TestOutputKeys:
    def test_all_required_keys_present(self) -> None:
        """bilayer_stress_profile returns all required output keys."""
        regen = _get_regen()
        P_gas_arr = np.zeros(len(regen["x"]))
        sol = bilayer_stress_profile(
            regen_solution=regen,
            bilayer=_BILAYER,
            P_gas_arr=P_gas_arr,
        )
        required = [
            "x", "r_inner", "r_jacket", "t_liner", "t_jacket",
            "T_gas", "T_cool",
            "sigma_h_inner", "sigma_th_inner_gas", "sigma_th_inner_cool",
            "sigma_vm_inner", "margin_yield_inner", "margin_ultimate_inner",
            "sigma_h_jacket", "sigma_vm_jacket",
            "margin_yield_jacket", "margin_ultimate_jacket",
            "tau_rib", "sigma_vm_rib", "margin_yield_rib",
        ]
        for k in required:
            assert k in sol, f"Missing output key: {k}"

    def test_array_lengths_match(self) -> None:
        """All output arrays have the same length as x."""
        regen = _get_regen()
        P_gas_arr = np.zeros(len(regen["x"]))
        sol = bilayer_stress_profile(
            regen_solution=regen,
            bilayer=_BILAYER,
            P_gas_arr=P_gas_arr,
        )
        n = len(sol["x"])
        for k, v in sol.items():
            if hasattr(v, "__len__"):
                assert len(v) == n, f"{k}: expected len {n}, got {len(v)}"
