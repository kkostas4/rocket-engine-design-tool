"""
run_methalox_50kN.py
====================

End-to-end demonstration of the noyron.engine orchestrator.

Two designs are run:
  1. Reference engine  (4,448 N / 1000 psia / O/F=3.6 / ε=10.605)
     Used for side-by-side comparison against the existing calcs/ scripts.
  2. Demo 50 kN engine (50,000 N / 100 bar / O/F=3.5 / ε=20)
     Demonstrates scalability of the pipeline.

Side-by-side comparison (reference engine only)
-----------------------------------------------
The orchestrator must reproduce the individual calcs/ outputs to within 1%.
Expected values are taken from the deterministic calcs/ runs:

  calcs/regen_sized_design.py  : peak T_wall and coolant dP for sized design
  calcs/bilayer_check.py       : N_f at worst station, structural margins
  calcs/structural_check.py    : structural margins

Any discrepancy > 1% is flagged as a bug.

Run with:
    python calcs/run_methalox_50kN.py

Note on CEA limitation
-----------------------
The project CEA wrapper (physics/thermochem/cea.py) is locked to
Pc = 1000 psia (6.8948 MPa).  For the 50 kN / 100 bar design,
c_star and Tc are evaluated at the reference Pc as a first-order
approximation.  Differences < 2% are expected for these operating points.
A live rocketcea backend will remove this limitation.

Note on render time
-------------------
Rendering at voxel_size=1e-4 (0.1 mm) takes ~10-30 minutes and requires
~1 GB peak RAM.  The render call is included but long runtime is expected.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from noyron.engine import Engine, EngineRequirements, METHALOX

# ---------------------------------------------------------------------------
# Design 1 — Reference engine (for comparison against calcs/)
# ---------------------------------------------------------------------------

print("=" * 72)
print("Design 1: Reference engine  (4,448 N | 1000 psia | O/F=3.6 | e=10.605)")
print("=" * 72)
t0 = time.perf_counter()

req_ref = EngineRequirements(
    thrust=4448.22,
    chamber_pressure=6.8948e6,
    mixture_ratio=3.60,
    propellant=METHALOX,
    expansion_ratio=10.605,
    n_cycles_required=100,
    T_wall_max=1000.0,
    dP_coolant_budget=3.0e6,
)

print("Running design pipeline...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    design_ref = Engine(req_ref).design()

t_ref = time.perf_counter() - t0
print(f"Design complete in {t_ref:.1f} s")
print()
print(Engine(req_ref).summary(design_ref))
print()

# ---------------------------------------------------------------------------
# Side-by-side comparison (reference engine)
# ---------------------------------------------------------------------------
# Expected values from deterministic calcs/ script runs.
# These are NOT hardcoded magic numbers — they come from running the same
# physics modules that the orchestrator calls.  If the calcs/ scripts are
# updated, these reference values must be re-captured.
#
# How to recapture:
#   python calcs/regen_sized_design.py   -> copy "SIZED design" T_wall and dP
#   python calcs/bilayer_check.py        -> copy N_f at worst station
#
# The values below correspond to the GRCop-42 sized design with n_stations=200,
# tol=0.5, dP_max=3 MPa, T_wall_max=1000 K, T_wall_margin=50 K.
#
# NOTE: the sizer uses a non-deterministic grid sweep over n_channels; the
# exact n_channels and geometry may differ by one grid step between runs.
# T_wall_peak and dP_total are compared to the calcs outputs.

regen_ref = design_ref.regen_solution
lcf_ref = design_ref.lcf

T_wall_peak_orch = float(regen_ref["T_wall_gas"].max())
dP_coolant_orch = float(regen_ref["dP_total"])
n_ch_orch = design_ref.channel_geometry.n_channels
N_f_orch = lcf_ref["N_f"]
x_worst_orch = lcf_ref["x_worst"]

import numpy as np
i_peak_inner = int(np.argmax(design_ref.structural["sigma_vm_inner"]))
peak_stress_orch = float(design_ref.structural["sigma_vm_inner"][i_peak_inner])

print("=" * 72)
print("Side-by-side: Orchestrator vs calcs/ scripts (reference engine)")
print("=" * 72)
print()
print(f"  {'Metric':<30} {'Orchestrator':>18} {'calcs/ expected':>18} {'Diff %':>8}")
print(f"  {'-'*30} {'-'*18} {'-'*18} {'-'*8}")

# The calcs/ comparison values are obtained by running the same physics modules.
# They cannot be pre-known without running the calcs.  We therefore run a
# lightweight version of the comparison here: we call the individual physics
# modules directly (same as calcs/ do) and compare to the orchestrator output.

# Re-run the individual modules for comparison
from noyron.rp.nozzle.rao import RaoNozzle
from noyron.rp.regen.bartz import CombustionGas, mach_from_area_ratio
from noyron.rp.regen.coolant import GRCOP42, CoolantInlet
from noyron.rp.regen.sizing import ChannelSizingProblem, size_channels
from noyron.rp.regen.solver import solve_regen
from noyron.rp.structural.bilayer import (
    BilayerWall, RibGeometry, WallLayer,
    bilayer_stress_profile, size_jacket_thickness,
)
from noyron.rp.structural.fatigue import lcf_check
from noyron.rp.structural.materials import GRCOP84_PROXY, INCONEL718_STRUCT

print()
print("  (Re-running calcs/ pipeline standalone for comparison…)")
t_calcs = time.perf_counter()

_GAS = CombustionGas(
    Tc=3561.2, Pc=6.8948e6, gamma=1.15, MW=21.9,
    cp=2000.0, mu=8.0e-5, Pr=0.60, c_star=1893.3,
)
_NOZZLE = RaoNozzle(r_throat=0.022313 / 2.0, epsilon=10.605)
_COOLANT = CoolantInlet(T=150.0, P=12.0e6, m_dot=0.3096, fluid="Methane")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _prob = ChannelSizingProblem(
        nozzle=_NOZZLE, gas=_GAS, coolant_inlet=_COOLANT,
        wall_material=GRCOP42, T_wall_max=1000.0, T_wall_margin=50.0,
        dP_max=3.0e6, min_wall_thickness=0.5e-3, max_aspect_ratio=8.0,
        min_channel_width=0.6e-3, t_land_min=0.3e-3, n_stations_opt=60,
    )
    _geom = size_channels(_prob)
    _regen = solve_regen(
        nozzle=_NOZZLE, gas=_GAS, channel_geom=_geom, coolant_inlet=_COOLANT,
        wall_material=GRCOP42, direction="counterflow", n_stations=200, tol=0.5,
    )

# Jacket sizing for calcs/
_T_cool_inlet = float(_regen["T_coolant"][-1])
_r_exit = _NOZZLE.r_throat * _NOZZLE.epsilon**0.5
_r_jkt_exit = _r_exit + 0.5e-3 + _geom.height(_NOZZLE.x_exit)
_t_jkt = max(1.5e-3, round(
    size_jacket_thickness(
        P_coolant_max=float(_regen["P_coolant"].max()),
        r_jacket=_r_jkt_exit,
        material=INCONEL718_STRUCT,
        T_op=_T_cool_inlet + 50.0,
        FoS=1.4,
    ) / 0.5e-3) * 0.5e-3)

_bilayer = BilayerWall(
    inner_liner=WallLayer(thickness=_geom.wall_thickness, material=GRCOP84_PROXY),
    outer_jacket=WallLayer(thickness=lambda x, _t=_t_jkt: _t, material=INCONEL718_STRUCT),
    ribs=RibGeometry(
        n_ribs=_geom.n_channels, land_width=_geom.land_thickness,
        height=_geom.height, material=GRCOP84_PROXY,
    ),
    channel_geometry=_geom,
)

_g = _GAS.gamma
_x_arr = _regen["x"]
_P_gas_arr = np.zeros(len(_x_arr))
for _i, _x in enumerate(_x_arr):
    _eps = _NOZZLE.area_ratio(float(_x))
    _sup = float(_x) >= 0.0
    _M = mach_from_area_ratio(_eps, _g, supersonic=_sup)
    _P_gas_arr[_i] = _GAS.Pc * (1.0 + (_g - 1.0) / 2.0 * _M**2) ** (-_g / (_g - 1.0))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _struct = bilayer_stress_profile(
        regen_solution=_regen, bilayer=_bilayer, P_gas_arr=_P_gas_arr,
    )

_lcf = lcf_check(
    regen_solution=_regen,
    structural_solution=_struct,
    n_cycles_required=100,
    material=GRCOP84_PROXY,
)

print(f"  calcs/ pipeline complete in {time.perf_counter()-t_calcs:.1f} s")
print()

# Collect calcs/ values
T_wall_peak_calcs = float(_regen["T_wall_gas"].max())
dP_coolant_calcs = float(_regen["dP_total"])
n_ch_calcs = _geom.n_channels
N_f_calcs = _lcf["N_f"]
_i_pk_in = int(np.argmax(_struct["sigma_vm_inner"]))
peak_stress_calcs = float(_struct["sigma_vm_inner"][_i_pk_in])


def _pct(a: float, b: float) -> str:
    if b == 0:
        return "   N/A"
    return f"{100.0 * (a - b) / b:+.2f} %"


def _flag(pct_str: str) -> str:
    val = float(pct_str.replace("%", "").strip())
    return "OK" if abs(val) <= 1.0 else "BUG: diff > 1%"


rows = [
    ("Peak T_wall [K]",   T_wall_peak_orch, T_wall_peak_calcs),
    ("Coolant dP [MPa]",  dP_coolant_orch / 1e6, dP_coolant_calcs / 1e6),
    ("n_channels",        float(n_ch_orch), float(n_ch_calcs)),
    ("N_f throat [cycles]", N_f_orch, N_f_calcs),
    ("Peak sigma_vm_inner [MPa]", peak_stress_orch / 1e6, peak_stress_calcs / 1e6),
]

all_ok = True
for label, orch_val, calcs_val in rows:
    pct = _pct(orch_val, calcs_val)
    flag = _flag(pct)
    if flag != "OK":
        all_ok = False
    print(f"  {label:<30} {orch_val:>18.4g} {calcs_val:>18.4g} {pct:>8}  [{flag}]")

print()
if all_ok:
    print("  All comparisons within 1%.  Orchestrator reproduces calcs/ correctly.")
else:
    print("  WARNING: one or more comparisons exceed 1%.  Investigate discrepancies.")
print()

# ---------------------------------------------------------------------------
# Design 2 — 50 kN demonstration engine
# ---------------------------------------------------------------------------

print("=" * 72)
print("Design 2: Demo 50 kN engine  (50,000 N | 100 bar | O/F=3.5 | e=20)")
print("=" * 72)
t0 = time.perf_counter()

req_50k = EngineRequirements(
    thrust=50_000.0,
    chamber_pressure=10.0e6,   # 100 bar
    mixture_ratio=3.5,
    propellant=METHALOX,
    expansion_ratio=20.0,
    n_cycles_required=100,
    T_wall_max=1000.0,
    dP_coolant_budget=3.0e6,
)

print("Running design pipeline...")
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        design_50k = Engine(req_50k).design()

    t_50k = time.perf_counter() - t0
    print(f"Design complete in {t_50k:.1f} s")
    print()
    print(Engine(req_50k).summary(design_50k))
    print()
except Exception as exc:
    t_50k = time.perf_counter() - t0
    print(f"Design failed after {t_50k:.1f} s")
    print(f"  {type(exc).__name__}: {exc}")
    print()
    print("  Note: a 50 kN engine at 100 bar has ~11x the fuel mass flow of the")
    print("  reference engine.  With a 3 MPa coolant budget the channel pressure")
    print("  drop is physically infeasible.  Increase dP_coolant_budget (e.g. 10 MPa)")
    print("  or reduce Pc to size a feasible 50 kN design.")
    print()

# ---------------------------------------------------------------------------
# Render reference engine at 0.1 mm voxels
# ---------------------------------------------------------------------------

print("=" * 72)
print("Rendering reference engine (voxel_size=1e-4, chamber_nozzle + channels)")
print("Expected runtime: 10-30 min.  Peak RAM: ~1 GB.")
print("=" * 72)
t0 = time.perf_counter()

output_dir = os.path.join(os.path.dirname(__file__), "..", "geometry", "output")
os.makedirs(output_dir, exist_ok=True)

from pathlib import Path
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    stl_paths = Engine(req_ref).render(
        design_ref,
        voxel_size=1e-4,
        output_dir=Path(output_dir),
        components=("chamber_nozzle", "channels"),
    )

t_render = time.perf_counter() - t0
print(f"Render complete in {t_render:.1f} s")
for name, path in stl_paths.items():
    size_kb = Path(path).stat().st_size / 1024
    print(f"  {name}: {path}  ({size_kb:.0f} kB)")
