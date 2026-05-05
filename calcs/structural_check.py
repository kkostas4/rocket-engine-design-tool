"""
structural_check.py
===================

First-pass structural analysis of the sized regenerative-cooling design.

Uses the channel geometry from sizing.py (76 ch, w=0.6 mm, h_throat=0.30 mm,
h_divergent=3.2 mm, GRCop-42) and runs:

  1. liner_stress_profile() -- hoop + through-wall thermal + von Mises
  2. lcf_check()            -- Manson-Coffin LCF life at worst station
  3. print_structural_report() -- formatted margin report

Stress components
-----------------
  sigma_hoop      -- (P_gas - P_coolant) * r / t_wall  [negative = compressive]
                     The liner is in compressive hoop since P_coolant > P_gas.
                     The outer jacket carries the tensile hoop load from P_coolant.
  sigma_th_gas    -- thermal stress at hot surface (compressive, negative)
  sigma_th_cool   -- thermal stress at cool surface (tensile, positive)
  sigma_vm_gas    -- von Mises at gas-side surface (governs failure)
  sigma_vm_cool   -- von Mises at coolant-side surface

P_gas profile is computed from isentropic relations P_gas = Pc * (1 + (g-1)/2 M^2)^(-g/(g-1))
using the local area ratio from the nozzle.

Material: GRCop-84 (proxy for GRCop-42). FLAG: GRCop-84 may be 10-15 %
stronger than GRCop-42 -- properties are non-conservative on the strength side.

Expected result: negative yield margin near throat (thermal stress >> sigma_y).
This is normal for regen liners -- the liner operates in the plastic range.
LCF (Manson-Coffin) governs failure, not static yield.

Run with:
    python calcs/structural_check.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from physics.nozzle.rao import RaoNozzle
from physics.regen.bartz import CombustionGas, mach_from_area_ratio
from physics.regen.coolant import GRCOP42, CoolantInlet
from physics.regen.sizing import ChannelSizingProblem, size_channels
from physics.regen.solver import solve_regen
from physics.structural.fatigue import lcf_check
from physics.structural.liner_stress import liner_stress_profile
from physics.structural.margin import check_all_margins, print_structural_report
from physics.structural.materials import GRCOP84_PROXY

# ---------------------------------------------------------------------------
# Reference engine
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
COOLANT = CoolantInlet(T=150.0, P=12.0e6, m_dot=0.3096, fluid="Methane")

# ---------------------------------------------------------------------------
# Sized channel geometry (reproduce from sizing.py)
# ---------------------------------------------------------------------------
print("Running channel sizer (~60-90 s)...")
_problem = ChannelSizingProblem(
    nozzle=NOZZLE,
    gas=GAS,
    coolant_inlet=COOLANT,
    wall_material=GRCOP42,
    T_wall_max=1000.0,
    T_wall_margin=50.0,
    dP_max=3.0e6,
    min_wall_thickness=0.5e-3,
    max_aspect_ratio=8.0,
    min_channel_width=0.6e-3,
    t_land_min=0.3e-3,
    n_stations_opt=60,
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    geom = size_channels(_problem)

# ---------------------------------------------------------------------------
# Regen solve (n=200 for good resolution)
# ---------------------------------------------------------------------------
print("Running regen solver (n=200)...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    regen = solve_regen(
        nozzle=NOZZLE,
        gas=GAS,
        channel_geom=geom,
        coolant_inlet=COOLANT,
        wall_material=GRCOP42,
        direction="counterflow",
        n_stations=200,
        tol=0.5,
    )

# ---------------------------------------------------------------------------
# Gas pressure profile (isentropic, station-by-station)
# ---------------------------------------------------------------------------
# P_gas(x) = Pc * (1 + (g-1)/2 * M(x)^2)^(-g/(g-1))
# Source: Anderson (2003) "Modern Compressible Flow" Eq. (3.30).
_g = GAS.gamma
_Pc = GAS.Pc
_x_regen = regen["x"]
_r_regen = regen["r"]
P_gas_arr = np.zeros(len(_x_regen))
for _i, (_x, _r) in enumerate(zip(_x_regen, _r_regen)):
    _eps = NOZZLE.area_ratio(_x)
    _sup = _x >= 0.0  # supersonic downstream of throat
    _M = mach_from_area_ratio(_eps, _g, supersonic=_sup)
    P_gas_arr[_i] = _Pc * (1.0 + (_g - 1.0) / 2.0 * _M**2) ** (-_g / (_g - 1.0))

print(
    f"  Gas pressure range: {P_gas_arr.min()/1e6:.3f} MPa (exit) "
    f"to {P_gas_arr.max()/1e6:.3f} MPa (chamber)"
)

# ---------------------------------------------------------------------------
# Structural analysis
# ---------------------------------------------------------------------------
print("Computing liner stress profile...")
structural = liner_stress_profile(
    regen_solution=regen,
    geometry=geom,
    material=GRCOP84_PROXY,
    P_gas_arr=P_gas_arr,
)

lcf = lcf_check(
    regen_solution=regen,
    structural_solution=structural,
    n_cycles_required=100,
    material=GRCOP84_PROXY,
)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print_structural_report(
    structural_solution=structural,
    lcf_result=lcf,
    material_name=GRCOP84_PROXY.name,
)

summary = check_all_margins(structural)
if summary["any_negative_margin"]:
    print()
    print("*** NEGATIVE MARGIN DETECTED ***")
    my = structural["margin_yield"]
    mu = structural["margin_ultimate"]
    x_arr = structural["x"] * 1e3
    neg_y = np.where(my < 0.0)[0]
    neg_u = np.where(mu < 0.0)[0]
    if len(neg_y):
        i_w = int(np.argmin(my))
        print(f"  Yield margin < 0 at {len(neg_y)} stations "
              f"(x = {x_arr[neg_y[0]]:.1f} to {x_arr[neg_y[-1]]:.1f} mm)")
        print(f"  Worst: M_yield = {my[i_w]:.3f} at x = {x_arr[i_w]:.1f} mm")
        sigma_hoop_w = structural["sigma_hoop"][i_w]
        sigma_th_w = structural["sigma_thermal_gas"][i_w]
        sigma_vm_w = structural["sigma_vm_gas"][i_w]
        T_gas_w = float(structural["T_gas"][i_w])
        hoop_frac = abs(sigma_hoop_w) / (abs(sigma_hoop_w) + abs(sigma_th_w) + 1.0)
        print(f"  Dominant component: sigma_vm_gas = {sigma_vm_w/1e6:.1f} MPa, "
              f"sigma_y(T_gas) = {GRCOP84_PROXY.sigma_y(T_gas_w)/1e6:.1f} MPa")
        print(f"    sigma_hoop = {sigma_hoop_w/1e6:.1f} MPa  "
              f"sigma_th_gas = {sigma_th_w/1e6:.1f} MPa  "
              f"T_wall_gas = {T_gas_w:.0f} K")
        if hoop_frac > 0.7:
            print("  Driver: COMPRESSIVE HOOP (P_gas - P_coolant)*r/t dominates at large exit radius")
            print("  What would need to change (hoop-dominated exit):")
            print("    - Staged coolant circuit: reduce P_coolant in divergent section")
            print("      (e.g. split manifold -- divergent sees inlet P, convergent sees outlet P)")
            print("    - Outer structural jacket takes hoop load; liner is a thermal insert only")
            print("    - Increase t_wall at exit to reduce sigma_hoop (increases R_wall but T_wall")
            print("      is low there -- thermally acceptable)")
            print(f"    - Accept: liner plastically strained; LCF (N_f={lcf['N_f']:.0f}) governs")
        else:
            print("  Driver: THERMAL STRESS (E*alpha*dT/(1-nu)) dominates near throat")
            print("  What would need to change (thermal-dominated throat):")
            print("    - Reduce t_wall (lower R_wall -> lower sigma_th) but t_min=0.5 mm is AM limit")
            print("    - Increase n_channels (more channels -> lower h per channel -> lower T_wall_gas)")
            print("    - Apply Pizzarelli correction (actual h_c 2-5x higher -> lower T_wall_gas)")
            print(f"    - Accept: liner plastically strained; LCF (N_f={lcf['N_f']:.0f}) governs")
    if len(neg_u):
        print(f"  Ultimate margin < 0 at {len(neg_u)} stations")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
os.makedirs("physics/structural/plots", exist_ok=True)
out_path = "physics/structural/plots/structural_check.png"

x_mm = structural["x"] * 1e3

fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

# --- Panel 1: Temperatures ---
ax = axes[0]
ax.plot(x_mm, structural["T_gas"], color="firebrick", lw=1.8, label="T_wall_gas")
ax.plot(x_mm, structural["T_cool"], color="steelblue", lw=1.5, ls="--",
        label="T_wall_cool")
ax.axhline(GRCOP84_PROXY.sigma_y(800.0) * 0.0 + 810.0, color="black",
           lw=0.8, ls=":", label="GRCop-42 service limit (810 K)")
ax.axvline(0.0, color="red", lw=0.7, ls="--", label="throat")
ax.set_ylabel("Temperature [K]")
ax.set_title(
    "Structural Check -- Sized Regen Design (76 ch, GRCop-42)\n"
    "GRCop-84 proxy properties (Ellis 2005 NASA/TM-2005-213566)"
)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 2: Stresses ---
ax = axes[1]
ax.plot(x_mm, structural["sigma_hoop"] / 1e6, color="steelblue", lw=1.5,
        label="sigma_hoop net = (P_gas-P_cool)*r/t")
ax.plot(x_mm, structural["sigma_thermal_gas"] / 1e6, color="firebrick", lw=1.5,
        ls="--", label="sigma_th_gas (compressive)")
ax.plot(x_mm, structural["sigma_thermal_cool"] / 1e6, color="darkorange", lw=1.5,
        ls=":", label="sigma_th_cool (tensile)")
ax.plot(x_mm, structural["sigma_vm_gas"] / 1e6, color="purple", lw=2.0,
        label="sigma_vm (gas side)")
ax.plot(x_mm, structural["sigma_vm_cool"] / 1e6, color="green", lw=1.5,
        ls="-.", label="sigma_vm (cool side)")
# Plot sigma_y at each station temperature
sy_arr = np.array([GRCOP84_PROXY.sigma_y(float(T)) for T in structural["T_gas"]])
ax.plot(x_mm, sy_arr / 1e6, color="black", lw=1.0, ls=":", label="sigma_y(T_gas)")
ax.axvline(0.0, color="red", lw=0.7, ls="--")
ax.axhline(0.0, color="gray", lw=0.5)
ax.set_ylabel("Stress [MPa]")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# --- Panel 3: Margins ---
ax = axes[2]
ax.plot(x_mm, structural["margin_yield"], color="steelblue", lw=1.8,
        label=f"M_yield (FoS={1.4})")
ax.plot(x_mm, structural["margin_ultimate"], color="darkorange", lw=1.5,
        ls="--", label=f"M_ultimate (FoS={2.0})")
ax.axhline(0.0, color="red", lw=1.2, ls="-", label="M = 0 (limit)")
ax.axvline(0.0, color="red", lw=0.7, ls="--")
ax.set_xlabel("Axial position x [mm]")
ax.set_ylabel("Safety margin M = (sigma_allow / (FoS * sigma_vm)) - 1")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_title("Safety margins (M >= 0 passes; M < 0 means LCF governs, not static yield)")

# Mark LCF worst station
x_lcf_mm = lcf["x_worst"] * 1e3
for axi in axes:
    axi.axvline(x_lcf_mm, color="magenta", lw=0.8, ls=":", alpha=0.7)

# Add annotation on panel 3
ax.annotate(
    f"LCF worst\nN_f={lcf['N_f']:.0f} cyc",
    xy=(x_lcf_mm, float(structural["margin_yield"][lcf["i_worst"]])),
    xytext=(x_lcf_mm + 5.0, 0.5),
    fontsize=7,
    color="magenta",
    arrowprops=dict(arrowstyle="->", color="magenta", lw=0.8),
)

fig.tight_layout()
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"\nPlot saved to {out_path}")
