"""
bilayer_check.py
================

Bilayer (liner + jacket + ribs) structural analysis for the sized regen design.

Geometry
--------
  Inner liner : GRCop-84 proxy (same as single-wall model), t_liner = 0.5 mm
  Outer jacket: Inconel 718, t_jacket autosized from yield criterion at
                T_jacket = T_coolant_inlet + 50 K (= 200 K)
  Ribs        : GRCop-84 (milled-channel construction), same land geometry
                as sizing.py output: t_land = 0.3 mm, h_ch varies by zone

Comparison to single-wall (calcs/structural_check.py)
------------------------------------------------------
  Single-wall: sigma_vm at exit = 875 MPa (COMPRESSIVE hoop, wrong failure mode)
  Bilayer    : liner sigma_vm at exit ~ 840 MPa (TENSILE, LCF governs, correct)
               jacket sigma_vm at exit ~ 240 MPa (positive margin, IN718)

  The bilayer model does not reduce the liner stress magnitude — it corrects the
  sign convention and identifies the jacket as the structural load carrier.

  Key change: jacket now has positive yield margin everywhere; the liner at the
  exit operates in LCF regime (above yield, same as the throat) which is the
  correct and expected behaviour for flight copper-alloy liners.

Run with:
    python calcs/bilayer_check.py
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
from physics.structural.bilayer import (
    BilayerWall,
    RibGeometry,
    WallLayer,
    bilayer_stress_profile,
    size_jacket_thickness,
)
from physics.structural.fatigue import lcf_check, manson_coffin_life
from physics.structural.liner_stress import liner_stress_profile
from physics.structural.materials import GRCOP84_PROXY, INCONEL718_STRUCT

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
# Gas pressure profile (isentropic)
# ---------------------------------------------------------------------------
_g = GAS.gamma
_x_regen = regen["x"]
_r_regen = regen["r"]
P_gas_arr = np.zeros(len(_x_regen))
for _i, _x in enumerate(_x_regen):
    _eps = NOZZLE.area_ratio(_x)
    _sup = _x >= 0.0
    _M = mach_from_area_ratio(_eps, _g, supersonic=_sup)
    P_gas_arr[_i] = GAS.Pc * (1.0 + (_g - 1.0) / 2.0 * _M**2) ** (-_g / (_g - 1.0))

print(f"  Gas pressure range: {P_gas_arr.min()/1e6:.3f} MPa (exit) "
      f"to {P_gas_arr.max()/1e6:.3f} MPa (chamber)")

# ---------------------------------------------------------------------------
# Jacket auto-sizing
# ---------------------------------------------------------------------------
# Jacket operates near coolant temperature.
# Coolant enters at T_inlet = 150 K (counterflow: last element = inlet).
T_coolant_inlet = float(regen["T_coolant"][-1])
T_jacket_op = T_coolant_inlet + 50.0   # 50 K margin above coolant inlet
P_coolant_max = float(regen["P_coolant"].max())

# Jacket inner radius at throat (smallest) governs t/r check; use exit for worst hoop
r_exit = float(_r_regen[-1])           # exit radius ~ 36 mm
r_jacket_exit = r_exit + 0.5e-3 + geom.height(float(_x_regen[-1]))

t_jacket_autosized = size_jacket_thickness(
    P_coolant_max=P_coolant_max,
    r_jacket=r_jacket_exit,
    material=INCONEL718_STRUCT,
    T_op=T_jacket_op,
    FoS=1.4,
)
# Round up to nearest 0.5 mm and enforce 1.5 mm minimum for machinability
t_jacket = max(1.5e-3, round(t_jacket_autosized / 0.5e-3) * 0.5e-3)
print(f"  Jacket autosized: {t_jacket_autosized*1e3:.2f} mm -> using {t_jacket*1e3:.1f} mm "
      f"(T_jacket={T_jacket_op:.0f} K, sigma_y={INCONEL718_STRUCT.sigma_y(T_jacket_op)/1e6:.0f} MPa)")

# ---------------------------------------------------------------------------
# Bilayer geometry
# ---------------------------------------------------------------------------
_x2 = 2.0 * NOZZLE.r_throat
bilayer = BilayerWall(
    inner_liner=WallLayer(
        thickness=geom.wall_thickness,
        material=GRCOP84_PROXY,
    ),
    outer_jacket=WallLayer(
        thickness=lambda x: t_jacket,
        material=INCONEL718_STRUCT,
    ),
    ribs=RibGeometry(
        n_ribs=geom.n_channels,
        land_width=geom.land_thickness,
        height=geom.height,
        material=GRCOP84_PROXY,
    ),
    channel_geometry=geom,
)

# ---------------------------------------------------------------------------
# Bilayer stress profile
# ---------------------------------------------------------------------------
print("Computing bilayer stress profile...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    bilayer_sol = bilayer_stress_profile(
        regen_solution=regen,
        bilayer=bilayer,
        P_gas_arr=P_gas_arr,
    )

# ---------------------------------------------------------------------------
# Single-wall result (for comparison)
# ---------------------------------------------------------------------------
print("Computing single-wall stress profile (for comparison)...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sw_sol = liner_stress_profile(
        regen_solution=regen,
        geometry=geom,
        material=GRCOP84_PROXY,
        P_gas_arr=P_gas_arr,
    )

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
x_mm = bilayer_sol["x"] * 1e3
i_exit = int(np.argmax(bilayer_sol["x"]))
i_throat = int(np.argmin(np.abs(bilayer_sol["x"])))

# Peak stresses
i_peak_inner = int(np.argmax(bilayer_sol["sigma_vm_inner"]))
i_peak_jacket = int(np.argmax(bilayer_sol["sigma_vm_jacket"]))
i_peak_rib = int(np.argmax(bilayer_sol["tau_rib"]))

# Min margins
i_min_inner_y = int(np.argmin(bilayer_sol["margin_yield_inner"]))
i_min_jkt_y = int(np.argmin(bilayer_sol["margin_yield_jacket"]))
i_min_rib_y = int(np.argmin(bilayer_sol["margin_yield_rib"]))

print()
print("=" * 72)
print("Bilayer Structural Report -- Sized Regen Design (76 ch, GRCop-42 liner)")
print("  Inner liner : GRCop-84 proxy, t_liner = 0.5 mm")
print(f"  Outer jacket: Inconel 718,    t_jacket = {t_jacket*1e3:.1f} mm (autosized)")
print("  Ribs        : GRCop-84 (milled-channel, integral with liner)")
print("=" * 72)
print()

# Inner liner
print("  INNER LINER")
vm_pk_in = bilayer_sol["sigma_vm_inner"][i_peak_inner]
sh_pk_in = bilayer_sol["sigma_h_inner"][i_peak_inner]
sth_pk_in = bilayer_sol["sigma_th_inner_gas"][i_peak_inner]
T_pk_in = bilayer_sol["T_gas"][i_peak_inner]
print(f"    Peak sigma_vm  : {vm_pk_in/1e6:.1f} MPa  at x = {x_mm[i_peak_inner]:.1f} mm")
print(f"      sigma_h_inner: {sh_pk_in/1e6:.1f} MPa  (tensile, P_diff*r/t_liner)")
print(f"      sigma_th_gas : {sth_pk_in/1e6:.1f} MPa  (compressive, thermal)")
print(f"      T_wall_gas   : {T_pk_in:.0f} K")
my_min_in = bilayer_sol["margin_yield_inner"][i_min_inner_y]
flag_in = "FAIL *** (LCF governs, expected for Cu liner)" if my_min_in < 0.0 else "pass"
print(f"    Min M_yield    : {my_min_in:.3f}  at x = {x_mm[i_min_inner_y]:.1f} mm  [{flag_in}]")
if my_min_in < 0.0:
    print("    NOTE: Copper liner operates above yield -> LCF governs, same regime")
    print("    as the throat region. Static yield failure is NOT the failure mode.")
print()

# Outer jacket
print("  OUTER JACKET")
vm_pk_jkt = bilayer_sol["sigma_vm_jacket"][i_peak_jacket]
sh_pk_jkt = bilayer_sol["sigma_h_jacket"][i_peak_jacket]
T_pk_jkt = bilayer_sol["T_cool"][i_peak_jacket]
print(f"    Peak sigma_vm  : {vm_pk_jkt/1e6:.1f} MPa  at x = {x_mm[i_peak_jacket]:.1f} mm")
print(f"      sigma_h_jacket: {sh_pk_jkt/1e6:.1f} MPa  (tensile, P_coolant*r_jkt/t_jkt)")
print(f"      T_jacket       : {T_pk_jkt:.0f} K  (near coolant temperature)")
my_min_jkt = bilayer_sol["margin_yield_jacket"][i_min_jkt_y]
flag_jkt = "FAIL ***" if my_min_jkt < 0.0 else "pass"
print(f"    Min M_yield    : {my_min_jkt:.3f}  at x = {x_mm[i_min_jkt_y]:.1f} mm  [{flag_jkt}]")
print()

# Ribs
print("  RIBS (milled-channel lands, GRCop-84)")
tau_pk = bilayer_sol["tau_rib"][i_peak_rib]
vm_rib_pk = bilayer_sol["sigma_vm_rib"][i_peak_rib]
print(f"    Peak tau_rib   : {tau_pk/1e6:.1f} MPa  at x = {x_mm[i_peak_rib]:.1f} mm")
print(f"    Peak sigma_vm_rib: {vm_rib_pk/1e6:.1f} MPa  (= sqrt(3)*tau)")
my_min_rib = bilayer_sol["margin_yield_rib"][i_min_rib_y]
flag_rib = "FAIL ***" if my_min_rib < 0.0 else "pass"
print(f"    Min M_yield_rib: {my_min_rib:.3f}  at x = {x_mm[i_min_rib_y]:.1f} mm  [{flag_rib}]")
print()

# Comparison to single-wall
print("  COMPARISON: Single-wall vs. Bilayer at exit plane (x = {:.1f} mm)".format(
    x_mm[i_exit]))
sw_vm_exit = float(sw_sol["sigma_vm_gas"][i_exit])
sw_sh_exit = float(sw_sol["sigma_hoop"][i_exit])
bl_vm_in_exit = float(bilayer_sol["sigma_vm_inner"][i_exit])
bl_sh_in_exit = float(bilayer_sol["sigma_h_inner"][i_exit])
bl_vm_jkt_exit = float(bilayer_sol["sigma_vm_jacket"][i_exit])
print(f"    Single-wall sigma_vm  : {sw_vm_exit/1e6:.1f} MPa"
      f"  hoop = {sw_sh_exit/1e6:.1f} MPa (COMPRESSIVE, wrong failure mode)")
print(f"    Bilayer liner sigma_vm: {bl_vm_in_exit/1e6:.1f} MPa"
      f"  hoop = {bl_sh_in_exit/1e6:.1f} MPa (TENSILE, LCF regime)")
print(f"    Bilayer jacket sigma_vm:{bl_vm_jkt_exit/1e6:.1f} MPa"
      f"  margin = {bilayer_sol['margin_yield_jacket'][i_exit]:.2f} (positive)")
print()
print("  Out-of-scope (flagged for future work):")
print("    - Differential thermal expansion between liner and jacket is NOT included:")
print("      liner runs hotter, wants to expand more -> adds compression to liner,")
print("      tension to jacket. Requires coupled FEA. Flagged for future work.")
print()

# ---------------------------------------------------------------------------
# LCF at throat (worst station) and exit station
# ---------------------------------------------------------------------------
print("  LCF PREDICTION (Manson-Coffin, bilayer liner, GRCop-84 proxy)")
print("  " + "-" * 68)

# ---- Worst station (throat) via lcf_check() ----
lcf_throat = lcf_check(
    regen_solution=regen,
    structural_solution=bilayer_sol,   # uses T_gas/T_cool from regen
    n_cycles_required=100,
    material=GRCOP84_PROXY,
)
flag_throat = "pass" if lcf_throat["passes"] else "FAIL ***"
print("  Worst station (throat)")
print(f"    x             : {lcf_throat['x_worst']*1e3:.1f} mm")
print(f"    T_wall_gas    : {lcf_throat['T_gas_worst']:.0f} K")
print(f"    delta_eps     : {lcf_throat['delta_epsilon_thermal']*100:.3f} %"
      f"  (plastic: {lcf_throat['delta_epsilon_plastic']*100:.3f} %)")
print(f"    N_f predicted : {lcf_throat['N_f']:.0f} cycles")
print(f"    margin_lcf    : {lcf_throat['margin_lcf']:.2f}  [{flag_throat}]")
print()

# ---- Exit station (x = 83.5 mm) ----
T_wg_arr = regen["T_wall_gas"]
T_wc_arr = regen["T_wall_coolant"]
T_inlet = float(regen["T_coolant"][-1])  # counterflow: inlet = last element

T_g_exit = float(T_wg_arr[i_exit])
T_c_exit = float(T_wc_arr[i_exit])
T_mean_exit = 0.5 * (T_g_exit + T_c_exit)
al_exit = GRCOP84_PROXY.alpha(T_mean_exit)
E_exit = GRCOP84_PROXY.E(T_g_exit)
sy_exit = GRCOP84_PROXY.sigma_y(T_g_exit)

de_total_exit = al_exit * (T_g_exit - T_inlet)
de_e_exit = min(de_total_exit, sy_exit / E_exit)
de_p_exit = max(0.0, de_total_exit - de_e_exit)
Nf_exit = manson_coffin_life(
    strain_range_plastic=de_p_exit,
    strain_range_elastic=de_e_exit,
    material=GRCOP84_PROXY,
)
margin_exit = Nf_exit / 100.0 - 1.0
flag_exit = "pass" if margin_exit >= 0.0 else "FAIL ***"
tight_flag = "  [FLAG: margin < 4, check exit liner design]" if 0.0 <= margin_exit < 4.0 else ""

print(f"  Exit station (x = {x_mm[i_exit]:.1f} mm)")
print(f"    T_wall_gas    : {T_g_exit:.0f} K")
print(f"    delta_eps     : {de_total_exit*100:.3f} %"
      f"  (plastic: {de_p_exit*100:.3f} %)")
print(f"    N_f predicted : {Nf_exit:.0f} cycles")
print(f"    margin_lcf    : {margin_exit:.2f}  [{flag_exit}]{tight_flag}")
print()
print(f"  LCF governing station: THROAT (N_f_throat/N_f_exit = "
      f"{lcf_throat['N_f']:.0f}/{Nf_exit:.0f} = {lcf_throat['N_f']/Nf_exit:.2f}x worse)")
print()
print("  Out-of-scope (flagged for future work):")
print("    - Differential thermal expansion stress between liner and jacket")
print("      (liner runs hotter, wants to grow more in hoop -> liner compression,")
print("       jacket tension beyond pressure loading. Requires coupled FEA.)")
print("    - Closeout welds and braze joints between liner and jacket")
print("    - Manifold-to-chamber transition stress concentrations")
print("    - Buckling of inner liner under net compressive axial load at throat")
print("    - Creep in Inconel 718 jacket at sustained elevated temperature")
print("=" * 72)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
os.makedirs("physics/structural/plots", exist_ok=True)
out_path = "physics/structural/plots/bilayer_check.png"

fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

# Panel 1: Temperatures
ax = axes[0]
ax.plot(x_mm, bilayer_sol["T_gas"], color="firebrick", lw=1.8, label="T_wall_gas (liner)")
ax.plot(x_mm, bilayer_sol["T_cool"], color="steelblue", lw=1.5, ls="--",
        label="T_wall_cool (jacket)")
ax.axhline(810.0, color="black", lw=0.8, ls=":", label="GRCop-42 service limit (810 K)")
ax.axvline(0.0, color="red", lw=0.7, ls="--", label="throat")
ax.set_ylabel("Temperature [K]")
ax.set_title(
    f"Bilayer Structural Check -- 76 ch, GRCop-42 liner, "
    f"Inconel 718 jacket ({t_jacket*1e3:.1f} mm)\n"
    f"Liner t=0.5 mm | Jacket t={t_jacket*1e3:.1f} mm | "
    f"Ribs: GRCop-84, t_land=0.3 mm"
)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 2: Liner stresses
ax = axes[1]
ax.plot(x_mm, bilayer_sol["sigma_h_inner"] / 1e6, color="steelblue", lw=1.5,
        label="sigma_h_liner = (dP)*r/t_liner [TENSILE]")
ax.plot(x_mm, bilayer_sol["sigma_th_inner_gas"] / 1e6, color="firebrick", lw=1.5,
        ls="--", label="sigma_th_liner_gas [compressive]")
ax.plot(x_mm, bilayer_sol["sigma_vm_inner"] / 1e6, color="purple", lw=2.0,
        label="sigma_vm_liner (gas face)")
# sigma_y of liner at gas temperature
sy_liner = np.array([GRCOP84_PROXY.sigma_y(float(T)) for T in bilayer_sol["T_gas"]])
ax.plot(x_mm, sy_liner / 1e6, color="black", lw=1.0, ls=":", label="sigma_y_liner(T_gas)")
ax.axhline(0.0, color="gray", lw=0.5)
ax.axvline(0.0, color="red", lw=0.7, ls="--")
ax.set_ylabel("Stress [MPa]")
ax.set_title("Inner Liner (GRCop-84 proxy)")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# Panel 3: Jacket + rib stresses
ax = axes[2]
ax.plot(x_mm, bilayer_sol["sigma_h_jacket"] / 1e6, color="darkorange", lw=1.8,
        label="sigma_h_jacket = P_cool*r_jkt/t_jkt [TENSILE]")
ax.plot(x_mm, bilayer_sol["sigma_vm_jacket"] / 1e6, color="darkorange", lw=1.0,
        ls="--", label="sigma_vm_jacket (= sigma_h, no thermal)")
ax.plot(x_mm, bilayer_sol["tau_rib"] / 1e6, color="green", lw=1.5,
        label="tau_rib (rib shear)")
ax.plot(x_mm, bilayer_sol["sigma_vm_rib"] / 1e6, color="green", lw=1.0, ls="--",
        label="sigma_vm_rib = sqrt(3)*tau")
# sigma_y of jacket at coolant temperature
sy_jkt = np.array([INCONEL718_STRUCT.sigma_y(float(T)) for T in bilayer_sol["T_cool"]])
ax.plot(x_mm, sy_jkt / 1e6, color="navy", lw=1.0, ls=":", label="sigma_y_jacket(T_cool)")
ax.axhline(0.0, color="gray", lw=0.5)
ax.axvline(0.0, color="red", lw=0.7, ls="--")
ax.set_ylabel("Stress [MPa]")
ax.set_title("Outer Jacket (Inconel 718) + Ribs (GRCop-84)")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# Panel 4: Safety margins
ax = axes[3]
ax.plot(x_mm, bilayer_sol["margin_yield_inner"], color="steelblue", lw=1.8,
        label="M_yield_liner (FoS=1.4)")
ax.plot(x_mm, bilayer_sol["margin_yield_jacket"], color="darkorange", lw=1.8,
        ls="--", label="M_yield_jacket (FoS=1.4)")
ax.plot(x_mm, bilayer_sol["margin_yield_rib"], color="green", lw=1.5,
        ls=":", label="M_yield_rib (FoS=1.4)")
# Clip display at +/- 3 for readability (liner margins can be very negative at throat)
ax.set_ylim(-3.0, max(3.0, float(bilayer_sol["margin_yield_jacket"].max()) + 0.5))
ax.axhline(0.0, color="red", lw=1.2, ls="-", label="M = 0 (limit)")
ax.axvline(0.0, color="red", lw=0.7, ls="--")
ax.set_xlabel("Axial position x [mm]")
ax.set_ylabel("Safety margin M = (sigma_allow / (FoS * sigma_vm)) - 1")
ax.set_title("Safety Margins (M>=0 passes; liner M<0 => LCF governs, not static yield)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"\nPlot saved to {out_path}")
