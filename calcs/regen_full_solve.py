"""
regen_full_solve.py
===================

Coupled 1-D regenerative cooling solve for the reference methalox engine.

Uses the same gas and nozzle conditions as bartz_heat_flux_profile.py
(Pc = 1000 psia, O/F = 3.6) and adds a full cooling-channel model with
LCH4 coolant (CoolProp), Dittus-Boelter h_c, Haaland friction, and a
GRCop-42 wall.

Channel geometry: 48 rectangular channels chosen so the circumferential
arc per channel at the throat (pi * D_t / n_ch = pi * 22.3 mm / 48 =
1.46 mm) accommodates w + t_land = 1.1 + 0.35 = 1.45 mm.

Outputs:
    - Plot of T_wall_gas, T_wall_coolant, T_coolant, q_dot vs x saved to
      physics/regen/plots/regen_full_solve.png
    - Printed summary: peak T_wall_gas, coolant DeltaT, coolant DeltaP, total Q.
      Compare peak T_wall to GRCop-42 continuous service limit of 810 K
      (Ellis 2005, NASA/TM-2005-213566; see also NASA SP-8087 Ch. 4).

Run with:
    python calcs/regen_full_solve.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib  # type: ignore[import-untyped]

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np

from noyron.rp.nozzle.rao import RaoNozzle
from noyron.rp.regen.bartz import CombustionGas
from noyron.rp.regen.coolant import GRCOP42, CoolingChannelGeometry, CoolantInlet
from noyron.rp.regen.solver import solve_regen

# ---------------------------------------------------------------------------
# Reference engine (CLAUDE.md)
# ---------------------------------------------------------------------------
GAS = CombustionGas(
    Tc=3561.2,       # K   (CEA, O/F=3.6, Pc=1000 psia)
    Pc=6.8948e6,     # Pa
    gamma=1.15,      # approximate methalox
    MW=21.9,         # kg/kmol
    cp=2000.0,       # J/(kg.K)
    mu=8.0e-5,       # Pa.s
    Pr=0.60,
    c_star=1893.3,   # m/s
)

NOZZLE = RaoNozzle(r_throat=0.022313 / 2.0, epsilon=10.605)

# 48-channel geometry: w=1.1 mm, H=3.0 mm, t_wall=0.4 mm, t_land=0.35 mm
# Arc per channel at throat: pi * 22.313 mm / 48 = 1.460 mm
# Channel pitch: w + t_land = 1.10 + 0.35 = 1.45 mm (fits within arc)
GEOM = CoolingChannelGeometry(
    n_channels=48,
    width=lambda x: 1.1e-3,          # 1.1 mm
    height=lambda x: 3.0e-3,         # 3.0 mm (radial depth)
    wall_thickness=lambda x: 4.0e-4,  # 0.4 mm
    land_thickness=lambda x: 3.5e-4,  # 0.35 mm
)

# LCH4 coolant: fuel mass flow at pump discharge conditions
COOLANT = CoolantInlet(T=150.0, P=12.0e6, m_dot=0.3096, fluid="Methane")

# ---------------------------------------------------------------------------
# Run solver (counterflow: coolant enters at exit, flows to inlet)
# ---------------------------------------------------------------------------
print("Running coupled regen solver (counterflow, n=300 stations)...")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    result = solve_regen(
        nozzle=NOZZLE,
        gas=GAS,
        channel_geom=GEOM,
        coolant_inlet=COOLANT,
        wall_material=GRCOP42,
        direction="counterflow",
        n_stations=300,
        tol=0.5,
    )

x_arr = result["x"] * 1e3         # mm
T_wg  = result["T_wall_gas"]      # K
T_wc  = result["T_wall_coolant"]  # K
T_c   = result["T_coolant"]       # K
P_c   = result["P_coolant"]       # Pa
q_dot = result["q_dot"] / 1e6    # MW/m2
dP    = result["dP_total"]        # Pa

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
T_peak = float(T_wg.max())
i_peak = int(np.argmax(T_wg))

# Counterflow: coolant enters at index -1 (exit), exits at index 0 (inlet)
T_c_in  = float(T_c[-1])   # coolant inlet temperature
T_c_out = float(T_c[0])    # coolant outlet temperature
dT_cool = T_c_out - T_c_in

# Total heat transferred (integrate q_dot over gas-side area)
r_arr = result["r"]
A_gas_arr = 2.0 * np.pi * r_arr / GEOM.n_channels  # m2/m per channel
dx_m = float((result["x"][1] - result["x"][0]))
Q_total = float(np.sum(result["q_dot"] * A_gas_arr) * dx_m * GEOM.n_channels)

GRCOP42_LIMIT_K = 810.0  # K, continuous service limit
# Source: Ellis, D.L. (2005). NASA/TM-2005-213566 Sec. 3.3 (GRCop-84 proxy);
#         also cited in NASA SP-8087 Ch. 4 for copper alloy chamber liners.

print()
print("=" * 65)
print("Regen Full Solve -- Reference Engine (Pc=1000 psia, O/F=3.6)")
print("=" * 65)
print(f"  Channel geometry  : {GEOM.n_channels} channels, "
      f"w=1.1 mm, H=3.0 mm, t_wall=0.4 mm")
print(f"  Coolant           : LCH4, {T_c_in:.0f} K inlet, {COOLANT.P/1e6:.0f} MPa")
print("  Flow direction    : counterflow (exit -> inlet)")
print()
print(f"  Peak T_wall_gas   : {T_peak:.1f} K  at x = {x_arr[i_peak]:.1f} mm")
print(f"  GRCop-42 limit    : {GRCOP42_LIMIT_K:.0f} K  (Ellis 2005, NASA/TM-2005-213566)")
if T_peak > GRCOP42_LIMIT_K:
    excess = T_peak - GRCOP42_LIMIT_K
    print(f"  *** EXCEEDS limit by {excess:.0f} K -- channel geometry requires redesign ***")
else:
    print("  Wall temperature within GRCop-42 continuous limit.")
print()
print(f"  Coolant DeltaT    : {dT_cool:.1f} K  ({T_c_in:.0f} -> {T_c_out:.0f} K)")
print(f"  Coolant DeltaP    : {dP/1e6:.3f} MPa")
print(f"  Total Q absorbed  : {Q_total/1e3:.1f} kW")
print("=" * 65)

# Print any solver warnings
if caught:
    print("\nSolver warnings:")
    seen = set()
    for w in caught:
        msg = str(w.message)
        if msg not in seen:
            seen.add(msg)
            print(f"  [WARNING] {msg}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
os.makedirs("physics/regen/plots", exist_ok=True)
out_path = "physics/regen/plots/regen_full_solve.png"

x_inlet_mm = NOZZLE.x_inlet * 1e3
x_exit_mm  = NOZZLE.x_exit  * 1e3

fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

# Panel 1: Wall temperatures
ax = axes[0]
ax.plot(x_arr, T_wg,  color="firebrick",   lw=1.8, label="T_wall_gas")
ax.plot(x_arr, T_wc,  color="darkorange",  lw=1.5, label="T_wall_coolant", ls="--")
ax.plot(x_arr, T_c,   color="steelblue",   lw=1.5, label="T_coolant (bulk)", ls="-.")
ax.axhline(GRCOP42_LIMIT_K, color="black", lw=0.9, ls=":", label=f"GRCop-42 limit ({GRCOP42_LIMIT_K:.0f} K)")
ax.axvline(0.0, color="red", lw=0.8, ls="--", label="throat")
ax.set_ylabel("Temperature [K]")
ax.set_title(
    "1-D Coupled Regen Cooling — Reference Engine\n"
    f"Pc=1000 psia, O/F=3.6, LCH4 counterflow, {GEOM.n_channels} channels"
)
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# Panel 2: Heat flux
ax = axes[1]
ax.plot(x_arr, q_dot, color="firebrick", lw=1.8)
ax.axvline(0.0, color="red", lw=0.8, ls="--")
ax.set_ylabel("q_dot  [MW/m²]")
ax.grid(True, alpha=0.3)

# Panel 3: Coolant pressure
ax = axes[2]
ax.plot(x_arr, P_c / 1e6, color="steelblue", lw=1.8)
ax.axvline(0.0, color="red", lw=0.8, ls="--", label="throat")
ax.set_xlabel("Axial position x  [mm]")
ax.set_ylabel("P_coolant  [MPa]")
ax.grid(True, alpha=0.3)

# Shade throat region on all axes
for ax in axes:
    ax.axvline(x_inlet_mm, color="gray", lw=0.7, ls=":")
    ax.axvline(x_exit_mm,  color="gray", lw=0.7, ls="-.")

fig.tight_layout()
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"\nPlot saved to {out_path}")
