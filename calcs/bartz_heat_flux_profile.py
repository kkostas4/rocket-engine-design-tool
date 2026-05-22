"""
bartz_heat_flux_profile.py
==========================

Uncoupled Bartz heat flux profile for the reference engine.

Uses the reference engine conditions from CLAUDE.md (Pc = 1000 psia,
O/F = 3.6) with a constant wall temperature guess of 800 K.  The coolant
coupling comes in Task 3 (regen_full_solve.py).

Outputs:
    - Plot of h_g(x) and q_dot_wall(x) saved to
      physics/regen/plots/bartz_heat_flux_profile.png
    - Printed table of q_dot at chamber inlet, throat, and exit.

Run with:
    python calcs/bartz_heat_flux_profile.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib  # type: ignore[import-untyped]

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]

from noyron.rp.nozzle.rao import RaoNozzle
from noyron.rp.regen.bartz import CombustionGas, heat_flux_profile

# ---------------------------------------------------------------------------
# Reference engine (CLAUDE.md)
# ---------------------------------------------------------------------------
GAS = CombustionGas(
    Tc=3561.2,  # K   (CEA, O/F=3.6, Pc=1000 psia)
    Pc=6.8948e6,  # Pa
    gamma=1.15,  # approximate methalox
    MW=21.9,  # kg/kmol
    cp=2000.0,  # J/(kg.K)  -- approximate; use live CEA in coupled solver
    mu=8.0e-5,  # Pa.s      -- approximate
    Pr=0.60,  # --
    c_star=1893.3,  # m/s
)

NOZZLE = RaoNozzle(r_throat=0.022313 / 2.0, epsilon=10.605)
T_WALL_GUESS = 800.0  # K -- uncoupled constant wall temperature

# ---------------------------------------------------------------------------
# Compute profile
# ---------------------------------------------------------------------------
prof = heat_flux_profile(nozzle=NOZZLE, gas=GAS, T_wall_guess=T_WALL_GUESS, n=300)

x = prof["x"] * 1e3  # mm
h_g = prof["h_g"] / 1e3  # kW/(m2.K)
q_dot = prof["q_dot_wall"] / 1e6  # MW/m2
M = prof["M"]
T_aw = prof["T_aw"]

x_throat_mm = 0.0
x_inlet_mm = NOZZLE.x_inlet * 1e3
x_exit_mm = NOZZLE.x_exit * 1e3

# Point values
def _nearest(x_arr: np.ndarray, x_target: float) -> int:
    return int(np.argmin(np.abs(x_arr - x_target)))


i_inlet = 0
i_throat = _nearest(x, x_throat_mm)
i_exit = -1

# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------
print()
print("Bartz Heat Flux Profile -- Reference Engine (Pc = 1000 psia, O/F = 3.6)")
print(f"  T_wall (constant guess) = {T_WALL_GUESS:.0f} K")
print(f"  {'Station':<18} {'x [mm]':>8} {'M [-]':>7} {'T_aw [K]':>10} "
      f"{'h_g [kW/(m2K)]':>16} {'q_dot [MW/m2]':>14}")
print("-" * 80)

for label, idx in [("Convergent inlet", i_inlet), ("Throat", i_throat), ("Exit", i_exit)]:
    print(f"  {label:<18} {x[idx]:>8.1f} {M[idx]:>7.3f} {T_aw[idx]:>10.1f} "
          f"{h_g[idx]:>16.2f} {q_dot[idx]:>14.2f}")

print("-" * 80)
print(f"  Peak q_dot = {q_dot.max():.2f} MW/m2 at x = {x[np.argmax(q_dot)]:.1f} mm")
print()

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
os.makedirs("physics/regen/plots", exist_ok=True)
out_path = "physics/regen/plots/bartz_heat_flux_profile.png"

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

ax1.plot(x, h_g, color="steelblue", lw=1.8)
ax1.axvline(x_throat_mm, color="red", lw=0.9, ls="--", label="throat")
ax1.axvline(x_inlet_mm, color="gray", lw=0.7, ls=":", label="inlet")
ax1.axvline(x_exit_mm, color="gray", lw=0.7, ls="-.", label="exit")
ax1.set_ylabel("h_g  [kW/(m2*K)]")
ax1.set_title(
    "Bartz Gas-Side Heat Transfer -- Reference Engine\n"
    f"Pc = 1000 psia, O/F = 3.6, T_wall = {T_WALL_GUESS:.0f} K (uncoupled)"
)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

ax2.plot(x, q_dot, color="firebrick", lw=1.8)
ax2.axvline(x_throat_mm, color="red", lw=0.9, ls="--")
ax2.axvline(x_inlet_mm, color="gray", lw=0.7, ls=":")
ax2.axvline(x_exit_mm, color="gray", lw=0.7, ls="-.")
ax2.set_xlabel("Axial position x  [mm]")
ax2.set_ylabel("q_dot  [MW/m2]")
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Plot saved to {out_path}")
