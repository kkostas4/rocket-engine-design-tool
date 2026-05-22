"""
rao_contour_check.py
====================

Plots Rao parabolic bell contours for eps = 10, 20, 40, 80 (length_fraction = 0.80
default angles) and prints the actual bell length fraction for each.

Run with:
    python calcs/rao_contour_check.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib  # type: ignore[import-untyped]

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]

from noyron.rp.nozzle.rao import RaoNozzle

EPS_LIST = [10, 20, 40, 80]
R_T = 0.050  # 50 mm throat radius (representative scale)

os.makedirs("physics/nozzle/plots", exist_ok=True)
out_path = "physics/nozzle/plots/rao_contours.png"

fig, ax = plt.subplots(figsize=(10, 5))
colours = plt.cm.viridis(np.linspace(0.15, 0.85, len(EPS_LIST)))

print()
print("Rao 80% Bell Nozzle -- Contour Check")
print("theta_n = 25 deg, theta_e = 8 deg  (Sutton & Biblarz Sec. 3.3, Table 3-2)")
print()
print(f"{'eps':>5}  {'x_exit [mm]':>12}  {'L_cone_15 [mm]':>16}  "
      f"{'Bell fraction':>15}  {'R_exit [mm]':>12}")
print("-" * 70)

for eps, col in zip(EPS_LIST, colours):
    nozzle = RaoNozzle(r_throat=R_T, epsilon=eps)
    x, r = nozzle.contour(300)

    # Normalise x by throat radius for comparison
    ax.plot(x / R_T, r / R_T, color=col, lw=1.8, label=f"eps = {eps}")

    # Report geometry
    R_e = R_T * math.sqrt(eps)
    L_cone = (R_e - R_T) / math.tan(math.radians(15.0))
    frac = nozzle.bell_length_fraction()
    print(
        f"{eps:>5}  {nozzle.x_exit*1e3:>12.2f}  {L_cone*1e3:>16.2f}  "
        f"{frac:>15.4f}  {R_e*1e3:>12.2f}"
    )

print("-" * 70)
print()
print("NOTE: Bell fraction ~0.88 rather than 0.80 because the default")
print("  angles (25 deg, 8 deg) from Sutton & Biblarz Table 3-2 are tabulated")
print("  for eps ~ 10-20.  For exact 80% bell at other expansion ratios,")
print("  use Rao (1958) Table 1 to look up theta_n and theta_e.")

ax.axvline(0.0, color="black", lw=0.8, ls=":", label="throat")
ax.set_xlabel("x / R_throat  [-]")
ax.set_ylabel("r / R_throat  [-]")
ax.set_title(
    "Rao Parabolic Bell Nozzle Contours\n"
    "theta_n = 25 deg, theta_e = 8 deg  (Sutton & Biblarz Sec. 3.3)"
)
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"\nPlot saved to {out_path}")
