"""
isp_vs_mr.py
============

Sweep vacuum Isp vs mixture ratio at fixed Pc and expansion ratio for METHALOX.
Uses the live rocketcea backend via cea_isp_vac.

Usage
-----
    python calcs/isp_vs_mr.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from noyron import METHALOX
from noyron.rp.thermochem.cea import cea_isp_vac

Pc = 7e6            # Pa
eps = 10.6          # expansion ratio
MRs = np.linspace(2.5, 4.5, 21)

Isp = [cea_isp_vac(propellant=METHALOX, Pc=Pc, MR=mr, eps=eps) for mr in MRs]

plt.figure(figsize=(7, 4))
plt.plot(MRs, Isp, marker="o")
plt.xlabel("MR (O/F)")
plt.ylabel("Isp_vac [s]")
plt.grid(True)
plt.tight_layout()
plt.savefig("output/isp_vs_mr.png", dpi=120)

i = int(np.argmax(Isp))
print(f"Peak Isp_vac = {Isp[i]:.2f} s at MR = {MRs[i]:.2f}")
print("MR    Isp_vac[s]")
for mr, s in zip(MRs, Isp):
    print(f"{mr:.2f}  {s:7.2f}")