"""
injector_pc_sweep.py
====================

Parametric sweep of chamber pressure (Pc) from 500 to 1500 psia in steps of
100 psia, holding dP/Pc = 0.45 constant.

NOTE: m_dot held constant at reference conditions (CEA sweep only covers
Pc=1000 psia). In reality, Isp rises slightly with Pc so m_dot would decrease
further, amplifying the orifice shrinkage trend.

Reference engine (Pc = 1000 psia):
    F       = 4448.22 N
    c_eff   = 3123.9 m/s
    O/F     = 3.60
    rho_fuel = 386 kg/m³ (LCH4 at 118 K)
    rho_ox   = 1116 kg/m³ (LOX at 100 K)

Usage
-----
    python calcs/injector_pc_sweep.py
"""

from __future__ import annotations

from physics.injector.pintle import PintleInjector
from physics.util.units import PSI_TO_PA

# ---------------------------------------------------------------------------
# Reference engine — held constant across sweep
# ---------------------------------------------------------------------------
THRUST_N = 4448.22          # N
C_EFF_M_S = 3123.9          # m/s  — effective exhaust velocity at ref point

M_DOT_TOTAL = THRUST_N / C_EFF_M_S    # kg/s

OF_RATIO = 3.60
M_DOT_FUEL = M_DOT_TOTAL / (1.0 + OF_RATIO)   # kg/s
M_DOT_OX = M_DOT_TOTAL - M_DOT_FUEL            # kg/s

RHO_FUEL = 386.0    # kg/m³ — LCH4 at 118 K (fixed)
RHO_OX = 1116.0     # kg/m³ — LOX at 100 K  (fixed)

CD = 0.6
PINTLE_OD = 0.030   # m
N_FUEL = 20

DP_PC_RATIO = 0.45  # fixed stiffness ratio across sweep

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
PC_START_PSIA = 500
PC_STOP_PSIA = 1500
PC_STEP_PSIA = 100

header = (
    f"{'Pc [psia]':>10} | {'dP [psi]':>9} | {'m_dot_f [kg/s]':>14} | "
    f"{'d_orifice [mm]':>14} | {'LOX ann OD [mm]':>15} | "
    f"{'V_ox [m/s]':>10} | {'V_fuel [m/s]':>12} | {'TMR':>6}"
)
separator = "-" * len(header)

print(separator)
print(header)
print(separator)

pc_psia = PC_START_PSIA
while pc_psia <= PC_STOP_PSIA:
    Pc = pc_psia * PSI_TO_PA
    dP = DP_PC_RATIO * Pc

    p = PintleInjector(
        m_dot_ox=M_DOT_OX,
        m_dot_fuel=M_DOT_FUEL,
        rho_ox=RHO_OX,
        rho_fuel=RHO_FUEL,
        dP_inj=dP,
        Cd_ox=CD,
        Cd_fuel=CD,
        pintle_OD=PINTLE_OD,
        n_fuel_orifices=N_FUEL,
    )

    dP_psi = dP / PSI_TO_PA
    d_mm = p.fuel_orifice_diameter * 1e3
    lox_od_mm = p.lox_annulus_OD * 1e3

    print(
        f"{pc_psia:>10} | {dP_psi:>9.1f} | {M_DOT_FUEL:>14.4f} | "
        f"{d_mm:>14.4f} | {lox_od_mm:>15.4f} | "
        f"{p.V_ox:>10.2f} | {p.V_fuel:>12.2f} | {p.TMR:>6.4f}"
    )

    pc_psia += PC_STEP_PSIA

print(separator)
