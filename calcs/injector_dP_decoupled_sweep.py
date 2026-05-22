"""
injector_dP_decoupled_sweep.py
===============================

Re-runs the injector Pc sweep with dP_fuel = 1.5 * dP_ox so that TMR
becomes a genuine design variable instead of the algebraic identity 1.000
that appeared in the uniform-ΔP era.

NOTE: m_dot is held constant at reference conditions because the CEA CSV
wrapper only covers Pc = 1000 psia.  In reality, Isp rises slightly with Pc
so m_dot would decrease further, amplifying the orifice shrinkage trend and
slightly raising velocities.

Reference engine baseline:
    thrust = 4448.22 N, c_eff = 3123.9 m/s, O/F = 3.60
    m_dot_total = 1.4239 kg/s  (fixed across sweep)
    m_dot_fuel  = 0.3096 kg/s
    m_dot_ox    = 1.1144 kg/s
    rho_fuel = 386 kg/m³, rho_ox = 1116 kg/m³  (density ~ constant vs Pc)

Run with:
    python calcs/injector_dP_decoupled_sweep.py
"""

from __future__ import annotations

from physics.injector.pintle import PintleInjector
from physics.injector.stability import check_stiffness
from noyron.kernel.util.units import PSI_TO_PA

# ---------------------------------------------------------------------------
# Fixed propellant and engine parameters
# ---------------------------------------------------------------------------
M_DOT_FUEL = 0.3096    # kg/s
M_DOT_OX = 1.1144      # kg/s
RHO_FUEL = 386.0       # kg/m³
RHO_OX = 1116.0        # kg/m³
CD_OX = 0.6
CD_FUEL = 0.6
PINTLE_OD = 0.030      # m
N_FUEL = 20

# Design choice: dP_fuel is 1.5× the oxidiser drop.
DP_OX_FRAC_PC = 0.45   # dP_ox / Pc
DP_FUEL_FACTOR = 1.5   # dP_fuel = DP_FUEL_FACTOR * dP_ox

# Pc sweep range
PC_MIN_PSIA = 500
PC_MAX_PSIA = 1500
PC_STEP_PSIA = 100


def run_sweep() -> None:
    """Perform the sweep and print a formatted table."""
    header = (
        f"{'Pc':>9} {'dP_ox':>8} {'dP_f':>8} "
        f"{'d_orif':>9} {'LOX OD':>9} "
        f"{'V_ox':>8} {'V_fuel':>8} {'TMR':>7} "
        f"{'stiff_ox':>10} {'stiff_f':>9}"
    )
    units = (
        f"{'[psia]':>9} {'[psi]':>8} {'[psi]':>8} "
        f"{'[mm]':>9} {'[mm]':>9} "
        f"{'[m/s]':>8} {'[m/s]':>8} {'[-]':>7} "
        f"{'[dP/Pc]':>10} {'[dP/Pc]':>9}"
    )
    sep = "-" * len(header)

    print()
    print("Noyron-Clone — Injector Decoupled-dP Pc Sweep")
    print(f"  dP_ox / Pc = {DP_OX_FRAC_PC:.2f}  |  dP_fuel = {DP_FUEL_FACTOR:.1f} × dP_ox")
    print(f"  m_dot_fuel = {M_DOT_FUEL} kg/s (fixed — CEA only at Pc=1000 psia)")
    print()
    print(header)
    print(units)
    print(sep)

    for pc_psia in range(PC_MIN_PSIA, PC_MAX_PSIA + PC_STEP_PSIA, PC_STEP_PSIA):
        Pc = pc_psia * PSI_TO_PA
        dP_ox = DP_OX_FRAC_PC * Pc
        dP_fuel = DP_FUEL_FACTOR * dP_ox

        p = PintleInjector(
            m_dot_ox=M_DOT_OX,
            m_dot_fuel=M_DOT_FUEL,
            rho_ox=RHO_OX,
            rho_fuel=RHO_FUEL,
            dP_ox=dP_ox,
            dP_fuel=dP_fuel,
            Cd_ox=CD_OX,
            Cd_fuel=CD_FUEL,
            pintle_OD=PINTLE_OD,
            n_fuel_orifices=N_FUEL,
        )

        stiff = check_stiffness(dP_ox=dP_ox, dP_fuel=dP_fuel, Pc=Pc)
        r_ox = stiff["ox"][1]
        r_fuel = stiff["fuel"][1]
        psi = 6_894.7572931783

        print(
            f"{pc_psia:>9.0f} "
            f"{dP_ox/psi:>8.1f} "
            f"{dP_fuel/psi:>8.1f} "
            f"{p.fuel_orifice_diameter*1e3:>9.4f} "
            f"{p.lox_annulus_OD*1e3:>9.4f} "
            f"{p.V_ox:>8.2f} "
            f"{p.V_fuel:>8.2f} "
            f"{p.TMR:>7.4f} "
            f"{r_ox:>10.4f} "
            f"{r_fuel:>9.4f}"
        )

    print(sep)
    print()
    tmr_analytic = (1.0 / (M_DOT_OX / M_DOT_FUEL)) * (RHO_OX / RHO_FUEL * DP_FUEL_FACTOR) ** 0.5
    print(
        f"NOTE: TMR != 1.000 because dP_fuel != dP_ox.\n"
        f"  With dP_fuel = {DP_FUEL_FACTOR}x dP_ox and equal Cd:\n"
        f"  TMR = (1/MR) * sqrt(rho_ox/rho_fuel * dP_fuel/dP_ox)\n"
        f"      = (1/{M_DOT_OX/M_DOT_FUEL:.1f}) * sqrt({RHO_OX}/{RHO_FUEL:.0f} * {DP_FUEL_FACTOR})\n"
        f"      = {tmr_analytic:.4f}  (Dressler & Bauer target: 0.5 - 2.0)"
    )


if __name__ == "__main__":
    run_sweep()
