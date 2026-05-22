"""
margin.py
=========

Safety margin reporting for the liner structural analysis.

Margin convention (NASA-STD-5001B §4.2):
    M = (sigma_allow / (FoS * sigma_actual)) - 1
    M >= 0  → passes
    M <  0  → fails (flag explicitly — do NOT silently relax)

Factors of safety:
    FoS_yield    = 1.4  (general aerospace practice)
    FoS_ultimate = 2.0  (general aerospace practice)
"""

from __future__ import annotations

import numpy as np


FOS_YIELD: float = 1.4
FOS_ULTIMATE: float = 2.0


def check_all_margins(structural_solution: dict) -> dict:
    """
    Summarise the minimum safety margins across all axial stations.

    Parameters
    ----------
    structural_solution : dict
        Output of liner_stress_profile().

    Returns
    -------
    dict with keys:
        min_margin_yield    float   minimum margin_yield value
        min_margin_ultimate float   minimum margin_ultimate value
        x_at_min_yield      float   axial location [m] of minimum yield margin
        x_at_min_ultimate   float   axial location [m] of minimum ultimate margin
        any_negative_margin bool    True if any margin < 0
        peak_vm_gas         float   peak von Mises stress at gas side [Pa]
        x_at_peak_vm        float   axial location [m] of peak VM stress
    """
    x = structural_solution["x"]
    my = structural_solution["margin_yield"]
    mu = structural_solution["margin_ultimate"]
    vm = structural_solution["sigma_vm_gas"]

    i_my = int(np.argmin(my))
    i_mu = int(np.argmin(mu))
    i_vm = int(np.argmax(vm))

    return {
        "min_margin_yield": float(my[i_my]),
        "min_margin_ultimate": float(mu[i_mu]),
        "x_at_min_yield": float(x[i_my]),
        "x_at_min_ultimate": float(x[i_mu]),
        "any_negative_margin": bool(np.any(my < 0.0) or np.any(mu < 0.0)),
        "peak_vm_gas": float(vm[i_vm]),
        "x_at_peak_vm": float(x[i_vm]),
    }


def print_structural_report(
    *,
    structural_solution: dict,
    lcf_result: dict,
    material_name: str,
) -> None:
    """
    Print a formatted structural margin report to stdout.

    Parameters
    ----------
    structural_solution : dict   Output of liner_stress_profile().
    lcf_result : dict            Output of lcf_check().
    material_name : str          Material name string for the report header.
    """
    s = structural_solution
    lcf = lcf_result
    summary = check_all_margins(s)

    x_arr = s["x"]
    i_vm = int(np.argmax(s["sigma_vm_gas"]))

    print()
    print("=" * 70)
    print(f"Structural Report -- Inner Liner  [{material_name}]")
    print("=" * 70)
    print("  Margin convention : M = (sigma_allow / (FoS x sigma_actual)) - 1")
    print(f"  FoS yield         : {FOS_YIELD}")
    print(f"  FoS ultimate      : {FOS_ULTIMATE}")
    print()

    # Peak stress
    vm_pk = summary["peak_vm_gas"]
    x_vm = summary["x_at_peak_vm"]
    T_pk = float(s["T_gas"][i_vm])
    sh_pk = float(s["sigma_hoop"][i_vm])
    sth_pk = float(s["sigma_thermal_gas"][i_vm])
    print("  Peak VM stress (gas side)")
    print(f"    sigma_vm        : {vm_pk/1e6:.1f} MPa  at x = {x_vm*1e3:.1f} mm")
    print(f"    T_wall_gas      : {T_pk:.1f} K")
    print(f"    sigma_hoop      : {sh_pk/1e6:.1f} MPa")
    print(f"    sigma_th_gas    : {sth_pk/1e6:.1f} MPa  (negative = compressive)")
    print()

    # Yield margin
    my_min = summary["min_margin_yield"]
    x_my = summary["x_at_min_yield"]
    flag_y = "FAIL ***" if my_min < 0.0 else "pass"
    print(f"  Min margin (yield, FoS={FOS_YIELD})")
    print(f"    M_yield_min     : {my_min:.3f}  at x = {x_my*1e3:.1f} mm  [{flag_y}]")
    if my_min < 0.0:
        print(
            "    NOTE: Negative yield margin is expected for regen liners --\n"
            "    the liner operates in the plastic range and LCF governs failure.\n"
            "    This is not a static yield failure unless stress is load-controlled."
        )
    print()

    # Ultimate margin
    mu_min = summary["min_margin_ultimate"]
    x_mu = summary["x_at_min_ultimate"]
    flag_u = "FAIL ***" if mu_min < 0.0 else "pass"
    print(f"  Min margin (ultimate, FoS={FOS_ULTIMATE})")
    print(f"    M_ultimate_min  : {mu_min:.3f}  at x = {x_mu*1e3:.1f} mm  [{flag_u}]")
    print()

    # LCF
    Nf = lcf["N_f"]
    m_lcf = lcf["margin_lcf"]
    flag_lcf = "FAIL ***" if not lcf["passes"] else "pass"
    x_w = lcf["x_worst"] * 1e3
    de_total = lcf["delta_epsilon_thermal"]
    de_p = lcf["delta_epsilon_plastic"]
    print("  LCF prediction (Manson-Coffin)")
    print(f"    Worst station   : x = {x_w:.1f} mm")
    print(f"    T_wall_gas      : {lcf['T_gas_worst']:.1f} K")
    print(f"    delta_eps_total : {de_total*100:.3f} %")
    print(f"    delta_eps_plastic:{de_p*100:.3f} %")
    print(f"    N_f predicted   : {Nf:.0f} cycles")
    print(f"    margin_lcf      : {m_lcf:.2f}  [{flag_lcf}]")
    print()

    # Out-of-scope flags
    print("  Out-of-scope items (flag for future work):")
    print("    - Rib/land stress between coolant channels (needs 2D/rib model)")
    print("    - Buckling of the inner liner under net compressive hoop load")
    print("    - Manifold and inlet/outlet stress concentrations")
    print("    - Creep at sustained elevated temperature")
    print("    - FEA validation of thermal gradient through rib vs. channel")
    print("=" * 70)
