"""
fatigue.py
==========

Low-cycle fatigue (LCF) life prediction for the inner liner of a
regeneratively-cooled rocket engine chamber.

Method
------
The Manson-Coffin-Morrow equation is used:

    Delta_epsilon / 2 = (sigma_f' / E) * (2 N_f)^b  +  epsilon_f' * (2 N_f)^c

where the first term is the elastic amplitude and the second is the plastic
amplitude.  This is solved numerically for N_f given the total strain range.

Thermal strain is the primary driver of LCF in rocket chamber liners.  At each
firing the wall cycles from near-ambient (T_coolant_inlet) to the peak
operating temperature (T_wall_gas_peak).  The strain range per cycle is:

    Delta_epsilon_thermal = alpha(T_mean) * (T_wall_gas - T_coolant_inlet)

The elastic fraction is approximated as:
    Delta_epsilon_elastic = min(Delta_epsilon_thermal, sigma_y(T_gas) / E(T_gas))

The plastic fraction is:
    Delta_epsilon_plastic = max(0, Delta_epsilon_thermal - Delta_epsilon_elastic)

Note: this strain estimate is conservative because it assumes the FULL thermal
gradient from cold soak to steady-state is recovered each cycle.  In practice,
the liner may not return to ambient between short-duration firings, reducing
the strain range and extending life.

Sources
-------
Manson-Coffin equation:
    Manson, S.S. (1954). "Behavior of Materials Under Conditions of Thermal
    Stress." NACA TN-2933. (Manson's form: Eq. 10.)
    Coffin, L.F. (1954). "A Study of the Effects of Cyclic Thermal Stresses
    on a Ductile Metal." Trans. ASME, Vol. 76, pp. 931-950.
    Combined (total-strain) form:
    Morrow, J.D. (1965). "Cyclic Plastic Strain Energy and Fatigue of
    Metals." ASTM STP-378, pp. 45-87.

Application to rocket chamber liners:
    Quentmeyer, R.J. (1977). "Experimental Fatigue Life Investigation of
    Cylindrical Thrust Chambers." NASA TM-X-73665. Sec. 3.

Thermal strain range estimate:
    NASA SP-8087 (1977) Sec. 4.4 -- cyclic thermal strain in the liner.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    from noyron.rp.structural.materials import StructuralMaterial


# ---------------------------------------------------------------------------
# Core Manson-Coffin solver
# ---------------------------------------------------------------------------


def manson_coffin_life(
    *,
    strain_range_plastic: float,
    strain_range_elastic: float,
    material: "StructuralMaterial",
) -> float:
    """
    Cycles to failure from the Manson-Coffin-Morrow equation.

    Solves:
        Delta_epsilon_p / 2 = epsilon_f' * (2 N_f)^c     [plastic part]
        Delta_epsilon_e / 2 = (sigma_f' / E_ref) * (2 N_f)^b  [elastic part]

    For the plastic part alone (Coffin-Manson):
        N_f_p = 0.5 * (Delta_epsilon_p / (2 * epsilon_f'))^(1/c)

    For the elastic part alone (Basquin):
        N_f_e = 0.5 * (Delta_epsilon_e * E_ref / (2 * sigma_f'))^(1/b)

    The combined life is estimated by solving the total equation numerically
    via bisection.  If either strain component is zero, the other controls.

    Source: Morrow (1965), ASTM STP-378; Manson (1954) NACA TN-2933.

    Parameters
    ----------
    strain_range_plastic : float
        Plastic strain range Delta_epsilon_p [-].
    strain_range_elastic : float
        Elastic strain range Delta_epsilon_e = sigma / E [-].
    material : StructuralMaterial
        Provides sigma_f_prime, b_fatigue, epsilon_f_prime, c_fatigue.

    Returns
    -------
    float
        N_f, predicted cycles to failure.  Returns math.inf if both strain
        ranges are effectively zero (< 1e-10).
    """
    eps_p = max(0.0, strain_range_plastic)
    eps_e = max(0.0, strain_range_elastic)

    if eps_p + eps_e < 1.0e-10:
        return math.inf

    sf = material.sigma_f_prime
    b = material.b_fatigue
    ef = material.epsilon_f_prime
    c = material.c_fatigue

    # Reference E for elastic term (use room-temperature as baseline since
    # sigma_f' is derived from room-temperature fatigue tests in most databases)
    E_ref = material.E(295.0)

    # Half-amplitudes
    half_p = eps_p / 2.0
    half_e = eps_e / 2.0
    half_total = half_p + half_e

    # Fast path: plastic-dominated (Coffin-Manson)
    if eps_p < 1.0e-12:
        # Elastic-dominated (Basquin)
        two_Nf = (half_e / (sf / E_ref)) ** (1.0 / b)
        return float(two_Nf) / 2.0

    if eps_e < 1.0e-12:
        # Plastic-dominated
        two_Nf = (half_p / ef) ** (1.0 / c)
        return float(two_Nf) / 2.0

    # General case: bisection on 2*N_f
    # f(2Nf) = (sf/E)*( 2Nf)^b + ef*(2Nf)^c  -  half_total = 0
    def f(two_nf: float) -> float:
        return (sf / E_ref) * two_nf ** b + ef * two_nf ** c - half_total

    # Bracket: start at 2Nf = 1 (lower) and grow upper bound until straddled.
    # Both b and c are negative so f(2Nf) is monotone decreasing.
    # f(lo=1) > 0 and f(hi) > 0  →  strain < endurance-limit of the model
    #   → life > 1e10 cycles: return math.inf (below model applicability).
    # f(lo=1) < 0  →  strain > 1-cycle amplitude → immediate failure.
    lo = 1.0
    f_lo = f(lo)
    if f_lo < 0.0:
        # Strain amplitude exceeds the 1-cycle Manson-Coffin prediction.
        return 0.5  # < 1 cycle: immediate failure

    hi = 2.0e12   # 1 trillion reversals = 500 billion cycles
    f_hi = f(hi)

    if f_hi > 0.0:
        # Even at 1e12 reversals the amplitude is above the M-C prediction.
        # Strain is below the model's effective endurance limit.
        return math.inf

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 0.5:
            break

    two_Nf = 0.5 * (lo + hi)
    return float(two_Nf) / 2.0


# ---------------------------------------------------------------------------
# LCF check using regen solution
# ---------------------------------------------------------------------------


def lcf_check(
    *,
    regen_solution: dict,
    structural_solution: dict,
    n_cycles_required: float,
    material: "StructuralMaterial",
) -> dict:
    """
    Predict LCF life at the worst-loaded axial station and report margin.

    Thermal strain range per cycle (Sec. 4.4, NASA SP-8087):
        Delta_epsilon = alpha(T_mean) * (T_wall_gas - T_coolant_inlet)

    where T_coolant_inlet is the coolant temperature entering the circuit
    (last element of T_coolant array in counterflow convention).

    The elastic and plastic fractions are split at sigma_y(T_gas):
        Delta_epsilon_e = min(Delta_epsilon, sigma_y(T_gas) / E(T_gas))
        Delta_epsilon_p = max(0, Delta_epsilon - Delta_epsilon_e)

    Worst station: maximum Delta_epsilon_thermal (typically near the throat).

    Source: Quentmeyer (1977), NASA TM-X-73665. Sec. 3.

    Parameters
    ----------
    regen_solution : dict    Output of solve_regen().
    structural_solution : dict   Output of liner_stress_profile().
    n_cycles_required : float    Required design life [cycles].
    material : StructuralMaterial

    Returns
    -------
    dict with keys:
        x_worst [m]              axial location of worst station
        i_worst                  array index of worst station
        T_gas_worst [K]          gas-side temperature at worst station
        T_cool_worst [K]         coolant-side temperature at worst station
        delta_epsilon_thermal    total thermal strain range [-]
        delta_epsilon_elastic    elastic portion [-]
        delta_epsilon_plastic    plastic portion [-]
        N_f [cycles]             predicted life
        margin_lcf               N_f / n_cycles_required - 1
        passes                   True if margin_lcf >= 0
    """
    x_arr = regen_solution["x"]
    T_wg = regen_solution["T_wall_gas"]
    T_wc = regen_solution["T_wall_coolant"]
    T_cool_arr = regen_solution["T_coolant"]

    # Coolant inlet temperature: in counterflow, coolant enters at exit (last index)
    T_inlet = float(T_cool_arr[-1])

    n = len(x_arr)
    delta_eps_arr = numpy.zeros(n)
    for i in range(n):
        T_g = float(T_wg[i])
        T_c = float(T_wc[i])
        T_mean = 0.5 * (T_g + T_c)
        al = material.alpha(T_mean)
        delta_eps_arr[i] = al * (T_g - T_inlet)

    i_worst = int(numpy.argmax(delta_eps_arr))
    T_g_w = float(T_wg[i_worst])
    T_c_w = float(T_wc[i_worst])
    T_mean_w = 0.5 * (T_g_w + T_c_w)

    de_total = float(delta_eps_arr[i_worst])
    E_w = material.E(T_g_w)
    sy_w = material.sigma_y(T_g_w)
    al_w = material.alpha(T_mean_w)

    de_e = min(de_total, sy_w / E_w)
    de_p = max(0.0, de_total - de_e)

    N_f = manson_coffin_life(
        strain_range_plastic=de_p,
        strain_range_elastic=de_e,
        material=material,
    )

    margin = N_f / n_cycles_required - 1.0 if n_cycles_required > 0.0 else float("inf")

    return {
        "x_worst": float(x_arr[i_worst]),
        "i_worst": i_worst,
        "T_gas_worst": T_g_w,
        "T_cool_worst": T_c_w,
        "delta_epsilon_thermal": de_total,
        "delta_epsilon_elastic": de_e,
        "delta_epsilon_plastic": de_p,
        "N_f": N_f,
        "margin_lcf": margin,
        "passes": margin >= 0.0,
    }
