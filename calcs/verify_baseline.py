"""
verify_baseline.py
==================

Reproduces every top-level number from the reference engine spec in CLAUDE.md
using first-principles equations. This is the canonical sanity check for the
project.

Run with:
    python calcs/verify_baseline.py

Expected: all checks pass with deviation < tolerance. If any fail, the issue
is either (a) a bug in the implementation, or (b) the reference numbers are
stale and CLAUDE.md needs updating. Either way, stop and investigate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# --- Reference values from CLAUDE.md (frozen baseline) ---------------------

REFERENCE = {
    "thrust_N": 4448.2216,
    "Pc_Pa": 6_894_757.29,
    "of_ratio": 3.60,
    "expansion_ratio": 10.605,
    "isp_delivered_s": 318.44,
    "cstar_m_s": 1893.3,
    "c_eff_m_s": 3123.9,
    "mdot_total_kg_s": 1.4239,
    "mdot_lox_kg_s": 1.1144,
    "mdot_lch4_kg_s": 0.30955,
    "throat_area_m2": 3.910116930626121e-4,
    "throat_dia_m": 0.022312587,
    "exit_area_m2": 4.146679005e-3,
    "exit_dia_m": 0.072661652,
    "Ac_At": 4.0,
    "chamber_dia_m": 0.044625175,
    "Lstar_m": 0.5,
    "chamber_length_m": 0.125,
}

G0 = 9.80665  # m/s^2 standard gravity


# --- Helpers ---------------------------------------------------------------

@dataclass
class Check:
    name: str
    expected: float
    computed: float
    tolerance_frac: float

    @property
    def deviation(self) -> float:
        if self.expected == 0:
            return abs(self.computed)
        return abs(self.computed - self.expected) / abs(self.expected)

    @property
    def passed(self) -> bool:
        return self.deviation <= self.tolerance_frac

    def report(self) -> str:
        flag = "PASS" if self.passed else "FAIL"
        return (
            f"[{flag}] {self.name:<32} "
            f"expected={self.expected:.6g}  "
            f"computed={self.computed:.6g}  "
            f"dev={self.deviation*100:.3f}%"
        )


def diameter_from_area(area_m2: float) -> float:
    return math.sqrt(4.0 * area_m2 / math.pi)


def area_from_diameter(dia_m: float) -> float:
    return math.pi * dia_m ** 2 / 4.0


# --- Independent recalculations --------------------------------------------

def check_isp_from_c_eff() -> Check:
    """Isp = c_eff / g0"""
    isp = REFERENCE["c_eff_m_s"] / G0
    return Check("Isp from c_eff/g0", REFERENCE["isp_delivered_s"], isp, 0.005)


def check_mdot_total() -> Check:
    """m_dot = F / c_eff = F / (Isp * g0)"""
    mdot = REFERENCE["thrust_N"] / REFERENCE["c_eff_m_s"]
    return Check("m_dot_total from F/c_eff", REFERENCE["mdot_total_kg_s"], mdot, 0.01)


def check_mdot_split() -> tuple[Check, Check]:
    """m_dot_ox = m_tot * OF/(OF+1), m_dot_f = m_tot * 1/(OF+1)"""
    of = REFERENCE["of_ratio"]
    mtot = REFERENCE["mdot_total_kg_s"]
    mdot_ox = mtot * of / (of + 1.0)
    mdot_f = mtot * 1.0 / (of + 1.0)
    return (
        Check("m_dot_LOX from O/F split", REFERENCE["mdot_lox_kg_s"], mdot_ox, 0.01),
        Check("m_dot_LCH4 from O/F split", REFERENCE["mdot_lch4_kg_s"], mdot_f, 0.01),
    )


def check_throat_area_from_cstar() -> Check:
    """At = m_dot * c* / Pc"""
    at = (REFERENCE["mdot_total_kg_s"] * REFERENCE["cstar_m_s"]) / REFERENCE["Pc_Pa"]
    return Check("Throat area from m_dot*c*/Pc", REFERENCE["throat_area_m2"], at, 0.01)


def check_throat_diameter() -> Check:
    dt = diameter_from_area(REFERENCE["throat_area_m2"])
    return Check("Throat diameter from At", REFERENCE["throat_dia_m"], dt, 0.005)


def check_exit_area() -> Check:
    """Ae = eps * At"""
    ae = REFERENCE["expansion_ratio"] * REFERENCE["throat_area_m2"]
    return Check("Exit area from eps*At", REFERENCE["exit_area_m2"], ae, 0.005)


def check_exit_diameter() -> Check:
    de = diameter_from_area(REFERENCE["exit_area_m2"])
    return Check("Exit diameter from Ae", REFERENCE["exit_dia_m"], de, 0.005)


def check_chamber_diameter() -> Check:
    """Dc = sqrt(4 * Ac/At * At / pi)"""
    ac = REFERENCE["Ac_At"] * REFERENCE["throat_area_m2"]
    dc = diameter_from_area(ac)
    return Check("Chamber diameter from Ac/At=4", REFERENCE["chamber_dia_m"], dc, 0.005)


def check_chamber_length() -> Check:
    """Lc = L* * At / Ac"""
    lc = REFERENCE["Lstar_m"] * REFERENCE["throat_area_m2"] / (REFERENCE["Ac_At"] * REFERENCE["throat_area_m2"])
    return Check("Chamber length from L* and Ac", REFERENCE["chamber_length_m"], lc, 0.005)


# --- Main ------------------------------------------------------------------

def main() -> int:
    checks: list[Check] = []
    checks.append(check_isp_from_c_eff())
    checks.append(check_mdot_total())
    a, b = check_mdot_split()
    checks.extend([a, b])
    checks.append(check_throat_area_from_cstar())
    checks.append(check_throat_diameter())
    checks.append(check_exit_area())
    checks.append(check_exit_diameter())
    checks.append(check_chamber_diameter())
    checks.append(check_chamber_length())

    print("=" * 72)
    print("Noyron-Clone — Baseline Engine Verification")
    print("=" * 72)
    for c in checks:
        print(c.report())
    print("=" * 72)

    fails = [c for c in checks if not c.passed]
    print(f"  {len(checks) - len(fails)}/{len(checks)} checks passed.")
    if fails:
        print(f"  {len(fails)} FAILURES — investigate before proceeding.")
        return 1
    print("  All baseline numbers reproduce. Reference engine is consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
