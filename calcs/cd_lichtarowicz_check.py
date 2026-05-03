"""
cd_lichtarowicz_check.py
========================

Verification of the Lichtarowicz–Duggins–Markland (1965) discharge
coefficient correlation against:

  1. The asymptotic high-Re limit Cd_inf = 0.827 − 0.0085·(L/D).
  2. The expected shape: Cd increases with Re and decreases with L/D.
  3. A plot of Cd vs Re for L/D = 2, 4, 6, 8, 10 saved to
     physics/injector/plots/cd_lichtarowicz.png.

The paper itself (Proc. IMechE 1965) does not publish a compact numerical
table suitable for point-by-point reproduction, but it does state
Equation (13) explicitly, and Figures 6–8 of the paper show the expected
monotonic increase in Cd with Re and decrease with L/D.  The checks below
validate that the implementation reproduces the stated formula and has the
correct qualitative behaviour.

Reference
---------
    Lichtarowicz, A., Duggins, R.K. and Markland, E. (1965).
    "Discharge coefficients for incompressible non-cavitating flow through
    long orifices."
    Proc. IMechE, J. Mech. Eng. Sci., Vol. 7, No. 2, pp. 210–219. Eq. (13).

Run with:
    python calcs/cd_lichtarowicz_check.py
"""

from __future__ import annotations

import os
import sys
import warnings

# Add repo root to path when run directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # type: ignore[import-untyped]
import matplotlib  # type: ignore[import-untyped]
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[import-untyped]

from physics.injector.discharge import cd_lichtarowicz

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LD_VALUES = [2, 4, 6, 8, 10]
RE_RANGE = np.logspace(1, 5, 300)       # 10 → 1e5 (validity range)
RE_ASYMP = 1e6                           # well above validity, for Cd_inf check


def numerical_checks() -> bool:
    """Run point checks and print a report.  Returns True if all pass."""
    all_passed = True
    print("=" * 65)
    print("  Lichtarowicz (1965) Eq. (13) -- Numerical Checks")
    print("=" * 65)

    # 1. Asymptotic values at Re = 1e6 match Cd_inf = 0.827 - 0.0085*(L/D)
    print("\n1. Asymptotic Cd at Re = 1e6 vs Cd_inf = 0.827 - 0.0085*(L/D)")
    print(f"   {'L/D':>4}  {'Cd_inf':>8}  {'Cd(Re=1e6)':>12}  {'dev %':>8}  {'pass':>6}")
    for ld in LD_VALUES:
        cd_inf = 0.827 - 0.0085 * ld
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cd_val = cd_lichtarowicz(Re=RE_ASYMP, LD=ld)
        dev_pct = 100.0 * abs(cd_val - cd_inf) / cd_inf
        passed = dev_pct < 1.0
        all_passed &= passed
        print(
            f"   {ld:>4}  {cd_inf:>8.4f}  {cd_val:>12.6f}  "
            f"{dev_pct:>8.4f}  {'PASS' if passed else 'FAIL':>6}"
        )

    # 2. Cd is monotonically increasing with Re (at fixed L/D = 4)
    print("\n2. Monotonic increase with Re at L/D = 4 (first 5 Re values shown)")
    re_check = [50, 200, 1_000, 1e4, 1e5]
    cds = [cd_lichtarowicz(re, 4.0) for re in re_check]
    mono = all(cds[i] < cds[i + 1] for i in range(len(cds) - 1))
    all_passed &= mono
    for re, cd in zip(re_check, cds):
        print(f"   Re = {re:>8.0f}  Cd = {cd:.5f}")
    print(f"   Monotonically increasing: {'PASS' if mono else 'FAIL'}")

    # 3. Cd decreases with L/D at fixed high Re
    print("\n3. Cd decreases with L/D at Re = 2e5 (upper validity boundary)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cds_ld = [cd_lichtarowicz(2e5, ld) for ld in LD_VALUES]
    mono_ld = all(cds_ld[i] > cds_ld[i + 1] for i in range(len(cds_ld) - 1))
    all_passed &= mono_ld
    for ld, cd in zip(LD_VALUES, cds_ld):
        print(f"   L/D = {ld:>2}  Cd = {cd:.5f}")
    print(f"   Monotonically decreasing with L/D: {'PASS' if mono_ld else 'FAIL'}")

    print("\n" + "=" * 65)
    print(f"  Overall: {'ALL PASS' if all_passed else 'SOME FAILURES'}")
    print("=" * 65)
    return all_passed


def make_plot() -> str:
    """Generate Cd vs Re curves for all L/D values and save the plot."""
    os.makedirs("physics/injector/plots", exist_ok=True)
    out_path = "physics/injector/plots/cd_lichtarowicz.png"

    fig, ax = plt.subplots(figsize=(9, 5))
    colours = plt.cm.viridis(np.linspace(0.15, 0.85, len(LD_VALUES)))

    for ld, colour in zip(LD_VALUES, colours):
        cds = []
        for re in RE_RANGE:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cds.append(cd_lichtarowicz(re, ld))
        ax.semilogx(RE_RANGE, cds, color=colour, lw=1.8, label=f"L/D = {ld}")

        # Mark the asymptote as a dashed line extending above the Re range
        cd_inf = 0.827 - 0.0085 * ld
        ax.axhline(cd_inf, color=colour, lw=0.7, ls="--", alpha=0.6)

    ax.set_xlabel("Reynolds number Re [-]")
    ax.set_ylabel("Discharge coefficient Cd [-]")
    ax.set_title(
        "Lichtarowicz–Duggins–Markland (1965) Eq. (13)\n"
        "Cd vs Re for sharp-edged cylindrical orifices\n"
        "(dashed lines = high-Re asymptote Cd_∞ = 0.827 − 0.0085·L/D)"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(RE_RANGE[0], RE_RANGE[-1])
    ax.set_ylim(0.2, 0.88)

    # Shade validity region
    ax.axvspan(10, 2e5, alpha=0.05, color="green", label="_validity")
    ax.text(
        15, 0.22, "Validity: 10 ≤ Re ≤ 2×10⁵",
        fontsize=7, color="darkgreen", alpha=0.7
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    ok = numerical_checks()
    path = make_plot()
    print(f"\nPlot saved to {path}")
    sys.exit(0 if ok else 1)
