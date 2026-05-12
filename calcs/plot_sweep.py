"""
plot_sweep.py
=============

Visualise a completed noyron sweep from results.csv.

Usage
-----
    python calcs/plot_sweep.py output/pc_mr_sweep/results.csv

Produces three figures:
  1. Isp_approx vs peak T_wall_gas  (Pareto: high Isp, low T_wall)
  2. Isp_approx vs coolant dP        (Pareto: high Isp, low dP)
  3. Peak T_wall_gas vs worst margin yield  (thermal vs structural trade)

Infeasible points are shown as red crosses.  Feasible points are coloured
by chamber_pressure (default) or mixture_ratio (--color-by mr).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _load(csv_path: Path) -> tuple[list[dict], list[dict]]:
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    ok = [r for r in rows if r["status"] == "ok"]
    bad = [r for r in rows if r["status"] != "ok"]
    return ok, bad


def _col(rows: list[dict], key: str) -> list[float]:
    return [float(r[key]) for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot noyron sweep results.")
    parser.add_argument("csv", type=Path, help="Path to results.csv")
    parser.add_argument(
        "--color-by", default="chamber_pressure", metavar="FIELD",
        help="CSV column used for scatter colour (default: chamber_pressure)",
    )
    parser.add_argument("--save", type=Path, default=None, metavar="DIR",
                        help="Save figures to DIR as PNG instead of showing.")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
    except ImportError:
        print("matplotlib not installed.  Run:  pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    ok, bad = _load(args.csv)
    if not ok:
        print("No feasible design points in results.csv.", file=sys.stderr)
        sys.exit(1)

    color_vals = np.array(_col(ok, args.color_by))
    norm = plt.Normalize(color_vals.min(), color_vals.max())
    cmap = cm.viridis
    colors = cmap(norm(color_vals))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Sweep: {args.csv}  ({len(ok)} feasible, {len(bad)} infeasible)")

    def _scatter(ax, x_key, y_key, xlabel, ylabel):
        x = _col(ok, x_key)
        y = _col(ok, y_key)
        sc = ax.scatter(x, y, c=colors, s=60, zorder=3)
        if bad:
            ax.scatter(
                _col(bad, x_key) if x_key not in ("", "infeasible") else [],
                _col(bad, y_key) if y_key not in ("", "infeasible") else [],
                marker="x", c="red", s=60, label="infeasible", zorder=4,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        return sc

    sc = _scatter(
        axes[0],
        "Isp_approx", "peak_T_wall_gas",
        "Isp approx [s]", "Peak T_wall_gas [K]",
    )
    _scatter(
        axes[1],
        "Isp_approx", "coolant_dP",
        "Isp approx [s]", "Coolant dP [Pa]",
    )
    _scatter(
        axes[2],
        "peak_T_wall_gas", "worst_margin_yield",
        "Peak T_wall_gas [K]", "Worst yield margin [-]",
    )

    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        label=args.color_by,
        fraction=0.02,
    )

    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
        out = args.save / "sweep_pareto.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    else:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
