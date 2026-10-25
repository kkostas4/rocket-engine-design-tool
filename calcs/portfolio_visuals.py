"""
portfolio_visuals.py
====================

Generate three portfolio PNGs from the real methalox_5kN design:

  1. physics_profiles.png    — 4-panel physics profile along the engine
  2. design_space_sweep.png  — Pc x MR grid of peak wall temperatures
  3. channel_taper.png       — cooling-channel width & height vs axial position

All values come from the solver — nothing is fabricated.
Output folder: portfolio_visuals/
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import yaml

from noyron import Engine, EngineRequirements, METHALOX


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_DIR = ROOT / "portfolio_visuals"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = ROOT / "calcs" / "design_examples" / "methalox_5kN.yaml"

mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 200,
    "font.size": 10,
})


def _load_requirements(pc: float | None = None,
                       mr: float | None = None) -> EngineRequirements:
    """Build EngineRequirements from methalox_5kN.yaml with optional Pc/MR override."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return EngineRequirements(
        thrust=float(cfg["thrust"]),
        chamber_pressure=float(pc if pc is not None else cfg["pc"]),
        mixture_ratio=float(mr if mr is not None else cfg["mr"]),
        propellant=METHALOX,
        expansion_ratio=float(cfg["epsilon"]),
        T_wall_max=float(cfg["t_wall_max"]),
        dP_coolant_budget=float(cfg["dp_coolant"]),
        n_cycles_required=int(cfg["n_cycles"]),
        pump_pressure_margin=float(cfg["pump_margin"]),
        L_star=float(cfg["l_star"]) if cfg.get("l_star") is not None else None,
        Ac_At=float(cfg.get("ac_at", 4.0)),
    )


# ---------------------------------------------------------------------------
# Graph 1 — physics profiles
# ---------------------------------------------------------------------------

def graph1_physics_profiles(design) -> Path:
    s = design.regen_solution
    x_m = np.asarray(s["x"])
    x_mm = x_m * 1e3

    r_mm = np.array([design.nozzle.r(float(xi)) for xi in x_m]) * 1e3
    q_MW = np.asarray(s["q_dot"]) / 1e6
    hg_kW = np.asarray(s["h_g"]) / 1e3
    T_wall = np.asarray(s["T_wall_gas"])
    T_cool = np.asarray(s["T_coolant"])

    fig, ax = plt.subplots(
        4, 1, figsize=(9.5, 11), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1, 1, 1.2]},
    )

    # Panel 1: engine profile (fill under r(z))
    ax[0].fill_between(x_mm, r_mm, 0.0, color="#5b7fb9", alpha=0.35,
                        edgecolor="#1f3d6b", linewidth=1.6)
    ax[0].plot(x_mm, r_mm, color="#1f3d6b", linewidth=1.8)
    ax[0].set_ylabel("nozzle radius\nr(z)  [mm]")
    ax[0].set_ylim(0, r_mm.max() * 1.15)

    # Panel 2: heat flux
    ax[1].plot(x_mm, q_MW, color="#c0392b", linewidth=1.8)
    ax[1].fill_between(x_mm, q_MW, 0.0, color="#c0392b", alpha=0.10)
    ax[1].set_ylabel("heat flux q''\n[MW/m²]")

    # Panel 3: gas-side h_g
    ax[2].plot(x_mm, hg_kW, color="#e67e22", linewidth=1.8)
    ax[2].fill_between(x_mm, hg_kW, 0.0, color="#e67e22", alpha=0.10)
    ax[2].set_ylabel("gas-side h_g\n[kW/m²·K]")

    # Panel 4: wall and coolant temperature
    ax[3].plot(x_mm, T_wall, color="#c0392b", linewidth=1.8, label="gas-side wall")
    ax[3].plot(x_mm, T_cool, color="#2874a6", linewidth=1.8, label="coolant")
    ax[3].fill_between(x_mm, T_cool, T_wall, color="gray", alpha=0.08)
    ax[3].set_ylabel("temperature [K]")
    ax[3].set_xlabel("axial position z  [mm]   (throat at 0)")
    ax[3].legend(loc="upper right", frameon=False)

    for a in ax:
        a.axvline(0.0, color="black", linestyle="--", linewidth=0.9, alpha=0.6)
        a.grid(True, alpha=0.25)
        a.margins(x=0.005)

    fig.suptitle("Regenerative cooling — the physics along the engine",
                  fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    out = OUT_DIR / "physics_profiles.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Graph 2 — design-space sweep
# ---------------------------------------------------------------------------

def graph2_design_space_sweep() -> Path:
    Pc_grid = np.array([5e6, 7e6, 9e6, 11e6])
    MR_grid = np.array([3.0, 3.3, 3.6, 3.9])

    T_peak = np.full((len(Pc_grid), len(MR_grid)), np.nan)
    feasible = np.zeros_like(T_peak, dtype=bool)
    reasons = np.empty_like(T_peak, dtype=object)

    T_wall_max_limit = _load_requirements().T_wall_max  # 1500 K

    for i, Pc in enumerate(Pc_grid):
        for j, MR in enumerate(MR_grid):
            t0 = time.time()
            try:
                req = _load_requirements(pc=Pc, mr=MR)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    d = Engine(req).design()
                peak = float(np.asarray(d.regen_solution["T_wall_gas"]).max())
                T_peak[i, j] = peak
                feasible[i, j] = peak <= T_wall_max_limit
                reasons[i, j] = "ok" if feasible[i, j] else f"T={peak:.0f}>lim"
            except Exception as exc:
                reasons[i, j] = type(exc).__name__
                feasible[i, j] = False
            dt = time.time() - t0
            print(f"  Pc={Pc/1e6:5.1f} MPa  MR={MR:.2f}  "
                  f"T_peak={T_peak[i, j]!s:>8}  "
                  f"feasible={feasible[i, j]}  ({dt:.1f}s)  {reasons[i, j]}")

    fig, ax = plt.subplots(figsize=(8.5, 7.0))

    # Colour by peak wall temperature; grey out infeasible cells (nan or > limit).
    T_display = np.where(feasible, T_peak, np.nan)
    cmap = plt.get_cmap("viridis")
    cmap.set_bad(color="#dddddd")

    vals = T_display[np.isfinite(T_display)]
    if vals.size > 0:
        vmin, vmax = float(vals.min()), float(vals.max())
    else:
        vmin, vmax = 500.0, T_wall_max_limit

    im = ax.imshow(T_display, origin="lower", cmap=cmap,
                    vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(MR_grid)))
    ax.set_xticklabels([f"{mr:.2f}" for mr in MR_grid])
    ax.set_yticks(range(len(Pc_grid)))
    ax.set_yticklabels([f"{pc/1e6:.1f}" for pc in Pc_grid])
    ax.set_xlabel("mixture ratio  O/F")
    ax.set_ylabel("chamber pressure  Pc  [MPa]")

    for i in range(len(Pc_grid)):
        for j in range(len(MR_grid)):
            if feasible[i, j]:
                ax.text(j, i, f"{T_peak[i, j]:.0f} K",
                         ha="center", va="center",
                         color="white", fontsize=10, fontweight="bold")
            else:
                # Grey cell with an X plus reason
                ax.text(j, i, "×", ha="center", va="center",
                         color="#666666", fontsize=28, fontweight="bold")
                label = ("no soln" if T_peak[i, j] != T_peak[i, j]
                          else f"{T_peak[i, j]:.0f} K")
                ax.text(j, i - 0.32, label,
                         ha="center", va="center",
                         color="#444444", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.set_label("peak gas-side wall temperature  [K]")

    ax.set_title("Design space explored automatically — one tool, sixteen engines",
                  fontsize=13)
    fig.tight_layout()
    out = OUT_DIR / "design_space_sweep.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Graph 3 — channel taper
# ---------------------------------------------------------------------------

def graph3_channel_taper(design) -> Path:
    nozzle = design.nozzle
    ch = design.channel_geometry
    x_m = np.linspace(nozzle.x_inlet, nozzle.x_exit, 400)
    x_mm = x_m * 1e3

    w_mm = np.array([ch.width(float(xi))  for xi in x_m]) * 1e3
    h_mm = np.array([ch.height(float(xi)) for xi in x_m]) * 1e3

    # Zone-2 (divergent) boundary: x = 2 r_throat
    x2_mm = 2.0 * nozzle.r_throat * 1e3

    fig, ax = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)

    ax[0].plot(x_mm, w_mm, color="#1f77b4", linewidth=2.0)
    ax[0].fill_between(x_mm, w_mm, 0.0, color="#1f77b4", alpha=0.20)
    ax[0].set_ylabel("channel width  w  [mm]")
    ax[0].set_ylim(0, max(w_mm.max(), h_mm.max()) * 1.15)

    ax[1].plot(x_mm, h_mm, color="#e67e22", linewidth=2.0)
    ax[1].fill_between(x_mm, h_mm, 0.0, color="#e67e22", alpha=0.20)
    ax[1].set_ylabel("channel height  h  [mm]")
    ax[1].set_xlabel("axial position z  [mm]   (throat at 0)")
    ax[1].set_ylim(0, h_mm.max() * 1.15)

    for a in ax:
        a.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6,
                   label="throat")
        a.axvline(x2_mm, color="#888888", linestyle=":", linewidth=1.0,
                   label="throat→divergent zone")
        a.grid(True, alpha=0.25)
        a.margins(x=0.005)

    ax[0].legend(loc="upper right", frameon=False)

    n_ch = ch.n_channels
    fig.suptitle(
        f"Optimizer's answer to the heat flux — channels taper "
        f"(n = {n_ch})",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))

    out = OUT_DIR / "channel_taper.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("[1/3] Building methalox_5kN baseline design ...")
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        baseline = Engine(_load_requirements()).design()
    print(f"       done in {time.time() - t0:.1f}s")

    print("[1/3] Graph 1: physics_profiles.png ...")
    p1 = graph1_physics_profiles(baseline)
    print(f"       saved -> {p1}")

    print("[2/3] Graph 2: design_space_sweep.png (16 designs) ...")
    p2 = graph2_design_space_sweep()
    print(f"       saved -> {p2}")

    print("[3/3] Graph 3: channel_taper.png ...")
    p3 = graph3_channel_taper(baseline)
    print(f"       saved -> {p3}")

    print("\nAll three PNGs written to portfolio_visuals/:")
    for p in (p1, p2, p3):
        print(f"  {p}")


if __name__ == "__main__":
    main()
