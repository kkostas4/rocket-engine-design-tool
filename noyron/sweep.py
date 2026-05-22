"""
sweep.py
========

Cartesian-product parameter sweep for Engine.design().

Public API
----------
SweepConfig       -- sweep configuration dataclass
cartesian_points  -- enumerate all Cartesian combinations
design_one_point  -- single-point worker (top-level, picklable)
run_sweep         -- run or resume a sweep, write results.csv
"""
from __future__ import annotations

import csv
import dataclasses
import itertools
import json
import math
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_G0 = 9.80665  # standard gravity [m/s²]


# ---------------------------------------------------------------------------
# SweepConfig
# ---------------------------------------------------------------------------

@dataclass
class SweepConfig:
    """
    Configuration for a Cartesian-product parameter sweep.

    Parameters
    ----------
    base : EngineRequirements
        Base requirements shared by all design points.
    variations : dict[str, list[Any]]
        Short YAML keys (pc, mr, epsilon, t_wall_max, …) mapped to lists
        of values.  The sweep covers all Cartesian combinations.
    output_dir : Path
        Root output directory.  designs/point_NNN/ and results.csv are
        written here.
    parallel : int
        Worker processes.  1 = sequential.
    render : bool
        Render chamber_nozzle STL for each feasible point.
    """
    base: Any          # EngineRequirements — typed as Any to defer import
    variations: dict[str, list[Any]]
    output_dir: Path
    parallel: int = 1
    render: bool = False


# ---------------------------------------------------------------------------
# cartesian_points
# ---------------------------------------------------------------------------

def cartesian_points(variations: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Return all Cartesian-product combinations as a list of override dicts."""
    if not variations:
        return [{}]
    keys = list(variations.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*variations.values())]


# ---------------------------------------------------------------------------
# _apply_overrides
# ---------------------------------------------------------------------------

# Maps CLI/YAML short keys to EngineRequirements field names.
_KEY_MAP: dict[str, str] = {
    "thrust":                "thrust",
    "pc":                    "chamber_pressure",
    "chamber_pressure":      "chamber_pressure",
    "mr":                    "mixture_ratio",
    "mixture_ratio":         "mixture_ratio",
    "epsilon":               "expansion_ratio",
    "expansion_ratio":       "expansion_ratio",
    "t_wall_max":            "T_wall_max",
    "T_wall_max":            "T_wall_max",
    "dp_coolant":            "dP_coolant_budget",
    "dP_coolant_budget":     "dP_coolant_budget",
    "n_cycles":              "n_cycles_required",
    "n_cycles_required":     "n_cycles_required",
    "pump_margin":           "pump_pressure_margin",
    "pump_pressure_margin":  "pump_pressure_margin",
    "n_channels":            "n_channels",
    "propellant":            "propellant",
    "am_process":            "am_rules",
}


def _apply_overrides(base: Any, overrides: dict[str, Any]) -> Any:
    """Return a new EngineRequirements with the given overrides applied."""
    from noyron.rp.engine import EngineRequirements, _PROPELLANT_PRESETS

    kwargs: dict[str, Any] = {
        f.name: getattr(base, f.name) for f in dataclasses.fields(base)
    }

    for short_key, val in overrides.items():
        field = _KEY_MAP.get(short_key)
        if field is None:
            raise ValueError(
                f"Unknown sweep field '{short_key}'.  "
                f"Valid keys: {sorted(_KEY_MAP)}"
            )
        if field == "propellant":
            prop_name = str(val).lower()
            if prop_name not in _PROPELLANT_PRESETS:
                raise ValueError(f"Unknown propellant '{prop_name}'")
            val = _PROPELLANT_PRESETS[prop_name]
        elif field == "am_rules":
            from noyron.kernel.manufacturing import AM_RULES_PRESETS
            key = str(val).lower()
            if key not in AM_RULES_PRESETS:
                raise ValueError(f"Unknown am_process '{key}'")
            val = AM_RULES_PRESETS[key]
        elif field == "n_channels":
            val = int(val)
        else:
            val = float(val)
        kwargs[field] = val

    return EngineRequirements(**kwargs)


# ---------------------------------------------------------------------------
# CSV layout
# ---------------------------------------------------------------------------

_CSV_COLS = [
    "point_id", "status", "error_message",
    "thrust", "chamber_pressure", "mixture_ratio", "expansion_ratio", "propellant",
    "throat_radius", "nozzle_bell_length", "n_channels",
    "peak_T_wall_gas", "coolant_dT", "coolant_dP",
    "peak_VM_stress_liner", "peak_VM_stress_jacket",
    "worst_margin_yield", "worst_margin_ultimate",
    "N_f_throat", "lcf_margin",
    "Isp_approx", "c_star",
    "design_time_seconds",
]


def _propellant_name_str(pair: Any) -> str:
    from noyron.rp.engine import _PROPELLANT_PRESETS
    for name, preset in _PROPELLANT_PRESETS.items():
        if preset == pair:
            return name
    return "custom"


def _success_row(point_id: int, req: Any, design: Any, dt: float) -> dict[str, Any]:
    import numpy as np
    from noyron.rp.thermochem.cea import cea_isp_vac

    rs = design.regen_solution
    struc = design.structural
    lcf = design.lcf

    T_cool = np.asarray(rs["T_coolant"])
    dT_cool = float(abs(T_cool[-1] - T_cool[0]))

    try:
        Isp_approx = cea_isp_vac(
            propellant=req.propellant,
            Pc=req.chamber_pressure,
            MR=req.mixture_ratio,
            eps=req.expansion_ratio,
        )
    except Exception:
        Isp_approx = 0.0

    def _fmin(key: str, default: float = 9999.0) -> float:
        arr = np.asarray(struc.get(key, [default]))
        return float(arr.min())

    def _fmax(key: str) -> float:
        arr = np.asarray(struc.get(key, [0.0]))
        return float(arr.max())

    worst_yield = min(
        _fmin("margin_yield_inner"),
        _fmin("margin_yield_jacket"),
        _fmin("margin_yield_rib"),
    )
    worst_ult = min(
        _fmin("margin_ultimate_inner"),
        _fmin("margin_ultimate_jacket"),
    )

    return {
        "point_id":              point_id,
        "status":                "ok",
        "error_message":         "",
        "thrust":                req.thrust,
        "chamber_pressure":      req.chamber_pressure,
        "mixture_ratio":         req.mixture_ratio,
        "expansion_ratio":       req.expansion_ratio,
        "propellant":            _propellant_name_str(req.propellant),
        "throat_radius":         float(design.nozzle.r_throat),
        "nozzle_bell_length":    float(design.nozzle.x_exit),
        "n_channels":            int(design.channel_geometry.n_channels),
        "peak_T_wall_gas":       float(np.max(np.asarray(rs["T_wall_gas"]))),
        "coolant_dT":            dT_cool,
        "coolant_dP":            float(rs["dP_total"]),
        "peak_VM_stress_liner":  _fmax("sigma_vm_inner"),
        "peak_VM_stress_jacket": _fmax("sigma_vm_jacket"),
        "worst_margin_yield":    worst_yield,
        "worst_margin_ultimate": worst_ult,
        "N_f_throat":            float(lcf["N_f"]),
        "lcf_margin":            float(lcf["margin_lcf"]),
        "Isp_approx":            Isp_approx,
        "c_star":                float(design.gas.c_star),
        "design_time_seconds":   dt,
    }


def _error_row(point_id: int, req: Any, msg: str, dt: float) -> dict[str, Any]:
    empty: dict[str, Any] = {col: "" for col in _CSV_COLS}
    empty.update({
        "point_id":          point_id,
        "status":            "infeasible",
        "error_message":     msg,
        "thrust":            getattr(req, "thrust", ""),
        "chamber_pressure":  getattr(req, "chamber_pressure", ""),
        "mixture_ratio":     getattr(req, "mixture_ratio", ""),
        "expansion_ratio":   getattr(req, "expansion_ratio", ""),
        "propellant":        _propellant_name_str(req.propellant)
                             if hasattr(req, "propellant") else "",
        "design_time_seconds": dt,
    })
    return empty


# ---------------------------------------------------------------------------
# design_one_point — top-level picklable worker
# ---------------------------------------------------------------------------

def design_one_point(
    base: Any,
    overrides: dict[str, Any],
    point_dir: Path,
    render: bool,
    point_id: int,
) -> dict[str, Any]:
    """
    Design one sweep point and write outputs to point_dir.

    Top-level function so it is picklable by ProcessPoolExecutor.

    Returns
    -------
    dict  One CSV row (status='ok' or 'infeasible').
    """
    t0 = time.perf_counter()

    try:
        req = _apply_overrides(base, overrides)
    except ValueError as exc:
        return _error_row(
            point_id, base, f"Override error: {exc}", time.perf_counter() - t0
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from noyron.rp.engine import Engine
            design = Engine(req).design()
    except Exception as exc:
        return _error_row(point_id, req, str(exc), time.perf_counter() - t0)

    dt = time.perf_counter() - t0

    try:
        point_dir.mkdir(parents=True, exist_ok=True)

        with (point_dir / "design.json").open("w") as fh:
            json.dump(design.to_dict(), fh, indent=2)

        from noyron.rp.engine import Engine as _Eng
        (point_dir / "summary.txt").write_text(
            _Eng(req).summary(design), encoding="utf-8"
        )

        req_dict: dict[str, Any] = {
            "thrust":      req.thrust,
            "pc":          req.chamber_pressure,
            "mr":          req.mixture_ratio,
            "propellant":  _propellant_name_str(req.propellant),
            "epsilon":     req.expansion_ratio,
            "t_wall_max":  req.T_wall_max,
            "dp_coolant":  req.dP_coolant_budget,
            "n_cycles":    req.n_cycles_required,
            "pump_margin": req.pump_pressure_margin,
        }
        try:
            import yaml
            with (point_dir / "requirements.yaml").open("w") as fh:
                yaml.dump(req_dict, fh, default_flow_style=False, sort_keys=False)
        except ModuleNotFoundError:
            with (point_dir / "requirements.yaml").open("w") as fh:
                json.dump(req_dict, fh, indent=2)

        if render:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _Eng(req).render(
                    design,
                    voxel_size=5e-4,
                    output_dir=point_dir,
                    components=("chamber_nozzle",),
                )
    except Exception:
        pass

    return _success_row(point_id, req, design, dt)


# ---------------------------------------------------------------------------
# _row_from_existing
# ---------------------------------------------------------------------------

def _row_from_existing(
    point_id: int,
    point_dir: Path,
    base: Any,
    overrides: dict[str, Any],
) -> dict[str, Any] | None:
    """Return a result row if design.json already exists and is loadable."""
    design_json = point_dir / "design.json"
    if not design_json.exists():
        return None
    try:
        with design_json.open() as fh:
            data = json.load(fh)
        from noyron.rp.engine import EngineDesign
        design = EngineDesign.from_dict(data)
        req = _apply_overrides(base, overrides)
        return _success_row(point_id, req, design, 0.0)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# run_sweep
# ---------------------------------------------------------------------------

def run_sweep(config: SweepConfig) -> Path:
    """
    Run (or resume) a Cartesian-product parameter sweep.

    Points whose design.json already exists are loaded from disk without
    re-running Engine.design() (idempotent resume).

    Returns
    -------
    Path  Path to results.csv.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    designs_dir = config.output_dir / "designs"
    results_csv = config.output_dir / "results.csv"

    points = cartesian_points(config.variations)
    _write_sweep_config(config)

    resumed: list[dict[str, Any]] = []
    pending: list[tuple[int, dict[str, Any], Path]] = []

    for i, overrides in enumerate(points):
        point_dir = designs_dir / f"point_{i:03d}"
        row = _row_from_existing(i, point_dir, config.base, overrides)
        if row is not None:
            resumed.append(row)
        else:
            pending.append((i, overrides, point_dir))

    print(
        f"Sweep: {len(points)} total, {len(resumed)} resumed, {len(pending)} to run."
    )

    new_rows: list[dict[str, Any]] = []

    if config.parallel > 1 and pending:
        with ProcessPoolExecutor(max_workers=config.parallel) as executor:
            futures = {
                executor.submit(
                    design_one_point,
                    config.base, overrides, point_dir, config.render, pid,
                ): pid
                for pid, overrides, point_dir in pending
            }
            for future in as_completed(futures):
                row = future.result()
                new_rows.append(row)
                print(f"  point {row['point_id']:03d}: {row['status']}")
    else:
        for pid, overrides, point_dir in pending:
            row = design_one_point(
                config.base, overrides, point_dir, config.render, pid,
            )
            new_rows.append(row)
            print(f"  point {row['point_id']:03d}: {row['status']}")

    all_rows = resumed + new_rows
    all_rows.sort(key=lambda r: int(r["point_id"]))

    with results_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    n_ok = sum(1 for r in all_rows if r["status"] == "ok")
    n_fail = len(all_rows) - n_ok
    print(f"\nSweep complete: {n_ok}/{len(points)} feasible, {n_fail} infeasible.")
    print(f"Results: {results_csv}")
    return results_csv


def _write_sweep_config(config: SweepConfig) -> None:
    """Write sweep_config.yaml to the output directory."""
    req = config.base
    data: dict[str, Any] = {
        "base": {
            "thrust":      req.thrust,
            "pc":          req.chamber_pressure,
            "mr":          req.mixture_ratio,
            "propellant":  _propellant_name_str(req.propellant),
            "epsilon":     req.expansion_ratio,
            "t_wall_max":  req.T_wall_max,
            "dp_coolant":  req.dP_coolant_budget,
            "n_cycles":    req.n_cycles_required,
            "pump_margin": req.pump_pressure_margin,
        },
        "variations": {k: list(v) for k, v in config.variations.items()},
        "parallel": config.parallel,
        "render": config.render,
    }
    out_path = config.output_dir / "sweep_config.yaml"
    try:
        import yaml
        with out_path.open("w") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=False)
    except ModuleNotFoundError:
        with out_path.open("w") as fh:
            json.dump(data, fh, indent=2)
