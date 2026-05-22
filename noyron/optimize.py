"""
optimize.py
===========

Constrained design optimization for the noyron engine model.

Public API
----------
OptimizationConfig  -- optimization configuration dataclass
design_eval         -- top-level (picklable) penalized objective
run_optimization    -- run DE or Nelder-Mead and write outputs
"""
from __future__ import annotations

import csv
import functools
import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# OptimizationConfig
# ---------------------------------------------------------------------------


@dataclass
class OptimizationConfig:
    """
    Configuration for a constrained design optimization.

    Parameters
    ----------
    base : EngineRequirements
        Fixed requirements. Free variables override specific fields.
    free_vars : dict[str, tuple[float, float]]
        Short YAML key → (lower_bound, upper_bound).
    objective : tuple[str, str]
        (metric_name, 'max' | 'min').
    constraints : list[tuple[str, str, float]]
        Each entry: (metric_name, op, threshold).
        op is '>=', '<=', '>', or '<'.
    max_evals : int
        Approximate maximum objective evaluations.
    parallel : int
        Worker processes for DE (1 = sequential).
    method : str
        'de' (differential evolution) or 'nelder-mead'.
    popsize : int
        DE population size multiplier (individuals = popsize × n_free).
    output_dir : Path
        Root output directory.
    """

    base: Any  # EngineRequirements
    free_vars: dict[str, tuple[float, float]]
    objective: tuple[str, str]
    constraints: list[tuple[str, str, float]] = field(default_factory=list)
    max_evals: int = 300
    parallel: int = 1
    method: str = "de"
    popsize: int = 15
    polish: bool = True
    output_dir: Path = Path("output/opt")


# ---------------------------------------------------------------------------
# _extract_metrics — module-level helper (picklable)
# ---------------------------------------------------------------------------


def _extract_metrics(req: Any, design: Any) -> dict[str, float]:
    """Extract a flat float-valued metrics dict from a completed design."""
    from noyron.rp.thermochem.cea import cea_isp_vac

    rs = design.regen_solution
    struc = design.structural
    lcf = design.lcf

    T_cool = np.asarray(rs["T_coolant"])
    dT_cool = float(abs(T_cool[-1] - T_cool[0]))

    try:
        isp_vac = cea_isp_vac(
            propellant=req.propellant,
            Pc=req.chamber_pressure,
            MR=req.mixture_ratio,
            eps=req.expansion_ratio,
        )
    except Exception:
        isp_vac = 0.0

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
        "Isp_approx": isp_vac,
        "Isp_vac": isp_vac,
        "c_star": float(design.gas.c_star),
        "thrust": float(req.thrust),
        "chamber_pressure": float(req.chamber_pressure),
        "mixture_ratio": float(req.mixture_ratio),
        "expansion_ratio": float(req.expansion_ratio),
        "throat_radius": float(design.nozzle.r_throat),
        "nozzle_bell_length": float(design.nozzle.x_exit),
        "n_channels": float(design.channel_geometry.n_channels),
        "peak_T_wall_gas": float(np.max(np.asarray(rs["T_wall_gas"]))),
        "coolant_dT": dT_cool,
        "coolant_dP": float(rs["dP_total"]),
        "peak_VM_stress_liner": _fmax("sigma_vm_inner"),
        "peak_VM_stress_jacket": _fmax("sigma_vm_jacket"),
        "worst_margin_yield": worst_yield,
        "worst_margin_ultimate": worst_ult,
        "N_f_throat": float(lcf["N_f"]),
        "lcf_margin": float(lcf["margin_lcf"]),
    }


# ---------------------------------------------------------------------------
# _constraint_violation — module-level helper (picklable)
# ---------------------------------------------------------------------------


def _constraint_violation(
    metrics: dict[str, float],
    constraints: list[tuple[str, str, float]],
) -> float:
    """Return total normalised constraint violation (0.0 if all satisfied)."""
    total = 0.0
    for metric_name, op, threshold in constraints:
        val = metrics.get(metric_name, 0.0)
        if op in (">=", ">"):
            viol = max(0.0, threshold - val)
        elif op in ("<=", "<"):
            viol = max(0.0, val - threshold)
        else:
            viol = 0.0
        if threshold != 0.0:
            viol /= abs(threshold)
        total += viol
    return total


# ---------------------------------------------------------------------------
# design_eval — top-level picklable objective
# ---------------------------------------------------------------------------


def design_eval(
    x: Any,
    free_var_names: list[str],
    base: Any,
    objective_metric: str,
    objective_sign: float,
    constraints: list[tuple[str, str, float]],
    penalty_scale: float = 1e6,
) -> float:
    """
    Penalized objective function for scipy optimizers.

    Top-level function so it is picklable by ProcessPoolExecutor (required
    for DE with workers > 1).  Infeasible designs and exceptions return a
    finite penalty — never inf or NaN.

    Parameters
    ----------
    x : array-like
        Decision vector, one entry per free variable.
    free_var_names : list[str]
        Short YAML key for each element of x.
    base : EngineRequirements
        Fixed base requirements.
    objective_metric : str
        Name of the metric to optimise (from the metrics dict).
    objective_sign : float
        +1.0 to minimise, -1.0 to maximise.
    constraints : list[tuple[str, str, float]]
        Each entry: (metric_name, op, threshold).
    penalty_scale : float
        Multiplicative scale for the constraint penalty.

    Returns
    -------
    float
        Penalised objective value.  All infeasible/exception cases return
        a finite positive value >= penalty_scale.
    """
    from noyron.sweep import _apply_overrides
    from noyron.rp.engine import Engine

    overrides = dict(zip(free_var_names, x))
    try:
        req = _apply_overrides(base, overrides)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            design = Engine(req).design()
        metrics = _extract_metrics(req, design)
    except Exception:
        return float(penalty_scale * 2.0)

    viol = _constraint_violation(metrics, constraints)
    obj_raw = metrics.get(objective_metric, 0.0)

    if viol > 0.0:
        return float(penalty_scale * (1.0 + viol))

    return float(objective_sign * obj_raw)


# ---------------------------------------------------------------------------
# _EvalStore — thread-local cache for sequential workers=1 runs
# ---------------------------------------------------------------------------


class _EvalStore:
    """
    Lightweight store that remembers the last N evaluations by x-tuple.

    Used by run_optimization (workers=1 only) so that the DE/NM callback
    can retrieve metrics for xk without re-running Engine.design().
    """

    def __init__(self, maxsize: int = 512) -> None:
        self._data: dict[tuple[float, ...], dict[str, Any]] = {}
        self._order: list[tuple[float, ...]] = []
        self._maxsize = maxsize

    def put(self, x: Any, record: dict[str, Any]) -> None:
        key = tuple(float(v) for v in x)
        if key not in self._data:
            if len(self._order) >= self._maxsize:
                old = self._order.pop(0)
                self._data.pop(old, None)
            self._order.append(key)
        self._data[key] = record

    def get(self, x: Any) -> dict[str, Any] | None:
        key = tuple(float(v) for v in x)
        return self._data.get(key)


# ---------------------------------------------------------------------------
# Internal I/O helpers
# ---------------------------------------------------------------------------


def _append_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    """Append rows to a CSV; write header if the file does not yet exist."""
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _write_yaml_file(data: dict[str, Any], path: Path) -> None:
    try:
        import yaml  # type: ignore[import-untyped]
        with path.open("w") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=False)
    except ModuleNotFoundError:
        with path.open("w") as fh:
            json.dump(data, fh, indent=2)


def _write_outputs(
    config: OptimizationConfig,
    free_var_names: list[str],
    best_x: list[float],
    best_metrics: dict[str, float],
    design: Any,
    req: Any,
    feasible: bool,
) -> None:
    """Write best_design.yaml, best_design.json, best_summary.txt."""
    from noyron.rp.engine import Engine, _PROPELLANT_NAMES

    out = config.output_dir

    prop_name = _PROPELLANT_NAMES.get(id(req.propellant), "custom")
    req_dict: dict[str, Any] = {
        "thrust": req.thrust,
        "pc": req.chamber_pressure,
        "mr": req.mixture_ratio,
        "propellant": prop_name,
        "epsilon": req.expansion_ratio,
        "t_wall_max": req.T_wall_max,
        "dp_coolant": req.dP_coolant_budget,
        "n_cycles": req.n_cycles_required,
        "n_channels": req.n_channels,
        "pump_margin": req.pump_pressure_margin,
    }
    _write_yaml_file(req_dict, out / "best_design.yaml")

    if design is not None:
        with (out / "best_design.json").open("w") as fh:
            json.dump(design.to_dict(), fh, indent=2)
        (out / "best_summary.txt").write_text(
            Engine(req).summary(design), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# run_optimization
# ---------------------------------------------------------------------------


def run_optimization(config: OptimizationConfig) -> dict[str, Any]:
    """
    Run constrained design optimization and write outputs.

    Dispatches to differential evolution (method='de') or Nelder-Mead
    (method='nelder-mead').  Writes iterations.csv incrementally via the
    scipy callback, then writes best_design.yaml / .json / best_summary.txt.

    Parameters
    ----------
    config : OptimizationConfig

    Returns
    -------
    dict
        best_x        -- list[float] best decision vector found
        best_metrics  -- dict[str, float] metrics at best point
        best_obj      -- float raw (un-penalised) objective value
        n_evals       -- int total objective evaluations reported by scipy
        wall_time_s   -- float wall-clock time [s]
        output_dir    -- Path
        feasible      -- bool all constraints satisfied at best_x
    """
    from scipy.optimize import differential_evolution, minimize

    config.output_dir.mkdir(parents=True, exist_ok=True)

    free_var_names = list(config.free_vars.keys())
    bounds = [config.free_vars[k] for k in free_var_names]
    obj_metric, obj_direction = config.objective
    obj_sign = -1.0 if obj_direction.lower() == "max" else 1.0

    # For sequential execution (workers=1) wrap func to populate an eval store
    # so the callback can log metrics without re-running Engine.design().
    store = _EvalStore()
    use_store = config.parallel <= 1

    if use_store:
        _base = config.base
        _fvn = free_var_names
        _constraints = config.constraints

        def _storing_func(x: Any) -> float:
            n_evals_tracker[0] += 1
            from noyron.sweep import _apply_overrides
            from noyron.rp.engine import Engine

            overrides = dict(zip(_fvn, x))
            try:
                req_s = _apply_overrides(_base, overrides)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    design_s = Engine(req_s).design()
                metrics_s = _extract_metrics(req_s, design_s)
                viol_s = _constraint_violation(metrics_s, _constraints)
                obj_s = metrics_s.get(obj_metric, 0.0)
                penalized = (
                    float(obj_sign * obj_s)
                    if viol_s == 0.0
                    else float(1e6 * (1.0 + viol_s))
                )
                store.put(x, {"metrics": metrics_s, "req": req_s, "design": design_s,
                               "feasible": viol_s == 0.0})
            except Exception:
                penalized = float(1e6 * 2.0)
                store.put(x, {"metrics": {}, "req": None, "design": None, "feasible": False})
            return penalized

        eval_func: Any = _storing_func
    else:
        eval_func = functools.partial(
            design_eval,
            free_var_names=free_var_names,
            base=config.base,
            objective_metric=obj_metric,
            objective_sign=obj_sign,
            constraints=config.constraints,
        )

    # Unique constraint metric names for the CSV
    con_cols = list(dict.fromkeys(c[0] for c in config.constraints))
    iter_fields = (
        ["iter", "eval"]
        + free_var_names
        + [obj_metric]
        + con_cols
        + ["feasible"]
    )
    iter_csv = config.output_dir / "iterations.csv"

    best_feasible: list[dict[str, Any]] = []
    n_evals_tracker = [0]

    def _record_iter(i_iter: int, xk: Any) -> None:
        """Append a row to iterations.csv, reading cached metrics when available."""
        rec = store.get(xk) if use_store else None
        if rec is not None:
            metrics_i = rec["metrics"]
            feas_i = rec["feasible"]
            req_i = rec["req"]
            design_i = rec["design"]
        else:
            # Parallel mode or cache miss — re-evaluate (unavoidable)
            n_evals_tracker[0] += 1
            try:
                from noyron.sweep import _apply_overrides
                from noyron.rp.engine import Engine

                req_i = _apply_overrides(config.base, dict(zip(free_var_names, xk)))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    design_i = Engine(req_i).design()
                metrics_i = _extract_metrics(req_i, design_i)
                viol_i = _constraint_violation(metrics_i, config.constraints)
                feas_i = viol_i == 0.0
            except Exception:
                metrics_i = {}
                feas_i = False
                req_i = None
                design_i = None

        if feas_i and req_i is not None:
            best_feasible.append(
                {"x": list(xk), "metrics": metrics_i, "req": req_i, "design": design_i}
            )

        row: dict[str, Any] = {
            "iter": i_iter,
            "eval": n_evals_tracker[0],
            **dict(zip(free_var_names, xk)),
            obj_metric: metrics_i.get(obj_metric, float("nan")),
            **{c: metrics_i.get(c, float("nan")) for c in con_cols},
            "feasible": int(feas_i),
        }
        _append_csv(iter_csv, iter_fields, [row])

    t0 = time.perf_counter()

    _method = config.method.lower().replace("_", "-")
    if _method in ("de", "differential-evolution"):
        _method = "de"

    if _method == "de":
        n_free = max(1, len(free_var_names))
        maxiter = max(1, config.max_evals // (config.popsize * n_free))
        de_iter = [0]

        def _de_cb(*args: Any, **kwargs: Any) -> None:
            de_iter[0] += 1
            # Handle both (xk, convergence) and (intermediate_result,) callback signatures
            first = args[0]
            xk = first.x if hasattr(first, "x") and not isinstance(first, np.ndarray) else first
            _record_iter(de_iter[0], xk)

        result = differential_evolution(
            eval_func,
            bounds=bounds,
            maxiter=maxiter,
            popsize=config.popsize,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.9,
            seed=42,
            workers=config.parallel if config.parallel > 1 else 1,
            callback=_de_cb,
            polish=config.polish,
        )
        best_x = list(result.x)
        if not use_store:
            n_evals_tracker[0] = result.nfev

    elif _method == "nelder-mead":
        x0 = np.array([(lo + hi) / 2.0 for lo, hi in bounds])
        nm_iter = [0]

        def _nm_cb(xk: Any) -> None:
            nm_iter[0] += 1
            _record_iter(nm_iter[0], xk)

        result = minimize(
            eval_func,
            x0,
            method="Nelder-Mead",
            callback=_nm_cb,
            options={"maxfev": config.max_evals, "xatol": 1e-4, "fatol": 1e-4},
        )
        best_x = list(result.x)
        if not use_store:
            n_evals_tracker[0] = result.nfev

    else:
        raise ValueError(
            f"Unknown method '{config.method}'. "
            "Valid: 'de' / 'differential-evolution', 'nelder-mead'."
        )

    wall_time = time.perf_counter() - t0

    # Retrieve or compute final metrics for best_x
    rec_best = store.get(best_x) if use_store else None
    if rec_best is not None and rec_best["metrics"]:
        req_best = rec_best["req"]
        design_best = rec_best["design"]
        best_metrics = rec_best["metrics"]
        feasible = rec_best["feasible"]
    else:
        from noyron.sweep import _apply_overrides
        from noyron.rp.engine import Engine

        overrides = dict(zip(free_var_names, best_x))
        try:
            req_best = _apply_overrides(config.base, overrides)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                design_best = Engine(req_best).design()
            best_metrics = _extract_metrics(req_best, design_best)
            viol_best = _constraint_violation(best_metrics, config.constraints)
            feasible = viol_best == 0.0
        except Exception:
            req_best = config.base
            design_best = None
            best_metrics = {}
            feasible = False

    # Fall back to the best recorded feasible point if the optimum is infeasible
    if not feasible and best_feasible:
        best_rec = max(
            best_feasible,
            key=lambda d: -obj_sign * d["metrics"].get(obj_metric, 0.0),
        )
        best_x = best_rec["x"]
        best_metrics = best_rec["metrics"]
        req_best = best_rec["req"]
        design_best = best_rec["design"]
        feasible = True

    _write_outputs(config, free_var_names, best_x, best_metrics, design_best, req_best, feasible)

    return {
        "best_x": dict(zip(free_var_names, best_x)),
        "best_metrics": best_metrics,
        "best_obj": best_metrics.get(obj_metric, float("nan")),
        "n_evals": n_evals_tracker[0],
        "wall_time_s": wall_time,
        "output_dir": config.output_dir,
        "feasible": feasible,
    }
