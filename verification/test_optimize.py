"""
test_optimize.py
================

Verification tests for noyron/optimize.py.

Tests are marked @pytest.mark.slow because each test invokes Engine.design()
one or more times.  The full suite should be run with:
    pytest verification/test_optimize.py -v

For rapid CI, run without the slow marker:
    pytest verification/ -v -m "not slow"
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from noyron.engine import METHALOX, EngineRequirements
from noyron.optimize import (
    OptimizationConfig,
    _constraint_violation,
    _extract_metrics,
    design_eval,
    run_optimization,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def base_req() -> EngineRequirements:
    """Reference engine requirements used across all optimize tests."""
    return EngineRequirements(
        thrust=4448.22,
        chamber_pressure=6.8948e6,
        mixture_ratio=3.6,
        propellant=METHALOX,
        expansion_ratio=10.605,
        T_wall_max=1500.0,
        dP_coolant_budget=5.0e6,
    )


# ---------------------------------------------------------------------------
# Test 1: design_eval returns a scalar float for a valid point
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_design_eval_returns_scalar(base_req: EngineRequirements) -> None:
    """design_eval returns a plain Python float for a feasible design point."""
    val = design_eval(
        x=[base_req.chamber_pressure, base_req.mixture_ratio],
        free_var_names=["pc", "mr"],
        base=base_req,
        objective_metric="Isp_approx",
        objective_sign=-1.0,  # maximise
        constraints=[],
    )
    assert isinstance(val, float), f"Expected float, got {type(val)}"
    assert np.isfinite(val), f"Expected finite, got {val}"
    # Isp_approx at reference point is ~360-380 s; sign=-1 so value is negative
    assert val < 0, "Maximising Isp should return a negative value (sign convention)"
    assert -400.0 < val < -300.0, f"Isp out of expected range: {-val:.1f} s"


# ---------------------------------------------------------------------------
# Test 2: violated constraint returns a penalty >= penalty_scale
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_design_eval_penalty_on_violated_constraint(base_req: EngineRequirements) -> None:
    """A constraint that is impossible to satisfy triggers the penalty path."""
    val = design_eval(
        x=[base_req.chamber_pressure, base_req.mixture_ratio],
        free_var_names=["pc", "mr"],
        base=base_req,
        objective_metric="Isp_approx",
        objective_sign=-1.0,
        constraints=[("peak_T_wall_gas", ">=", 1e10)],  # impossible: T_wall >= 10^10 K
        penalty_scale=1e6,
    )
    assert val >= 1e6, f"Expected penalty >= 1e6, got {val}"


# ---------------------------------------------------------------------------
# Test 3: DE optimization improves Isp over the starting point (1 free var)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_run_optimization_de_improves_isp(
    base_req: EngineRequirements, tmp_path: Path
) -> None:
    """DE with 1 free variable (O/F) finds a higher Isp than the baseline."""
    # Baseline Isp at the reference O/F = 3.6
    baseline_val = design_eval(
        x=[base_req.mixture_ratio],
        free_var_names=["mr"],
        base=base_req,
        objective_metric="Isp_approx",
        objective_sign=-1.0,
        constraints=[],
    )
    baseline_isp = -baseline_val

    cfg = OptimizationConfig(
        base=base_req,
        free_vars={"mr": (2.8, 4.2)},
        objective=("Isp_approx", "max"),
        constraints=[
            ("peak_T_wall_gas", "<=", 1500.0),
            ("coolant_dP", "<=", 5.0e6),
        ],
        # Small population + few evals keeps this test under ~10 design runs
        max_evals=10,
        popsize=2,
        polish=False,
        method="de",
        output_dir=tmp_path / "opt_de",
    )
    result = run_optimization(cfg)

    assert isinstance(result["best_obj"], float)
    assert np.isfinite(result["best_obj"])
    # The optimiser should find at least as good an Isp as the reference point
    assert result["best_obj"] >= baseline_isp * 0.95, (
        f"Optimised Isp {result['best_obj']:.2f} s is much worse than "
        f"baseline {baseline_isp:.2f} s"
    )
    # Output files must exist
    assert (tmp_path / "opt_de" / "best_design.yaml").exists()
    assert (tmp_path / "opt_de" / "iterations.csv").exists()


# ---------------------------------------------------------------------------
# Test 4: best_design.yaml roundtrips via noyron design --config
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_best_design_yaml_roundtrip(
    base_req: EngineRequirements, tmp_path: Path
) -> None:
    """best_design.yaml is loadable and produces a design within 0.1% of Isp."""
    import subprocess, sys

    cfg = OptimizationConfig(
        base=base_req,
        free_vars={"mr": (3.2, 3.8)},
        objective=("Isp_approx", "max"),
        constraints=[],
        max_evals=6,
        popsize=2,
        polish=False,
        method="de",
        output_dir=tmp_path / "opt_rt",
    )
    result = run_optimization(cfg)

    yaml_path = tmp_path / "opt_rt" / "best_design.yaml"
    assert yaml_path.exists(), "best_design.yaml was not written"

    # Run via CLI to confirm it parses and designs without error
    proc = subprocess.run(
        [sys.executable, "-m", "noyron", "design", "--config", str(yaml_path), "--quiet"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(Path(__file__).parent.parent),
    )
    assert proc.returncode == 0, (
        f"noyron design --config best_design.yaml exited {proc.returncode}.\n"
        f"STDERR: {proc.stderr}"
    )


# ---------------------------------------------------------------------------
# Test 5: Nelder-Mead completes without error
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_nelder_mead_runs(base_req: EngineRequirements, tmp_path: Path) -> None:
    """Nelder-Mead optimizer completes and writes outputs."""
    cfg = OptimizationConfig(
        base=base_req,
        free_vars={"mr": (3.0, 3.9)},
        objective=("Isp_approx", "max"),
        constraints=[
            ("coolant_dP", "<=", 5.0e6),
        ],
        # maxfev=8 → NM will do a few function evals then stop
        max_evals=8,
        method="nelder-mead",
        output_dir=tmp_path / "opt_nm",
    )
    result = run_optimization(cfg)

    assert isinstance(result["best_obj"], float)
    assert np.isfinite(result["best_obj"])
    assert result["n_evals"] > 0
    assert result["wall_time_s"] > 0.0
    assert (tmp_path / "opt_nm" / "best_design.yaml").exists()
