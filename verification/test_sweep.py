"""
test_sweep.py
=============

Tests for noyron/sweep.py — cartesian_points(), design_one_point(),
and run_sweep().

Fast tests (no Engine.design()) run without marks.
Slow tests (calling Engine.design()) are marked @pytest.mark.slow.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fast: cartesian_points
# ---------------------------------------------------------------------------

def test_cartesian_2x3() -> None:
    """2 values on one axis × 3 on another gives 6 combinations."""
    from noyron.sweep import cartesian_points

    pts = cartesian_points({"thrust": [4000.0, 5000.0], "pc": [5e6, 7e6, 9e6]})
    assert len(pts) == 6

    # Each combination must appear exactly once
    thrusts = [p["thrust"] for p in pts]
    pcs = [p["pc"] for p in pts]
    assert sorted(set(thrusts)) == [4000.0, 5000.0]
    assert sorted(set(pcs)) == [5e6, 7e6, 9e6]


def test_cartesian_empty_gives_one_empty_dict() -> None:
    """No axes → one point with no overrides."""
    from noyron.sweep import cartesian_points

    pts = cartesian_points({})
    assert pts == [{}]


def test_cartesian_single_axis() -> None:
    """Single axis with N values → N points."""
    from noyron.sweep import cartesian_points

    pts = cartesian_points({"mr": [3.3, 3.5, 3.7]})
    assert len(pts) == 3
    assert all("mr" in p for p in pts)


# ---------------------------------------------------------------------------
# Helpers shared by slow tests
# ---------------------------------------------------------------------------

def _base_req():
    from noyron.engine import EngineRequirements, METHALOX
    return EngineRequirements(
        thrust=5000.0,
        chamber_pressure=6894800.0,
        mixture_ratio=3.6,
        propellant=METHALOX,
        expansion_ratio=10.605,
    )


def _read_csv_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Slow: run_sweep integration
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_sweep_single_threaded_runs(tmp_path: Path) -> None:
    """2x2 serial sweep writes results.csv with 4 rows, all status=ok."""
    from noyron.sweep import SweepConfig, run_sweep

    cfg = SweepConfig(
        base=_base_req(),
        variations={"thrust": [4800.0, 5200.0], "mixture_ratio": [3.5, 3.6]},
        output_dir=tmp_path,
        parallel=1,
    )
    csv_path = run_sweep(cfg)

    assert csv_path.exists()
    rows = _read_csv_rows(csv_path)
    assert len(rows) == 4, f"Expected 4 rows, got {len(rows)}"

    for row in rows:
        assert row["status"] == "ok", f"Point {row['point_id']} failed: {row['error_message']}"

    # designs/ subdirectory must exist with 4 point dirs
    designs = tmp_path / "designs"
    point_dirs = sorted(designs.iterdir())
    assert len(point_dirs) == 4

    # Each point dir must have design.json
    for pd in point_dirs:
        assert (pd / "design.json").exists()

    # sweep_config.yaml written
    assert (tmp_path / "sweep_config.yaml").exists()


@pytest.mark.slow
def test_sweep_parallel_2_workers(tmp_path: Path) -> None:
    """2x2 sweep with parallel=2 completes correctly (same result as serial)."""
    from noyron.sweep import SweepConfig, run_sweep

    cfg = SweepConfig(
        base=_base_req(),
        variations={"thrust": [4800.0, 5200.0], "mixture_ratio": [3.5, 3.6]},
        output_dir=tmp_path,
        parallel=2,
    )
    csv_path = run_sweep(cfg)

    rows = _read_csv_rows(csv_path)
    assert len(rows) == 4
    for row in rows:
        assert row["status"] == "ok", f"Point {row['point_id']} failed: {row['error_message']}"

    # point_ids must be 0–3 in sorted order
    ids = [int(r["point_id"]) for r in rows]
    assert ids == sorted(ids)


@pytest.mark.slow
def test_sweep_infeasible_point_does_not_abort(tmp_path: Path) -> None:
    """A single infeasible point (Pc too high) does not abort the whole sweep."""
    from noyron.sweep import SweepConfig, run_sweep

    # Pc = 50 MPa will be infeasible (channel sizing cannot close)
    cfg = SweepConfig(
        base=_base_req(),
        variations={"chamber_pressure": [6894800.0, 50.0e6]},
        output_dir=tmp_path,
        parallel=1,
    )
    csv_path = run_sweep(cfg)

    rows = _read_csv_rows(csv_path)
    assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"

    statuses = {row["status"] for row in rows}
    assert "ok" in statuses, "No feasible point found"
    assert "infeasible" in statuses, "Expected one infeasible point"


@pytest.mark.slow
def test_sweep_resume(tmp_path: Path) -> None:
    """
    Re-running a sweep on the same output dir resumes from existing results.

    After the initial 2x2 run, one design.json is deleted.  The second run
    must recreate only that point and leave the other three untouched.
    """
    from noyron.sweep import SweepConfig, run_sweep

    cfg = SweepConfig(
        base=_base_req(),
        variations={"thrust": [4800.0, 5200.0], "mixture_ratio": [3.5, 3.6]},
        output_dir=tmp_path,
        parallel=1,
    )

    # ── First run ──────────────────────────────────────────────────────────
    run_sweep(cfg)

    designs_dir = tmp_path / "designs"
    point_dirs = sorted(designs_dir.iterdir())
    assert len(point_dirs) == 4

    # Record mtimes of all design.json files
    mtimes = {pd.name: (pd / "design.json").stat().st_mtime for pd in point_dirs}

    # Delete point_001's design.json
    deleted_dir = designs_dir / "point_001"
    (deleted_dir / "design.json").unlink()
    assert not (deleted_dir / "design.json").exists()

    # ── Second run (resume) ────────────────────────────────────────────────
    run_sweep(cfg)

    # The deleted point was recreated
    assert (deleted_dir / "design.json").exists()

    # The other three points were NOT re-run (mtime unchanged)
    for pd in point_dirs:
        if pd.name == "point_001":
            continue
        new_mtime = (pd / "design.json").stat().st_mtime
        assert new_mtime == mtimes[pd.name], (
            f"{pd.name}/design.json was re-run when it should have been skipped"
        )

    # CSV still has 4 rows
    rows = _read_csv_rows(tmp_path / "results.csv")
    assert len(rows) == 4
