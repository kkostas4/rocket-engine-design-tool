"""
test_engine.py
==============

Integration tests for the noyron.engine orchestrator.

Conventions (project-wide):
  - All physics in SI.
  - Slow tests (> a few seconds) are marked @pytest.mark.slow.
  - Run with:  pytest verification/test_engine.py -v
  - Skip slow:  pytest verification/test_engine.py -v -m "not slow"
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from noyron.engine import (
    Engine,
    EngineDesign,
    EngineRequirements,
    METHALOX,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reference_design() -> EngineDesign:
    """Reference 1000-lbf methalox engine design (shared across tests)."""
    req = EngineRequirements(
        thrust=4448.22,
        chamber_pressure=6.8948e6,
        mixture_ratio=3.60,
        propellant=METHALOX,
        expansion_ratio=10.605,
    )
    return Engine(req).design()


# ---------------------------------------------------------------------------
# test_methalox_design_end_to_end
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_methalox_design_end_to_end(reference_design: EngineDesign) -> None:
    """Every field of EngineDesign is populated (not None, not empty dict)."""
    d = reference_design

    assert d.requirements is not None
    assert d.gas is not None
    assert d.nozzle is not None
    assert d.injector is not None
    assert d.channel_geometry is not None
    assert d.bilayer is not None

    # regen_solution must have the expected array keys
    for key in ("x", "r", "T_wall_gas", "T_wall_coolant", "P_coolant", "T_coolant"):
        assert key in d.regen_solution, f"regen_solution missing key '{key}'"
        assert len(d.regen_solution[key]) > 0, f"regen_solution['{key}'] is empty"

    # structural must have margin arrays
    for key in ("sigma_vm_inner", "sigma_vm_jacket", "margin_yield_inner"):
        assert key in d.structural, f"structural missing key '{key}'"

    # lcf must have scalar outputs
    for key in ("N_f", "margin_lcf", "x_worst", "passes"):
        assert key in d.lcf, f"lcf missing key '{key}'"

    # Sanity ranges
    assert d.gas.Tc > 2000.0, "Tc unrealistically low"
    assert d.nozzle.r_throat > 0.005, "throat radius too small"
    assert d.channel_geometry.n_channels >= 40
    assert d.lcf["N_f"] > 0.0


# ---------------------------------------------------------------------------
# test_design_reproducibility
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_design_reproducibility() -> None:
    """Two independent calls with the same requirements return equal designs."""
    req = EngineRequirements(
        thrust=4448.22,
        chamber_pressure=6.8948e6,
        mixture_ratio=3.60,
        propellant=METHALOX,
        expansion_ratio=10.605,
    )
    d1 = Engine(req).design()
    d2 = Engine(req).design()

    assert d1.nozzle.r_throat == pytest.approx(d2.nozzle.r_throat, rel=1e-9)
    assert d1.gas.c_star == pytest.approx(d2.gas.c_star, rel=1e-9)
    assert d1.channel_geometry.n_channels == d2.channel_geometry.n_channels
    assert d1.lcf["N_f"] == pytest.approx(d2.lcf["N_f"], rel=1e-6)


# ---------------------------------------------------------------------------
# test_changing_pc_changes_design
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_changing_pc_changes_design(reference_design: EngineDesign) -> None:
    """Bumping Pc by 10% changes throat radius and peak T_wall."""
    req_hi = EngineRequirements(
        thrust=4448.22,
        chamber_pressure=6.8948e6 * 1.10,   # +10%
        mixture_ratio=3.60,
        propellant=METHALOX,
        expansion_ratio=10.605,
        T_wall_max=1600.0,        # relaxed: higher Pc → higher heat flux (D-B conservative)
        dP_coolant_budget=5.0e6,  # relaxed: allow sizer to find a solution
    )
    d_hi = Engine(req_hi).design()
    d_ref = reference_design

    # Higher Pc → smaller At → smaller throat radius
    assert d_hi.nozzle.r_throat < d_ref.nozzle.r_throat, (
        "Higher Pc should give smaller throat"
    )

    # Higher Pc → higher heat flux → higher peak T_wall
    T_ref = float(d_ref.regen_solution["T_wall_gas"].max())
    T_hi = float(d_hi.regen_solution["T_wall_gas"].max())
    assert T_hi > T_ref, "Higher Pc should raise peak T_wall"


# ---------------------------------------------------------------------------
# test_summary_has_key_numbers
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_summary_has_key_numbers(reference_design: EngineDesign) -> None:
    """summary() output contains thrust, Isp estimate, peak T_wall, and LCF margin."""
    eng = Engine(reference_design.requirements)
    text = eng.summary(reference_design)

    assert "4448" in text, "Summary missing thrust value"
    assert "Isp" in text, "Summary missing Isp label"
    assert "T_wall" in text or "T_wall_gas" in text, "Summary missing T_wall"
    assert "LCF" in text or "N_f" in text, "Summary missing LCF/N_f"
    assert "margin" in text.lower(), "Summary missing margin"


# ---------------------------------------------------------------------------
# test_render_emits_files
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_render_emits_files(tmp_path: Path, reference_design: EngineDesign) -> None:
    """render() at voxel_size=5e-4 produces non-empty STL files."""
    eng = Engine(reference_design.requirements)
    result = eng.render(
        reference_design,
        voxel_size=5e-4,
        output_dir=tmp_path,
        components=("chamber_nozzle", "channels"),
    )

    assert "chamber_nozzle" in result
    assert "channels" in result

    for name, path in result.items():
        assert path.exists(), f"STL file for '{name}' was not written"
        assert path.stat().st_size > 0, f"STL file for '{name}' is empty"


# ---------------------------------------------------------------------------
# test_design_under_60s
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_design_under_60s() -> None:
    """Engine.design() completes in under 60 seconds."""
    import time

    req = EngineRequirements(
        thrust=4448.22,
        chamber_pressure=6.8948e6,
        mixture_ratio=3.60,
        propellant=METHALOX,
        expansion_ratio=10.605,
    )
    t0 = time.perf_counter()
    Engine(req).design()
    elapsed = time.perf_counter() - t0
    assert elapsed < 60.0, f"design() took {elapsed:.1f} s (limit 60 s)"
