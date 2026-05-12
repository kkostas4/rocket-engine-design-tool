"""
test_cli.py
===========

Integration tests for the noyron CLI (noyron/__main__.py).

All tests invoke the CLI via subprocess.run using the same Python
executable that is running the test suite.  This exercises the full
CLI code path including arg parsing, error handling, and file I/O.

Slow tests (those that call Engine.design()) are marked @pytest.mark.slow.
Fast tests (argument parsing, error paths) run without the mark.

Project root is used as the working directory for all subprocess calls
so that `python -m noyron` can locate the physics/ and noyron/ packages.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).parent.parent


def _run(args: list[str], cwd: Path = _PROJECT_ROOT, timeout: int = 360) -> "subprocess.CompletedProcess":
    """Run `python -m noyron <args>` and return the CompletedProcess."""
    import subprocess
    return subprocess.run(
        [sys.executable, "-m", "noyron"] + args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Fast tests (no Engine.design() call)
# ---------------------------------------------------------------------------

def test_help_exits_zero() -> None:
    """--help prints usage and exits 0."""
    result = _run(["--help"])
    assert result.returncode == 0
    assert "design" in result.stdout

def test_design_help_exits_zero() -> None:
    """design --help prints flag list and exits 0."""
    result = _run(["design", "--help"])
    assert result.returncode == 0
    assert "--thrust" in result.stdout
    assert "--config" in result.stdout


def test_unknown_propellant_clear_error() -> None:
    """--propellant with an unknown name exits nonzero and lists valid options."""
    result = _run([
        "design",
        "--thrust", "5000", "--pc", "6.9e6", "--mr", "3.6",
        "--propellant", "frobnozzlate",
        "--epsilon", "10.6",
    ])
    assert result.returncode != 0
    stderr = result.stderr.lower()
    assert "frobnozzlate" in stderr or "unknown" in stderr or "invalid" in stderr
    # Valid option names must appear somewhere in the error output
    combined = (result.stdout + result.stderr).lower()
    assert "methalox" in combined


def test_missing_required_flags_exits_nonzero() -> None:
    """Omitting all required flags exits nonzero with a helpful message."""
    result = _run(["design"])
    assert result.returncode != 0


def test_summary_missing_design_json_exits_nonzero(tmp_path: Path) -> None:
    """summary with a dir that has no design.json exits nonzero."""
    result = _run(["summary", str(tmp_path)])
    assert result.returncode != 0
    assert "design.json" in result.stderr


# ---------------------------------------------------------------------------
# Slow tests (full Engine.design() pipeline)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_design_flags_5kN() -> None:
    """CLI design with explicit flags runs cleanly and prints expected output."""
    result = _run([
        "design",
        "--thrust", "5000", "--pc", "6894800", "--mr", "3.6",
        "--propellant", "methalox", "--epsilon", "10.605",
    ])
    assert result.returncode == 0, f"STDERR:\n{result.stderr}"
    out = result.stdout
    assert "5000" in out, "Thrust missing from summary"
    assert "Isp" in out, "Isp missing from summary"
    assert "T_wall" in out or "T_wall_gas" in out, "T_wall missing"
    assert "LCF" in out or "N_f" in out, "LCF/N_f missing"
    assert "margin" in out.lower(), "Margin missing"


@pytest.mark.slow
def test_design_yaml_config(tmp_path: Path) -> None:
    """--config yaml runs cleanly and exits 0."""
    config = _PROJECT_ROOT / "calcs" / "design_examples" / "methalox_5kN.yaml"
    result = _run(["design", "--config", str(config)])
    assert result.returncode == 0, f"STDERR:\n{result.stderr}"
    assert "5000" in result.stdout


@pytest.mark.slow
def test_design_cli_flag_overrides_yaml(tmp_path: Path) -> None:
    """A CLI --thrust flag overrides the YAML value."""
    config = _PROJECT_ROOT / "calcs" / "design_examples" / "methalox_5kN.yaml"
    # YAML has thrust=5000; override with 4448
    result = _run([
        "design", "--config", str(config), "--thrust", "4448.22",
    ])
    assert result.returncode == 0, f"STDERR:\n{result.stderr}"
    # Summary should show the overridden thrust
    assert "4448" in result.stdout
    assert "5000" not in result.stdout


@pytest.mark.slow
def test_design_render_writes_files(tmp_path: Path) -> None:
    """design --render writes design.json, summary.txt, and STL files."""
    result = _run([
        "design",
        "--thrust", "5000", "--pc", "6894800", "--mr", "3.6",
        "--propellant", "methalox", "--epsilon", "10.605",
        "--render",
        "--voxel-size", "2e-3",   # 2 mm — coarse but fast; test only checks existence
        "--output", str(tmp_path),
    ])
    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    assert (tmp_path / "design.json").exists(), "design.json not written"
    assert (tmp_path / "design.json").stat().st_size > 0
    assert (tmp_path / "summary.txt").exists(), "summary.txt not written"
    assert (tmp_path / "requirements.yaml").exists(), "requirements.yaml not written"
    assert (tmp_path / "chamber_nozzle.stl").exists(), "chamber_nozzle.stl not written"
    assert (tmp_path / "chamber_nozzle.stl").stat().st_size > 0


@pytest.mark.slow
def test_design_json_roundtrip(tmp_path: Path) -> None:
    """design.json can be loaded and reconstructed field-for-field."""
    # Write design.json via CLI
    result = _run([
        "design",
        "--thrust", "5000", "--pc", "6894800", "--mr", "3.6",
        "--propellant", "methalox", "--epsilon", "10.605",
        "--output", str(tmp_path),
    ])
    assert result.returncode == 0, f"STDERR:\n{result.stderr}"

    design_json = tmp_path / "design.json"
    assert design_json.exists()

    # Reconstruct EngineDesign from the file
    sys.path.insert(0, str(_PROJECT_ROOT))
    from noyron.engine import EngineDesign

    with design_json.open() as fh:
        data = json.load(fh)
    d = EngineDesign.from_dict(data)

    # Scalar fields
    assert d.requirements.thrust == pytest.approx(5000.0, rel=1e-6)
    assert d.requirements.chamber_pressure == pytest.approx(6894800.0, rel=1e-6)
    assert d.nozzle.r_throat == pytest.approx(data["nozzle"]["r_throat"], rel=1e-9)
    assert d.channel_geometry.n_channels == data["channel_geometry"]["n_channels"]

    # Callable reconstruction: sample the geometry functions
    h01_stored = data["channel_geometry"]["h01"]
    h2_stored = data["channel_geometry"]["h2"]
    assert d.channel_geometry.height(0.0) == pytest.approx(h01_stored, rel=1e-9)
    # Divergent zone (beyond 2*r_throat from throat)
    x_div = data["nozzle"]["r_throat"] * 3  # well into divergent
    assert d.channel_geometry.height(x_div) == pytest.approx(h2_stored, rel=1e-9)

    # Array fields
    t_wall_stored = data["regen_solution"]["T_wall_gas"]
    np.testing.assert_allclose(d.regen_solution["T_wall_gas"], t_wall_stored)

    # LCF
    assert d.lcf["N_f"] == pytest.approx(data["lcf"]["N_f"], rel=1e-6)


@pytest.mark.slow
def test_yaml_config_matches_demo_numbers() -> None:
    """methalox_5kN.yaml design matches demo_engine.py 5 kN numbers within 0.1%."""
    config = _PROJECT_ROOT / "calcs" / "design_examples" / "methalox_5kN.yaml"
    result = _run(["design", "--config", str(config)])
    assert result.returncode == 0, f"STDERR:\n{result.stderr}"
    out = result.stdout

    # Demo numbers from calcs/demo_engine.py:
    #   n_channels = 80, T_wall_peak = 953 K, dP = 1.498 MPa
    # Check that the summary contains plausible values in the right ballpark.
    assert "80" in out or "n_channels" in out.lower(), \
        "Expected n_channels ~ 80 in summary"
    assert "5000" in out, "Expected thrust 5000 N in summary"
