"""
__main__.py
===========

CLI entry point for the noyron engine orchestrator.

Usage
-----
    python -m noyron <subcommand> [options]

    # Or, after `pip install -e .`:
    noyron <subcommand> [options]

Subcommands
-----------
  design    Build a new engine design from flags or a YAML config file.
  summary   Print the summary for a previously saved design.
  render    Re-render STL files for a previously saved design.
  optimize  Constrained design optimization (DE or Nelder-Mead).

Run `python -m noyron <subcommand> --help` for per-subcommand usage.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Propellant name mapping (shared with engine module)
# ---------------------------------------------------------------------------

def _propellant_map() -> dict[str, Any]:
    """Return {name: PropellantPair} without triggering heavy imports at parse time."""
    from noyron.engine import _PROPELLANT_PRESETS
    return _PROPELLANT_PRESETS


_VALID_PROPELLANTS = ("methalox", "kerolox", "hydrolox")


# ---------------------------------------------------------------------------
# Argument parsers
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="noyron",
        description="Computational rocket engine design tool.",
    )
    sub = parser.add_subparsers(dest="command", metavar="SUBCOMMAND")
    sub.required = True

    # ------------------------------------------------------------------ design
    p = sub.add_parser(
        "design",
        help="Run the full design pipeline and optionally render STLs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Build an EngineRequirements from flags (or --config yaml), "
            "run Engine.design(), and print the summary.\n\n"
            "Required flags (unless --config provides them):\n"
            "  --thrust, --pc, --mr, --propellant, --epsilon\n\n"
            "Example:\n"
            "  python -m noyron design --thrust 5e3 --pc 6.9e6 \\\n"
            "      --mr 3.6 --propellant methalox --epsilon 10.6\n\n"
            "  python -m noyron design --config calcs/design_examples/methalox_5kN.yaml"
        ),
    )
    p.add_argument("--thrust", type=float, metavar="N",
                   help="Thrust [N] (required unless in --config)")
    p.add_argument("--pc", type=float, metavar="Pa",
                   help="Chamber pressure [Pa] (required unless in --config)")
    p.add_argument("--mr", type=float, metavar="[-]",
                   help="Oxidiser/fuel mass ratio (required unless in --config)")
    p.add_argument("--propellant", type=str, metavar="NAME",
                   choices=_VALID_PROPELLANTS,
                   help=f"Propellant pair: {', '.join(_VALID_PROPELLANTS)} "
                        "(required unless in --config)")
    p.add_argument("--epsilon", type=float, metavar="[-]",
                   help="Expansion ratio Ae/At (required unless in --config)")
    p.add_argument("--t-wall-max", type=float, default=None, metavar="K",
                   help="Max gas-wall temperature [K] (default: 1500)")
    p.add_argument("--dp-coolant", type=float, default=None, metavar="Pa",
                   help="Coolant pressure-drop budget [Pa] (default: 5e6)")
    p.add_argument("--n-cycles", type=float, default=None, metavar="N",
                   help="Required LCF life [cycles] (default: 100)")
    p.add_argument("--n-channels", type=int, default=None, metavar="N",
                   help="Channel count hint (sizer sweeps internally; default: 76)")
    p.add_argument("--pump-margin", type=float, default=None, metavar="Pa",
                   help="Feed-system pump head margin above Pc+dP_inj [Pa] (default: 2e6)")
    p.add_argument("--config", type=Path, default=None, metavar="YAML",
                   help="YAML config file.  CLI flags override file values.")
    p.add_argument("--render", action="store_true",
                   help="Write STLs, design.json, summary.txt, and requirements.yaml.")
    p.add_argument("--voxel-size", type=float, default=5e-4, metavar="M",
                   help="Voxel grid spacing for STL render [m] (default: 5e-4 = 0.5 mm)")
    p.add_argument("--output", type=Path, default=None, metavar="DIR",
                   help="Output directory for --render (default: output/<timestamp>/)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress the design summary.")
    p.add_argument("--verbose", action="store_true",
                   help="Show full Python tracebacks on errors.")

    # ----------------------------------------------------------------- summary
    p2 = sub.add_parser(
        "summary",
        help="Print summary for a previously saved design.",
        description=(
            "Read <design_dir>/design.json and print the engine summary.\n\n"
            "Example:\n"
            "  python -m noyron summary output/2026-01-01_120000/"
        ),
    )
    p2.add_argument("design_dir", type=Path, metavar="DESIGN_DIR",
                    help="Directory containing design.json (written by design --render).")

    # ------------------------------------------------------------------ render
    p3 = sub.add_parser(
        "render",
        help="Re-render STL files for an existing design.",
        description=(
            "Read <design_dir>/design.json and write STL files to that directory.\n\n"
            "Example:\n"
            "  python -m noyron render output/2026-01-01_120000/ --voxel-size 1e-4"
        ),
    )
    p3.add_argument("design_dir", type=Path, metavar="DESIGN_DIR",
                    help="Directory containing design.json.")
    p3.add_argument("--voxel-size", type=float, default=5e-4, metavar="M",
                    help="Voxel grid spacing [m] (default: 5e-4 = 0.5 mm)")

    # ------------------------------------------------------------------ sweep
    p4 = sub.add_parser(
        "sweep",
        help="Run a Cartesian-product parameter sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Run Engine.design() over all Cartesian combinations of the --vary axes.\n\n"
            "Base requirements from --config yaml (design or sweep format) or CLI flags.\n"
            "Each --vary FIELD=v1,v2,... adds one sweep axis.\n\n"
            "Example (CLI):\n"
            "  python -m noyron sweep \\\n"
            "      --config calcs/design_examples/methalox_5kN.yaml \\\n"
            "      --vary pc=5e6,7e6,9e6 --vary mr=3.3,3.6 \\\n"
            "      --parallel 4 --output output/pc_mr_sweep/\n\n"
            "Example (sweep YAML):\n"
            "  python -m noyron sweep \\\n"
            "      --config calcs/sweeps/pc_mr_methalox.yaml\n"
        ),
    )
    p4.add_argument("--config", type=Path, default=None, metavar="YAML",
                    help="Design or sweep YAML.  A sweep YAML has 'base' and "
                         "'variations' keys; a design YAML is used as the base.")
    p4.add_argument("--vary", action="append", default=None, metavar="FIELD=v1,v2,...",
                    help="Sweep axis: FIELD=v1,v2,...  Repeatable.  "
                         "FIELD is a short name (pc, mr, epsilon, t_wall_max, ...).")
    p4.add_argument("--parallel", type=int, default=1, metavar="N",
                    help="Number of worker processes (default: 1).")
    p4.add_argument("--render", action="store_true",
                    help="Render chamber_nozzle STL for each feasible point.")
    p4.add_argument("--output", type=Path, default=None, metavar="DIR",
                    help="Output directory (default: output/sweep_<timestamp>/).")
    # Base design parameters (optional; loaded from --config if present)
    p4.add_argument("--thrust", type=float, default=None, metavar="N")
    p4.add_argument("--pc", type=float, default=None, metavar="Pa")
    p4.add_argument("--mr", type=float, default=None, metavar="[-]")
    p4.add_argument("--propellant", type=str, default=None, metavar="NAME",
                    choices=_VALID_PROPELLANTS)
    p4.add_argument("--epsilon", type=float, default=None, metavar="[-]")
    p4.add_argument("--t-wall-max", type=float, default=None, metavar="K")
    p4.add_argument("--dp-coolant", type=float, default=None, metavar="Pa")
    p4.add_argument("--n-cycles", type=float, default=None, metavar="N")
    p4.add_argument("--pump-margin", type=float, default=None, metavar="Pa")
    p4.add_argument("--n-channels", type=int, default=None, metavar="N")
    p4.add_argument("--verbose", action="store_true",
                    help="Show full Python tracebacks on errors.")

    # --------------------------------------------------------------- optimize
    p5 = sub.add_parser(
        "optimize",
        help="Constrained design optimization (DE or Nelder-Mead).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Optimize engine design parameters subject to thermal, coolant,\n"
            "structural, and LCF constraints.\n\n"
            "Example:\n"
            "  python -m noyron optimize \\\n"
            "      --config calcs/opt_examples/methalox_max_isp.yaml\n"
        ),
    )
    p5.add_argument("--config", type=Path, required=True, metavar="YAML",
                    help="Optimization YAML config (required).")
    p5.add_argument("--output", type=Path, default=None, metavar="DIR",
                    help="Output directory override.")
    p5.add_argument("--parallel", type=int, default=None, metavar="N",
                    help="Worker processes override.")
    p5.add_argument("--max-evals", type=int, default=None, metavar="N",
                    dest="max_evals",
                    help="Max objective evaluations override.")
    p5.add_argument("--method", type=str, default=None,
                    choices=["de", "nelder-mead"],
                    help="Optimizer: 'de' (default) or 'nelder-mead'.")
    p5.add_argument("--verbose", action="store_true",
                    help="Show full Python tracebacks on errors.")

    return parser


# ---------------------------------------------------------------------------
# YAML / config helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config file.  Falls back to JSON if pyyaml is absent."""
    try:
        import yaml  # type: ignore[import-untyped]
        with path.open("r") as fh:
            return yaml.safe_load(fh) or {}
    except ModuleNotFoundError:
        # pyyaml not installed — try plain JSON (same extension, different content)
        with path.open("r") as fh:
            return json.load(fh)


def _write_yaml(data: dict[str, Any], path: Path) -> None:
    """Write a YAML file, falling back to JSON if pyyaml is absent."""
    try:
        import yaml  # type: ignore[import-untyped]
        with path.open("w") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=False)
    except ModuleNotFoundError:
        with path.open("w") as fh:
            json.dump(data, fh, indent=2)


# ---------------------------------------------------------------------------
# Requirements builder
# ---------------------------------------------------------------------------

# Mapping from YAML config key → (EngineRequirements field, default)
_CONFIG_FIELDS: dict[str, tuple[str, Any]] = {
    "thrust":         ("thrust",              None),
    "pc":             ("chamber_pressure",    None),
    "mr":             ("mixture_ratio",       None),
    "propellant":     ("propellant",          None),
    "epsilon":        ("expansion_ratio",     None),
    "t_wall_max":     ("T_wall_max",          1500.0),
    "dp_coolant":     ("dP_coolant_budget",   5.0e6),
    "n_cycles":       ("n_cycles_required",   100.0),
    "n_channels":     ("n_channels",          76),
    "pump_margin":    ("pump_pressure_margin", 2.0e6),
}


def _build_requirements(args: argparse.Namespace, verbose: bool = False):
    """
    Build EngineRequirements from CLI args + optional YAML config.

    CLI flags override YAML values.  All five required fields (thrust, pc,
    mr, propellant, epsilon) must be present after merging.

    Returns
    -------
    EngineRequirements

    Raises
    ------
    SystemExit
        If a required field is missing or the propellant name is unknown.
    """
    from noyron.engine import EngineRequirements, _PROPELLANT_PRESETS

    # Start with file values (if any)
    cfg: dict[str, Any] = {}
    if args.config is not None:
        if not args.config.exists():
            _die(f"Config file not found: {args.config}", verbose)
        cfg = _load_yaml(args.config)

    # Override with CLI flags where provided
    cli_map = {
        "thrust":      args.thrust,
        "pc":          args.pc,
        "mr":          args.mr,
        "propellant":  args.propellant,
        "epsilon":     args.epsilon,
        "t_wall_max":  args.t_wall_max,
        "dp_coolant":  args.dp_coolant,
        "n_cycles":    args.n_cycles,
        "n_channels":  args.n_channels,
        "pump_margin": args.pump_margin,
    }
    for k, v in cli_map.items():
        if v is not None:
            cfg[k] = v

    # Validate required fields
    required = ("thrust", "pc", "mr", "propellant", "epsilon")
    missing = [k for k in required if cfg.get(k) is None]
    if missing:
        _die(
            f"Missing required design parameters: {', '.join('--' + k for k in missing)}\n"
            f"Provide them as CLI flags or in a --config YAML file.",
            verbose,
        )

    # Resolve propellant
    prop_name = str(cfg["propellant"]).lower()
    if prop_name not in _PROPELLANT_PRESETS:
        _die(
            f"Unknown propellant '{prop_name}'. "
            f"Valid options: {', '.join(sorted(_PROPELLANT_PRESETS))}",
            verbose,
        )
    propellant = _PROPELLANT_PRESETS[prop_name]

    return EngineRequirements(
        thrust=float(cfg["thrust"]),
        chamber_pressure=float(cfg["pc"]),
        mixture_ratio=float(cfg["mr"]),
        propellant=propellant,
        expansion_ratio=float(cfg["epsilon"]),
        n_cycles_required=float(cfg.get("n_cycles", 100.0)),
        T_wall_max=float(cfg.get("t_wall_max", 1500.0)),
        dP_coolant_budget=float(cfg.get("dp_coolant", 5.0e6)),
        pump_pressure_margin=float(cfg.get("pump_margin", 2.0e6)),
        n_channels=int(cfg.get("n_channels", 76)),
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_outputs(
    design,
    output_dir: Path,
    voxel_size: float,
    render: bool,
    verbose: bool,
) -> None:
    """Write design.json, summary.txt, requirements.yaml, and optionally STLs."""
    from noyron.engine import Engine

    output_dir.mkdir(parents=True, exist_ok=True)

    # design.json
    design_json = output_dir / "design.json"
    with design_json.open("w") as fh:
        json.dump(design.to_dict(), fh, indent=2)

    # summary.txt
    summary_txt = output_dir / "summary.txt"
    eng = Engine(design.requirements)
    summary_text = eng.summary(design)
    summary_txt.write_text(summary_text, encoding="utf-8")

    # requirements.yaml  (round-trippable as --config input)
    req = design.requirements
    from noyron.engine import _PROPELLANT_NAMES
    prop_name = _PROPELLANT_NAMES.get(id(req.propellant), "custom")
    req_yaml: dict[str, Any] = {
        "thrust":      req.thrust,
        "pc":          req.chamber_pressure,
        "mr":          req.mixture_ratio,
        "propellant":  prop_name,
        "epsilon":     req.expansion_ratio,
        "t_wall_max":  req.T_wall_max,
        "dp_coolant":  req.dP_coolant_budget,
        "n_cycles":    req.n_cycles_required,
        "n_channels":  req.n_channels,
        "pump_margin": req.pump_pressure_margin,
    }
    _write_yaml(req_yaml, output_dir / "requirements.yaml")

    # STLs
    if render:
        print(f"Rendering STLs to {output_dir} (voxel_size={voxel_size:.2e} m)...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stl_paths = eng.render(
                design,
                voxel_size=voxel_size,
                output_dir=output_dir,
                components=("chamber_nozzle", "channels"),
            )
        for name, path in stl_paths.items():
            size_kb = path.stat().st_size / 1024
            print(f"  {name}: {path.name}  ({size_kb:.0f} kB)")


# ---------------------------------------------------------------------------
# Error helper
# ---------------------------------------------------------------------------

def _die(msg: str, verbose: bool = False, exc: BaseException | None = None) -> None:
    """Print an error to stderr and exit 1."""
    print(f"ERROR: {msg}", file=sys.stderr)
    if verbose and exc is not None:
        traceback.print_exc()
    sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------

def cmd_design(args: argparse.Namespace) -> None:
    from noyron.engine import Engine
    from physics.util.exceptions import SizingInfeasibleError

    verbose = args.verbose

    req = _build_requirements(args, verbose)

    print(f"Designing: thrust={req.thrust:.0f} N, "
          f"Pc={req.chamber_pressure/1e6:.3f} MPa, "
          f"O/F={req.mixture_ratio:.3f}, "
          f"epsilon={req.expansion_ratio:.3f}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            design = Engine(req).design()
    except SizingInfeasibleError as exc:
        _die(
            f"Channel sizing failed.\n\n{exc}\n\n"
            "Suggestions: increase --dp-coolant, increase --t-wall-max,\n"
            "or reduce --pc.",
            verbose, exc,
        )
    except Exception as exc:
        _die(f"Design failed: {exc}", verbose, exc)

    eng = Engine(req)
    if not args.quiet:
        print(eng.summary(design))

    if args.render:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_dir = args.output or (Path("output") / ts)
        _write_outputs(design, out_dir, args.voxel_size, render=True, verbose=verbose)
        print(f"\nOutputs written to: {out_dir}")
    elif args.output is not None:
        # --output without --render: just write design.json + summary + yaml
        _write_outputs(design, args.output, args.voxel_size, render=False, verbose=verbose)
        print(f"\nOutputs written to: {args.output}")


def cmd_summary(args: argparse.Namespace) -> None:
    from noyron.engine import Engine, EngineDesign

    design_json = args.design_dir / "design.json"
    if not design_json.exists():
        _die(f"design.json not found in {args.design_dir}")

    with design_json.open("r") as fh:
        data = json.load(fh)

    try:
        design = EngineDesign.from_dict(data)
    except Exception as exc:
        _die(f"Failed to load design: {exc}")

    print(Engine(design.requirements).summary(design))


def cmd_sweep(args: argparse.Namespace) -> None:
    from noyron.sweep import SweepConfig, run_sweep

    verbose = args.verbose

    # ── Resolve base requirements ──────────────────────────────────────────
    cfg_file_variations: dict[str, list] = {}
    cfg_file_parallel: int = 1
    cfg_file_render: bool = False
    base_cfg: dict[str, Any] = {}

    if args.config is not None:
        if not args.config.exists():
            _die(f"Config file not found: {args.config}", verbose)
        raw = _load_yaml(args.config)
        if "base" in raw:
            # Full sweep YAML: extract base section + variations
            base_cfg = raw["base"]
            cfg_file_variations = raw.get("variations", {})
            cfg_file_parallel = int(raw.get("parallel", 1))
            cfg_file_render = bool(raw.get("render", False))
        else:
            # Plain design YAML
            base_cfg = raw

    # CLI flags override base_cfg values
    cli_overrides: dict[str, Any] = {
        "thrust":      args.thrust,
        "pc":          args.pc,
        "mr":          args.mr,
        "propellant":  args.propellant,
        "epsilon":     args.epsilon,
        "t_wall_max":  args.t_wall_max,
        "dp_coolant":  args.dp_coolant,
        "n_cycles":    args.n_cycles,
        "n_channels":  args.n_channels,
        "pump_margin": args.pump_margin,
    }
    for k, v in cli_overrides.items():
        if v is not None:
            base_cfg[k] = v

    # Validate required fields for the base
    required = ("thrust", "pc", "mr", "propellant", "epsilon")
    missing = [k for k in required if base_cfg.get(k) is None]
    if missing:
        _die(
            f"Missing required base parameters: {', '.join('--' + k for k in missing)}\n"
            f"Provide them via --config or as CLI flags.",
            verbose,
        )

    from noyron.engine import EngineRequirements, _PROPELLANT_PRESETS

    prop_name = str(base_cfg["propellant"]).lower()
    if prop_name not in _PROPELLANT_PRESETS:
        _die(
            f"Unknown propellant '{prop_name}'. "
            f"Valid: {', '.join(sorted(_PROPELLANT_PRESETS))}",
            verbose,
        )

    base_req = EngineRequirements(
        thrust=float(base_cfg["thrust"]),
        chamber_pressure=float(base_cfg["pc"]),
        mixture_ratio=float(base_cfg["mr"]),
        propellant=_PROPELLANT_PRESETS[prop_name],
        expansion_ratio=float(base_cfg["epsilon"]),
        n_cycles_required=float(base_cfg.get("n_cycles", 100.0)),
        T_wall_max=float(base_cfg.get("t_wall_max", 1500.0)),
        dP_coolant_budget=float(base_cfg.get("dp_coolant", 5.0e6)),
        pump_pressure_margin=float(base_cfg.get("pump_margin", 2.0e6)),
        n_channels=int(base_cfg.get("n_channels", 76)),
    )

    # ── Resolve sweep variations ────────────────────────────────────────────
    variations: dict[str, list] = dict(cfg_file_variations)
    for vary_arg in (args.vary or []):
        field, _, vals_str = vary_arg.partition("=")
        field = field.strip()
        raw_vals = [v.strip() for v in vals_str.split(",")]
        parsed_vals: list[Any] = []
        for v in raw_vals:
            try:
                parsed_vals.append(float(v))
            except ValueError:
                parsed_vals.append(v)  # propellant name or other string
        variations[field] = parsed_vals

    if not variations:
        _die(
            "No sweep axes defined. Use --vary FIELD=v1,v2,... or provide "
            "a sweep YAML with a 'variations' section.",
            verbose,
        )

    parallel = max(1, args.parallel or cfg_file_parallel)
    render = args.render or cfg_file_render
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = args.output or (Path("output") / f"sweep_{ts}")

    n_points = 1
    for vals in variations.values():
        n_points *= len(vals)
    print(
        f"Starting sweep: {n_points} point(s) across "
        f"{len(variations)} axis/axes, {parallel} worker(s)"
    )
    print(f"Output: {out_dir}")

    cfg = SweepConfig(
        base=base_req,
        variations=variations,
        output_dir=out_dir,
        parallel=parallel,
        render=render,
    )
    run_sweep(cfg)


def cmd_optimize(args: argparse.Namespace) -> None:
    from noyron.optimize import OptimizationConfig, run_optimization

    verbose = args.verbose

    if not args.config.exists():
        _die(f"Config file not found: {args.config}", verbose)

    raw = _load_yaml(args.config)

    # ── Base requirements ──────────────────────────────────────────────────
    base_cfg = raw.get("base", {})
    if not base_cfg:
        _die("Config must have a 'base:' section.", verbose)

    required = ("thrust", "pc", "mr", "propellant", "epsilon")
    missing = [k for k in required if base_cfg.get(k) is None]
    if missing:
        _die(
            f"Missing required base parameters: {', '.join(missing)}\n"
            "Provide them in the 'base:' section of the config YAML.",
            verbose,
        )

    from noyron.engine import EngineRequirements, _PROPELLANT_PRESETS

    prop_name = str(base_cfg["propellant"]).lower()
    if prop_name not in _PROPELLANT_PRESETS:
        _die(
            f"Unknown propellant '{prop_name}'. "
            f"Valid: {', '.join(sorted(_PROPELLANT_PRESETS))}",
            verbose,
        )

    base_req = EngineRequirements(
        thrust=float(base_cfg["thrust"]),
        chamber_pressure=float(base_cfg["pc"]),
        mixture_ratio=float(base_cfg["mr"]),
        propellant=_PROPELLANT_PRESETS[prop_name],
        expansion_ratio=float(base_cfg["epsilon"]),
        n_cycles_required=float(base_cfg.get("n_cycles", 100.0)),
        T_wall_max=float(base_cfg.get("t_wall_max", 1500.0)),
        dP_coolant_budget=float(base_cfg.get("dp_coolant", 5.0e6)),
        pump_pressure_margin=float(base_cfg.get("pump_margin", 2.0e6)),
        n_channels=int(base_cfg.get("n_channels", 76)),
    )

    # ── Free variables ─────────────────────────────────────────────────────
    free_vars_raw = raw.get("free_vars", {})
    if not free_vars_raw:
        _die("Config must have a non-empty 'free_vars:' section.", verbose)

    free_vars: dict[str, tuple[float, float]] = {}
    for k, v in free_vars_raw.items():
        if not (isinstance(v, (list, tuple)) and len(v) == 2):
            _die(
                f"free_vars['{k}'] must be [lower, upper], got: {v}",
                verbose,
            )
        free_vars[k] = (float(v[0]), float(v[1]))

    # ── Objective ──────────────────────────────────────────────────────────
    obj_raw = raw.get("objective", {})
    obj_metric = str(obj_raw.get("metric", "Isp_approx"))
    obj_direction = str(obj_raw.get("direction", "max"))
    if obj_direction not in ("max", "min"):
        _die(f"objective.direction must be 'max' or 'min', got '{obj_direction}'.", verbose)

    # ── Constraints ────────────────────────────────────────────────────────
    constraints: list[tuple[str, str, float]] = []
    for item in raw.get("constraints", []):
        if not (isinstance(item, (list, tuple)) and len(item) == 3):
            _die(
                f"Bad constraint format: {item!r}. "
                "Expected [metric_name, op, value].",
                verbose,
            )
        constraints.append((str(item[0]), str(item[1]), float(item[2])))

    # ── Overrides from CLI ─────────────────────────────────────────────────
    max_evals = args.max_evals or int(raw.get("max_evals", 300))
    parallel = args.parallel or int(raw.get("parallel", 1))
    method = args.method or str(raw.get("method", "de"))
    popsize = int(raw.get("popsize", 15))

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    default_out = Path("output") / f"opt_{ts}"
    out_dir = args.output or Path(str(raw.get("output_dir", default_out)))

    cfg = OptimizationConfig(
        base=base_req,
        free_vars=free_vars,
        objective=(obj_metric, obj_direction),
        constraints=constraints,
        max_evals=max_evals,
        parallel=parallel,
        method=method,
        popsize=popsize,
        output_dir=out_dir,
    )

    print(
        f"Optimising: {len(free_vars)} free variable(s), "
        f"method={method}, max_evals={max_evals}"
    )
    print(f"Objective : {obj_direction}({obj_metric})")
    for k, (lo, hi) in free_vars.items():
        print(f"  {k}: [{lo:.4g}, {hi:.4g}]")
    print(f"Output    : {out_dir}")

    try:
        result = run_optimization(cfg)
    except Exception as exc:
        _die(f"Optimisation failed: {exc}", verbose, exc)

    print("\nOptimisation complete.")
    print(f"  Feasible          : {result['feasible']}")
    print(f"  Best {obj_metric:<20}: {result['best_obj']:.4f}")
    for k, v in result["best_x"].items():
        print(f"  {k:<26}: {v:.6g}")
    print(f"  Evaluations       : {result['n_evals']}")
    print(f"  Wall time         : {result['wall_time_s']:.1f} s")
    print(f"  Output dir        : {result['output_dir']}")
    if result["feasible"]:
        print(f"\nRoundtrip check:")
        print(f"  python -m noyron design --config {result['output_dir']}/best_design.yaml")


def cmd_render(args: argparse.Namespace) -> None:
    from noyron.engine import Engine, EngineDesign
    from physics.util.exceptions import SizingInfeasibleError

    design_json = args.design_dir / "design.json"
    if not design_json.exists():
        _die(f"design.json not found in {args.design_dir}")

    with design_json.open("r") as fh:
        data = json.load(fh)

    try:
        design = EngineDesign.from_dict(data)
    except Exception as exc:
        _die(f"Failed to load design: {exc}")

    out_dir = args.design_dir
    print(f"Rendering STLs to {out_dir} (voxel_size={args.voxel_size:.2e} m)...")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stl_paths = Engine(design.requirements).render(
                design,
                voxel_size=args.voxel_size,
                output_dir=out_dir,
                components=("chamber_nozzle", "channels"),
            )
    except Exception as exc:
        _die(f"Render failed: {exc}")

    for name, path in stl_paths.items():
        size_kb = path.stat().st_size / 1024
        print(f"  {name}: {path.name}  ({size_kb:.0f} kB)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "design":
        cmd_design(args)
    elif args.command == "summary":
        cmd_summary(args)
    elif args.command == "render":
        cmd_render(args)
    elif args.command == "sweep":
        cmd_sweep(args)
    elif args.command == "optimize":
        cmd_optimize(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
