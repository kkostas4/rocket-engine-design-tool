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
  design   Build a new engine design from flags or a YAML config file.
  summary  Print the summary for a previously saved design.
  render   Re-render STL files for a previously saved design.

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
                   help="Max gas-wall temperature [K] (default: 1000)")
    p.add_argument("--dp-coolant", type=float, default=None, metavar="Pa",
                   help="Coolant pressure-drop budget [Pa] (default: 3e6)")
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
    "t_wall_max":     ("T_wall_max",          1000.0),
    "dp_coolant":     ("dP_coolant_budget",   3.0e6),
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
        T_wall_max=float(cfg.get("t_wall_max", 1000.0)),
        dP_coolant_budget=float(cfg.get("dp_coolant", 3.0e6)),
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
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
