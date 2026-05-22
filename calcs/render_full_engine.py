"""
render_full_engine.py
=====================

Render the complete engine assembly — chamber, nozzle, cooling channels,
injector, LOX dome, fuel manifold, coolant outlet manifold, coolant inlet
manifold — to a single STL file.

Usage
-----
    python calcs/render_full_engine.py [--voxel-mm FLOAT] [--outdir PATH]

Default voxel size: 0.4 mm.  Full render at 0.4 mm takes ~30 s on a
modern laptop.  Use 1.0 mm for a quick preview.

Output
------
output/full_engine.stl  (default)

Blender import hint
-------------------
    File > Import > STL   then   Object > Quick Effects > Shade Smooth
    Dimensions are in mm (written at 1000× scale by write_stl).
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

# Ensure project root is on the path when run directly
_HERE = Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render full engine STL")
    parser.add_argument("--voxel-mm", type=float, default=0.4,
                        help="Voxel size in mm (default 0.4)")
    parser.add_argument("--outdir", type=Path, default=Path("output"),
                        help="Output directory (default: output/)")
    args = parser.parse_args()

    voxel_size = args.voxel_mm * 1e-3

    print("Building reference engine design …")
    t0 = time.perf_counter()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from noyron.rp.engine import Engine, EngineRequirements, METHALOX

    req = EngineRequirements(
        thrust=4448.22,
        chamber_pressure=6.8948e6,
        mixture_ratio=3.60,
        propellant=METHALOX,
        expansion_ratio=10.605,
    )
    engine = Engine(req)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        design = engine.design()

    t_design = time.perf_counter() - t0
    print(f"  design() complete in {t_design:.1f} s")

    # -----------------------------------------------------------------
    # Verification report
    # -----------------------------------------------------------------
    print("\nVerification report:")
    reports = design.verify_all()
    any_fail = False
    for name, report in sorted(reports.items()):
        status = "PASS" if report.passed else "FAIL"
        print(f"  [{status}] {name}")
        for issue in report.issues:
            icon = {"error": "  ERROR", "warning": "  WARN ", "info": "  INFO "}[issue.severity]
            print(f"         {icon}: {issue.message}")
        if not report.passed:
            any_fail = True
    if any_fail:
        print("  *** Some components failed verification — check geometry. ***")

    # -----------------------------------------------------------------
    # Render
    # -----------------------------------------------------------------
    print(f"\nRendering full_engine at {args.voxel_mm:.1f} mm voxels …")
    t_render = time.perf_counter()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = engine.render(
            design,
            voxel_size=voxel_size,
            output_dir=args.outdir,
            components=("full_engine",),
        )

    elapsed = time.perf_counter() - t_render
    stl_path = out["full_engine"]
    size_mb = stl_path.stat().st_size / 1e6
    print(f"  Written: {stl_path}  ({size_mb:.1f} MB)  in {elapsed:.1f} s")

    # -----------------------------------------------------------------
    # Bounding box summary
    # -----------------------------------------------------------------
    nozzle = design.nozzle
    mflds = [
        design.lox_dome,
        design.fuel_manifold,
        design.coolant_outlet_manifold,
        design.coolant_inlet_manifold,
    ]
    z_min = nozzle.x_chamber_start
    z_max = nozzle.x_exit
    r_max = 0.0
    for m in mflds:
        if m is not None:
            lo, hi = m.bbox()
            z_min = min(z_min, float(lo[2]))
            z_max = max(z_max, float(hi[2]))
            r_max = max(r_max, float(hi[0]), float(hi[1]))

    print(f"\nFull engine bounding box:")
    print(f"  z: [{z_min*1e3:.1f}, {z_max*1e3:.1f}] mm  "
          f"(total length {(z_max - z_min)*1e3:.1f} mm)")
    print(f"  r_max: {r_max*1e3:.1f} mm  (envelope OD {2*r_max*1e3:.1f} mm)")
    print(f"\nBlender hint:")
    print(f"  File > Import > STL > {stl_path.name}")
    print("  Object > Quick Effects > Shade Smooth")
    print("  (Mesh units are mm)")


if __name__ == "__main__":
    main()
