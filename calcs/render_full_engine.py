"""
render_full_engine.py
=====================

Render the complete engine assembly — chamber, nozzle, cooling channels,
injector, LOX dome, fuel manifold, coolant outlet manifold, coolant inlet
manifold — to a single STL file.

Usage
-----
    python calcs/render_full_engine.py [--voxel-mm FLOAT] [--outdir PATH]
                                       [--no-blend]

Default voxel size: 0.4 mm.  Full render at 0.4 mm takes ~30 s on a
modern laptop.  Use 1.0 mm for a quick preview.

--no-blend produces a sharp-boolean assembly (BlendConfig all-zeros) saved
as full_engine_sharp.stl alongside the smooth version for visual comparison.

Output
------
output/full_engine.stl        (default — smooth fillets)
output/full_engine_sharp.stl  (with --no-blend)

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
    parser.add_argument("--no-blend", action="store_true",
                        help="Disable smooth fillets; all boolean ops are sharp "
                             "(saves as full_engine_sharp.stl)")
    args = parser.parse_args()

    voxel_size = args.voxel_mm * 1e-3

    print("Building reference engine design …")
    t0 = time.perf_counter()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from noyron.rp.engine import Engine, EngineRequirements, METHALOX
        from noyron.rp.geometry.blend_config import BlendConfig

    blend_cfg = (
        BlendConfig(
            boss_to_manifold=0.0,
            manifold_to_chamber=0.0,
            manifold_to_jacket=0.0,
            chamber_to_nozzle=0.0,
            pintle_to_face_plate=0.0,
            default=0.0,
        )
        if args.no_blend
        else BlendConfig()
    )

    req = EngineRequirements(
        thrust=4448.22,
        chamber_pressure=6.8948e6,
        mixture_ratio=3.60,
        propellant=METHALOX,
        expansion_ratio=10.605,
        blend_config=blend_cfg,
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
    label = "sharp" if args.no_blend else "smooth"
    stl_key = "full_engine_sharp" if args.no_blend else "full_engine"
    print(f"\nRendering full_engine ({label}) at {args.voxel_mm:.1f} mm voxels …")
    t_render = time.perf_counter()

    # Render into a staging directory so --no-blend never clobbers full_engine.stl.
    import tempfile, shutil
    with tempfile.TemporaryDirectory() as _tmp:
        _tmp_path = Path(_tmp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = engine.render(
                design,
                voxel_size=voxel_size,
                output_dir=_tmp_path,
                components=("full_engine",),
            )
        _src = out["full_engine"]
        _dest_name = "full_engine_sharp.stl" if args.no_blend else "full_engine.stl"
        stl_path = args.outdir / _dest_name
        args.outdir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(_src), stl_path)

    elapsed = time.perf_counter() - t_render
    size_mb = stl_path.stat().st_size / 1e6
    try:
        import trimesh  # type: ignore[import-untyped]
        mesh = trimesh.load(stl_path, force="mesh")
        verts = len(mesh.vertices)
        faces = len(mesh.faces)
        watertight = mesh.is_watertight
        mesh_info = f"  verts={verts:,}  faces={faces:,}  watertight={watertight}"
    except Exception:
        mesh_info = "  (trimesh not available — skipping mesh stats)"
    print(f"  Written: {stl_path}  ({size_mb:.1f} MB)  in {elapsed:.1f} s")
    print(mesh_info)

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
