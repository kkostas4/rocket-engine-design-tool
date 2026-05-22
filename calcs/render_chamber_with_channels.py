"""
render_chamber_with_channels.py
================================

Generate the outer jacket STL for the reference engine chamber and nozzle
with axial cooling channel voids carved in.

Output
------
  geometry/output/chamber_nozzle_channels.stl   (coordinates in mm)

Run with:
    python calcs/render_chamber_with_channels.py

Expected runtime
----------------
  Channel sizer   ~90 s
  Mesh extraction ~5-30 min at 0.1 mm voxels (chunked)
  Decimation      ~10-30 s

Memory budget at 0.1 mm voxels, 10 mm slabs
--------------------------------------------
  Outer diameter at chamber: ~55 mm -> radial bbox half-width ~30 mm
  Slab: 620 × 620 × 100 ≈ 38M cells × 4 bytes = 154 MB float32.  Fine.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import trimesh  # type: ignore[import-untyped]

from noyron.rp.geometry.cooling_channels import chamber_with_channels_sdf
from noyron.kernel.geometry.mesh.chunked_extractor import chunked_sdf_to_mesh
from noyron.kernel.geometry.mesh.extractor import adaptive_bbox
from noyron.kernel.geometry.mesh.stl_export import is_watertight, write_stl
from noyron.rp.nozzle.rao import RaoNozzle
from noyron.rp.regen.coolant import GRCOP42, CoolantInlet
from noyron.rp.regen.bartz import CombustionGas
from noyron.rp.regen.sizing import ChannelSizingProblem, size_channels
from noyron.rp.structural.bilayer import (
    BilayerWall,
    RibGeometry,
    WallLayer,
    size_jacket_thickness,
)
from noyron.rp.structural.materials import GRCOP84_PROXY, INCONEL718_STRUCT

# ---------------------------------------------------------------------------
# Reference engine (frozen baseline)
# ---------------------------------------------------------------------------
GAS = CombustionGas(
    Tc=3561.2,
    Pc=6.8948e6,
    gamma=1.15,
    MW=21.9,
    cp=2000.0,
    mu=8.0e-5,
    Pr=0.60,
    c_star=1893.3,
)
NOZZLE = RaoNozzle(r_throat=0.022313 / 2.0, epsilon=10.605)
COOLANT = CoolantInlet(T=150.0, P=12.0e6, m_dot=0.3096, fluid="Methane")

# ---------------------------------------------------------------------------
# Channel sizing
# ---------------------------------------------------------------------------
print("Running channel sizer (~60-90 s)...")
_problem = ChannelSizingProblem(
    nozzle=NOZZLE,
    gas=GAS,
    coolant_inlet=COOLANT,
    wall_material=GRCOP42,
    T_wall_max=1000.0,
    T_wall_margin=50.0,
    dP_max=3.0e6,
    min_wall_thickness=0.5e-3,
    max_aspect_ratio=8.0,
    min_channel_width=0.6e-3,
    t_land_min=0.3e-3,
    n_stations_opt=60,
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    geom = size_channels(_problem)

# ---------------------------------------------------------------------------
# Jacket thickness
# ---------------------------------------------------------------------------
_r_exit = NOZZLE.r_throat * NOZZLE.epsilon**0.5
_r_jacket_exit = _r_exit + 0.5e-3 + geom.height(NOZZLE.x_exit)
t_jacket = max(
    1.5e-3,
    round(
        size_jacket_thickness(
            P_coolant_max=12.0e6,
            r_jacket=_r_jacket_exit,
            material=INCONEL718_STRUCT,
            T_op=200.0,
            FoS=1.4,
        )
        / 0.5e-3
    )
    * 0.5e-3,
)
print(f"  Jacket thickness: {t_jacket*1e3:.1f} mm")

# ---------------------------------------------------------------------------
# Bilayer wall
# ---------------------------------------------------------------------------
bilayer = BilayerWall(
    inner_liner=WallLayer(
        thickness=geom.wall_thickness,
        material=GRCOP84_PROXY,
    ),
    outer_jacket=WallLayer(
        thickness=lambda x: t_jacket,
        material=INCONEL718_STRUCT,
    ),
    ribs=RibGeometry(
        n_ribs=geom.n_channels,
        land_width=geom.land_thickness,
        height=geom.height,
        material=GRCOP84_PROXY,
    ),
    channel_geometry=geom,
)

# ---------------------------------------------------------------------------
# Build SDF with channels
# ---------------------------------------------------------------------------
print("Building chamber+nozzle SDF with channels...")
t0 = time.perf_counter()
sdf, profile = chamber_with_channels_sdf(
    rao_nozzle=NOZZLE,
    bilayer=bilayer,
    L_chamber=0.125,
)
print(f"  SDF built in {time.perf_counter()-t0:.1f} s")
print(f"  Axial domain  : z = {profile['z_min']*1e3:.1f} mm "
      f"to {profile['z_max']*1e3:.1f} mm")
print(f"  Max OD        : {profile['r_outer_max']*2e3:.1f} mm")
print(f"  n_channels    : {profile['n_channels']}")

# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------
VOXEL_SIZE = 1e-4   # 0.1 mm — SDF intermediates are float32 so peak ~1.1 GB

bbox = adaptive_bbox(
    nozzle_x_min=profile["z_min"],
    nozzle_x_max=profile["z_max"],
    r_outer_max=profile["r_outer_max"],
    radial_margin=5e-3,
    axial_margin=5e-3,
)
x_min, x_max, y_min, y_max, z_min_b, z_max_b = bbox
print(f"\nBounding box: {(x_max-x_min)*1e3:.0f}×{(y_max-y_min)*1e3:.0f}×"
      f"{(z_max_b-z_min_b)*1e3:.0f} mm  ({VOXEL_SIZE*1e3:.2f} mm voxels)")

# ---------------------------------------------------------------------------
# Chunked mesh extraction
# ---------------------------------------------------------------------------
print("\nExtracting mesh (chunked marching cubes)...")
t0 = time.perf_counter()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    verts, faces = chunked_sdf_to_mesh(
        sdf,
        bbox,
        voxel_size=VOXEL_SIZE,
        slab_axial_steps=25,   # 25 → 2.5 mm slabs; ~94 MB per float32 array, ~1.1 GB peak
        verbose=True,
    )
print(f"\n  Extraction time: {time.perf_counter()-t0:.1f} s")
print(f"  Pre-decimation: {len(verts):,} verts / {len(faces):,} faces")

# ---------------------------------------------------------------------------
# Decimation to ~500 K faces
# ---------------------------------------------------------------------------
TARGET_FACES = 500_000
if len(faces) > TARGET_FACES:
    print(f"\nDecimating to ~{TARGET_FACES//1000}K faces ...")
    t0 = time.perf_counter()
    try:
        import fast_simplification  # noqa: F401 — checked for availability only
        mesh_full = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh_dec = mesh_full.simplify_quadric_decimation(face_count=TARGET_FACES)
        verts = np.array(mesh_dec.vertices, dtype=np.float64)
        faces = np.array(mesh_dec.faces, dtype=np.int32)
        print(f"  Post-decimation: {len(verts):,} verts / {len(faces):,} faces "
              f"({time.perf_counter()-t0:.1f} s)")
    except ImportError:
        print(f"  fast_simplification not installed — skipping decimation.")
        print(f"  Install with: pip install fast-simplification")
        print(f"  Exporting full {len(faces):,}-face mesh.")
else:
    print("  No decimation needed.")

# ---------------------------------------------------------------------------
# STL export
# ---------------------------------------------------------------------------
os.makedirs("geometry/output", exist_ok=True)
out_path = "geometry/output/chamber_nozzle_channels.stl"

write_stl(verts, faces, out_path, scale_to_mm=True)
file_kb = os.path.getsize(out_path) / 1024
print(f"\nSTL written: {out_path}  ({file_kb:.0f} kB, coordinates in mm)")

# ---------------------------------------------------------------------------
# Manifold check
# ---------------------------------------------------------------------------
WATERTIGHT_FACE_LIMIT = 5_000_000  # edge-adjacency matrix OOMs above ~5M faces
if len(faces) > WATERTIGHT_FACE_LIMIT:
    print(f"Watertight check skipped ({len(faces):,} faces > {WATERTIGHT_FACE_LIMIT//1_000_000}M limit).")
    print("  Install fast-simplification, decimate, then re-check.")
else:
    watertight = is_watertight(verts, faces)
    print(f"Watertight (manifold): {watertight}")
    if not watertight:
        print("  NOTE: non-manifold likely at channel edges or decimation seams.")
        print("  Increase resolution, reduce decimation, or heal in MeshLab.")

# ---------------------------------------------------------------------------
# Bounding box of output
# ---------------------------------------------------------------------------
print("\nMesh bounding box (mm):")
print(f"  X: [{verts[:,0].min()*1e3:.1f}, {verts[:,0].max()*1e3:.1f}]")
print(f"  Y: [{verts[:,1].min()*1e3:.1f}, {verts[:,1].max()*1e3:.1f}]")
print(f"  Z: [{verts[:,2].min()*1e3:.1f}, {verts[:,2].max()*1e3:.1f}]")

print()
print("Open in Blender: File → Import → Stl...")
print("  geometry/output/chamber_nozzle_channels.stl")
print("  Channels should appear as rectangular grooves spiraling")
print("  (actually axial — parallel to z) around the chamber wall.")
