"""
render_chamber.py
=================

Generate the outer jacket (jacket OD) STL for the reference engine chamber
and nozzle.  No cooling channels — this is the smooth outer bell shape.

Geometry inputs: same bilayer sizing used in calcs/bilayer_check.py.

Output
------
  geometry/output/chamber_nozzle.stl   (coordinates in mm, for CAD/slicers)

Run with:
    python calcs/render_chamber.py

Expected runtime: channel sizer ~90 s, mesh extraction ~5-15 s.

Memory budget at 0.5 mm voxels
--------------------------------
  Outer diameter at chamber: ~55 mm -> radial bbox half-width ~30 mm
  Axial length: chamber + nozzle ≈ 125 + 30 (conv) + 85 (div) ≈ 240 mm
  Grid: 120 × 120 × 480 ≈ 6.9M cells × 4 bytes = 27 MB float32.  Fine.
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


from noyron.rp.geometry.chamber_nozzle import chamber_nozzle_sdf
from noyron.kernel.geometry.mesh.extractor import adaptive_bbox, sdf_to_mesh
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
# Channel sizing (reproduces bilayer_check.py geometry)
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
# Jacket thickness (autosized, same as bilayer_check.py)
# ---------------------------------------------------------------------------
_r_exit = NOZZLE.r_throat * NOZZLE.epsilon ** 0.5
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
# Bilayer wall description
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
# Build SDF and outer profile
# ---------------------------------------------------------------------------
print("Building chamber+nozzle SDF...")
sdf, profile = chamber_nozzle_sdf(
    rao_nozzle=NOZZLE,
    bilayer=bilayer,
    L_chamber=0.125,
)

print(f"  Axial domain  : z = {profile['z_min']*1e3:.1f} mm "
      f"to {profile['z_max']*1e3:.1f} mm")
print(f"  Max OD        : {profile['r_outer_max']*2e3:.1f} mm")
print(f"  Max r_inner   : {profile['r_inner'].max()*2e3:.1f} mm  (exit)")

# ---------------------------------------------------------------------------
# Voxel grid and mesh extraction
# ---------------------------------------------------------------------------
VOXEL_SIZE = 5e-4   # 0.5 mm

bbox = adaptive_bbox(
    nozzle_x_min=profile["z_min"],
    nozzle_x_max=profile["z_max"],
    r_outer_max=profile["r_outer_max"],
    radial_margin=5e-3,
    axial_margin=5e-3,
)
x_min, x_max, y_min, y_max, z_min, z_max = bbox

nx = round((x_max - x_min) / VOXEL_SIZE)
ny = round((y_max - y_min) / VOXEL_SIZE)
nz = round((z_max - z_min) / VOXEL_SIZE)
mem_mb = nx * ny * nz * 4 / 1e6
print(f"\nVoxel grid: {nx} x {ny} x {nz}  ({mem_mb:.0f} MB float32)")
print("Extracting mesh (marching cubes)...")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    verts, faces = sdf_to_mesh(sdf, bbox, voxel_size=VOXEL_SIZE)

print(f"  Vertices : {len(verts):,}")
print(f"  Faces    : {len(faces):,}")

# ---------------------------------------------------------------------------
# STL export
# ---------------------------------------------------------------------------
os.makedirs("geometry/output", exist_ok=True)
out_path = "geometry/output/chamber_nozzle.stl"

write_stl(verts, faces, out_path, scale_to_mm=True)
file_kb = os.path.getsize(out_path) / 1024
print(f"\nSTL written: {out_path}  ({file_kb:.0f} kB, coordinates in mm)")

# ---------------------------------------------------------------------------
# Manifold check
# ---------------------------------------------------------------------------
watertight = is_watertight(verts, faces)
print(f"Watertight (manifold): {watertight}")
if not watertight:
    print("  NOTE: non-manifold mesh — likely due to thin features at voxel scale.")
    print("  Increase resolution or inspect with MeshLab/Blender.")

# ---------------------------------------------------------------------------
# Bounding box of the output mesh
# ---------------------------------------------------------------------------
print("\nMesh bounding box (mm):")
print(f"  X: [{verts[:,0].min()*1e3:.1f}, {verts[:,0].max()*1e3:.1f}]")
print(f"  Y: [{verts[:,1].min()*1e3:.1f}, {verts[:,1].max()*1e3:.1f}]")
print(f"  Z: [{verts[:,2].min()*1e3:.1f}, {verts[:,2].max()*1e3:.1f}]")
print()
print("Open geometry/output/chamber_nozzle.stl in any STL viewer (MeshLab,")
print("Windows 3D Viewer, PrusaSlicer) to visually confirm the bell shape.")
