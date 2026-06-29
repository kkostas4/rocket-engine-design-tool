"""
diag_lox_dome_contact.py
========================
Diagnose LOX dome contact with the injector face plate.

Checks:
  1. Dome geometry (OD, inner r, floor hole OD, z range)
  2. Face plate geometry (OD, inner r, z range)
  3. Support ring width at the contact plane (z = x_chamber_start)
  4. 8-point SDF probe around the support ring mid-radius
"""

from __future__ import annotations
import sys, warnings, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

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

from noyron.rp.geometry.blend_config import BlendConfig
from noyron.rp.geometry.manifolds import LoxDome, ManifoldConfig, _FACE_THICKNESS
from noyron.rp.geometry.injector import injector_sdf
from noyron.kernel.geometry.sdf.operations import smooth_union, difference

blend = BlendConfig()

# -----------------------------------------------------------------------
# 1. LOX Dome geometry
# -----------------------------------------------------------------------
ld = design.lox_dome
g = ld._geom()
cfg = ld.config

dome_r_inner   = g["r_inner"]           # = pintle_OD / 2
dome_r_outer   = g["r_outer"]           # = jacket OD at chamber
dome_wall_t    = cfg.wall_thickness     # 4 mm
floor_hole_r   = (design.injector.lox_annulus_OD + 4e-3) / 2.0  # flow_path radius
floor_hole_OD  = floor_hole_r * 2.0
dome_z_lo      = g["z_lo"]             # upstream cap
dome_z_hi      = g["z_hi"]             # = x_chamber_start

print("=== 1. LOX Dome geometry ===")
print(f"  dome OD            : {dome_r_outer*2e3:.3f} mm")
print(f"  dome r_inner       : {dome_r_inner*1e3:.3f} mm  (pintle_OD/2 = {design.injector.pintle_OD*1e3:.3f} mm)")
print(f"  dome wall_t        : {dome_wall_t*1e3:.1f} mm")
print(f"  dome cavity r range: [{(dome_r_inner+dome_wall_t)*1e3:.3f}, {(dome_r_outer-dome_wall_t)*1e3:.3f}] mm")
print(f"  floor hole OD      : {floor_hole_OD*1e3:.3f} mm  (lox_annulus_OD {design.injector.lox_annulus_OD*1e3:.3f} + 4 mm)")
print(f"  floor hole radius  : {floor_hole_r*1e3:.3f} mm")
print(f"  dome z_lo (upstream cap)  : {dome_z_lo*1e3:.3f} mm")
print(f"  dome z_hi (floor / x_ch_start): {dome_z_hi*1e3:.3f} mm")
print(f"  dome height        : {g['dome_height']*1e3:.3f} mm")

# -----------------------------------------------------------------------
# 2. Face plate geometry
# -----------------------------------------------------------------------
_, fp_info = injector_sdf(
    pintle=design.injector,
    nozzle=design.nozzle,
    bilayer=design.bilayer,
    face_thickness=_FACE_THICKNESS,
    blend_config=blend,
)

fp_r_outer  = fp_info["r_ch_inner"]        # = r_chamber (inner wall)
fp_r_inner  = fp_info["r_inner_face"]      # = lox_annulus_OD / 2
fp_z_face   = fp_info["z_face"]            # upstream face = x_chamber_start
fp_z_down   = fp_info["z_face_downstream"] # downstream face

print("\n=== 2. Face plate geometry ===")
print(f"  face plate OD (r_ch_inner) : {fp_r_outer*1e3:.3f} mm")
print(f"  face plate inner r         : {fp_r_inner*1e3:.3f} mm  (lox_annulus_OD/2)")
print(f"  face plate z_upstream      : {fp_z_face*1e3:.3f} mm")
print(f"  face plate z_downstream    : {fp_z_down*1e3:.3f} mm")

# -----------------------------------------------------------------------
# 3. Support ring calculation
# -----------------------------------------------------------------------
# At z = x_chamber_start:
#   Dome floor material spans r = floor_hole_r to r_outer (minus flow path cut)
#   Face plate material spans r = fp_r_inner to fp_r_outer
#
# Contact ring = radial overlap of both

support_ring_inner = max(floor_hole_r, fp_r_inner)
support_ring_outer = min(dome_r_outer, fp_r_outer)
support_width      = support_ring_outer - support_ring_inner
support_mid_r      = 0.5 * (support_ring_inner + support_ring_outer)

print("\n=== 3. Support ring at z = x_chamber_start ===")
print(f"  dome floor solid from r    : {floor_hole_r*1e3:.3f} mm (flow_path edge)")
print(f"  dome floor solid to   r    : {dome_r_outer*1e3:.3f} mm (dome OD/2)")
print(f"  face plate solid from r    : {fp_r_inner*1e3:.3f} mm")
print(f"  face plate solid to   r    : {fp_r_outer*1e3:.3f} mm")
print(f"  support_ring_inner         : {support_ring_inner*1e3:.3f} mm  = max({floor_hole_r*1e3:.3f}, {fp_r_inner*1e3:.3f})")
print(f"  support_ring_outer         : {support_ring_outer*1e3:.3f} mm  = min({dome_r_outer*1e3:.3f}, {fp_r_outer*1e3:.3f})")
print(f"  support_width              : {support_width*1e3:.3f} mm")

if support_width <= 0:
    verdict = "FLOATING — no radial overlap between dome floor and face plate"
elif support_width < 2e-3:
    verdict = f"MARGINAL — {support_width*1e3:.2f} mm < 2 mm; may not resolve at 0.4 mm voxels"
else:
    verdict = f"SUPPORTED — {support_width*1e3:.2f} mm contact ring"
print(f"  VERDICT: {verdict}")

# -----------------------------------------------------------------------
# 4. Full assembly SDF probe at 8 azimuthal points on support ring
# -----------------------------------------------------------------------
# Build the full assembly SDF (same logic as _build_full_assembly_sdf)
from noyron.rp.geometry.injector import engine_assembly_sdf

sdf_fn, _ = engine_assembly_sdf(
    nozzle=design.nozzle,
    bilayer=design.bilayer,
    channel_geometry=design.channel_geometry,
    pintle=design.injector,
    blend_config=blend,
)
for mfld in (design.lox_dome, design.fuel_manifold,
             design.coolant_outlet_manifold, design.coolant_inlet_manifold):
    if mfld is not None:
        sdf_fn = smooth_union(sdf_fn, mfld.to_sdf(blend_config=blend),
                              k=blend.manifold_to_jacket)
if design.fuel_manifold is not None:
    sdf_fn = difference(sdf_fn, design.fuel_manifold.flow_path_sdf())

z_contact = dome_z_hi  # = fp_z_face
N = 8
angles = [i * 2 * math.pi / N for i in range(N)]
r_probe = support_mid_r

print(f"\n=== 4. Assembly SDF at support ring (r={r_probe*1e3:.2f} mm, z={z_contact*1e3:.2f} mm) ===")
if support_width > 0:
    xs = np.array([r_probe * math.cos(a) for a in angles])
    ys = np.array([r_probe * math.sin(a) for a in angles])
    zs = np.full(N, z_contact)
    vals = sdf_fn(xs, ys, zs)
    for i, (v, a) in enumerate(zip(vals, angles)):
        tag = "SOLID" if v < -1e-9 else ("VOID" if v > 1e-9 else "SURFACE")
        print(f"  az={math.degrees(a):5.1f}deg  SDF={v*1e3:+.3f} mm  [{tag}]")
    n_solid = int(np.sum(vals < -1e-9))
    n_void  = int(np.sum(vals > 1e-9))
    print(f"  Summary: {n_solid}/8 solid, {n_void}/8 void")
    if n_void > 0:
        print(f"  *** VOID POINTS FOUND — dome NOT continuously solid at contact plane ***")
    else:
        print(f"  All points solid — dome floor contacts face plate.")
else:
    print(f"  Skipped — no support ring (support_width <= 0)")

# -----------------------------------------------------------------------
# Also sample at z slightly ABOVE contact (z_hi - 1 mm, inside dome) and
# z slightly BELOW (z_hi + 1 mm, inside face plate) to check connectivity
# -----------------------------------------------------------------------
print(f"\n=== 4b. Above/below contact plane at r={r_probe*1e3:.2f} mm ===")
for label, z_off in [
    ("z_hi - 2mm (inside dome floor)", -2e-3),
    ("z_hi exactly (contact)",          0.0),
    ("z_hi + 2mm (inside face plate)", +2e-3),
]:
    z = z_contact + z_off
    xs = np.array([r_probe * math.cos(a) for a in angles[:4]])
    ys = np.array([r_probe * math.sin(a) for a in angles[:4]])
    zs = np.full(4, z)
    vals = sdf_fn(xs, ys, zs)
    tags = ["S" if v < -1e-9 else ("V" if v > 1e-9 else "0") for v in vals]
    print(f"  {label}: {tags}  (SDF mean={vals.mean()*1e3:+.2f} mm)")
