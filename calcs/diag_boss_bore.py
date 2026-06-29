"""
diag_boss_bore.py
=================
Probe boss bore penetration through manifold walls.

Samples ABCD along each manifold's boss bore axis:
  A – boss tip (outside)
  B – boss/wall junction (outer manifold surface)
  C – 2 mm inside the manifold wall
  D – 7 mm inside (well into the cavity)

VOID = SDF > 0, SOLID = SDF < 0.
"""

from __future__ import annotations
import sys, warnings
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
blend = BlendConfig()


def tag(v: float) -> str:
    if v > 1e-9:
        return "VOID"
    if v < -1e-9:
        return "SOLID"
    return "SURFACE"


def probe(sdf, points: list[tuple[str, float, float, float]]) -> None:
    for label, x, y, z in points:
        v = float(sdf(np.array([x]), np.array([y]), np.array([z]))[0])
        print(f"  {label:35s}  SDF={v*1e3:+8.3f} mm  [{tag(v)}]")


# -----------------------------------------------------------------------
# LoxDome — axial boss pointing –z from (0, 0, z_lo)
# -----------------------------------------------------------------------
ld = design.lox_dome
gl = ld._geom()
cfg_ld = ld.config
sdf_ld = ld.to_sdf(blend_config=blend)
z_lo_ld = gl["z_lo"]
bp = cfg_ld.boss_protrusion           # 15 mm
t_ld = cfg_ld.wall_thickness          # 4 mm
bore_inner_r_ld = (cfg_ld.lox_dome_inlet_OD - 2 * t_ld) / 2.0   # 2 mm
px = bore_inner_r_ld * 0.5            # 1 mm — inside bore

print("=== LoxDome ===")
print(f"  boss_OD={cfg_ld.lox_dome_inlet_OD*1e3:.0f}mm  "
      f"bore_inner_r={bore_inner_r_ld*1e3:.1f}mm  "
      f"wall_t={t_ld*1e3:.0f}mm  probe_r={px*1e3:.1f}mm")
print(f"  z_lo={z_lo_ld*1e3:.2f}mm  cavity_z_lo={(z_lo_ld+t_ld)*1e3:.2f}mm")
probe(sdf_ld, [
    ("A boss_tip    z=z_lo-bp+1mm",   px, 0.0,  z_lo_ld - bp + 0.001),
    ("B junction    z=z_lo",          px, 0.0,  z_lo_ld),
    ("C 2mm in wall z=z_lo+0.002",    px, 0.0,  z_lo_ld + 0.002),
    ("D 7mm in      z=z_lo+0.007",    px, 0.0,  z_lo_ld + 0.007),
])

# -----------------------------------------------------------------------
# FuelManifold — radial boss pointing +x from (r_outer, 0, z_ctr)
# -----------------------------------------------------------------------
fm = design.fuel_manifold
gf = fm._geom()
cfg_fm = fm.config
sdf_fm = fm.to_sdf(blend_config=blend)   # shell+boss union, no flow_path
r_out_f = gf["r_outer"]
z_ctr_f = gf["z_ctr"]
t_fm = cfg_fm.wall_thickness
bore_inner_r_fm = (cfg_fm.fuel_manifold_inlet_OD - 2 * t_fm) / 2.0  # 1 mm
py = bore_inner_r_fm * 0.5  # 0.5 mm — inside bore (perpendicular offset in y)

print(f"\n=== FuelManifold ===")
print(f"  boss_OD={cfg_fm.fuel_manifold_inlet_OD*1e3:.0f}mm  "
      f"bore_inner_r={bore_inner_r_fm*1e3:.1f}mm  "
      f"wall_t={t_fm*1e3:.0f}mm  probe_y={py*1e3:.1f}mm")
print(f"  r_outer={r_out_f*1e3:.2f}mm  cavity_r_outer={(r_out_f-t_fm)*1e3:.2f}mm")
probe(sdf_fm, [
    ("A boss_tip    x=r_out+bp-1mm",  r_out_f + bp - 0.001, py, z_ctr_f),
    ("B junction    x=r_out",         r_out_f,               py, z_ctr_f),
    ("C 2mm in wall x=r_out-0.002",   r_out_f - 0.002,       py, z_ctr_f),
    ("D 7mm in      x=r_out-0.007",   r_out_f - 0.007,       py, z_ctr_f),
])

# -----------------------------------------------------------------------
# CoolantOutletManifold — radial boss pointing +y from (0, r_outer, z_ctr)
# -----------------------------------------------------------------------
co = design.coolant_outlet_manifold
go = co._geom()
cfg_co = co.config
sdf_co = co.to_sdf(blend_config=blend)
r_out_o = go["r_outer"]
z_ctr_o = go["z_ctr"]
t_co = cfg_co.wall_thickness
bore_inner_r_co = (cfg_co.coolant_manifold_inlet_OD - 2 * t_co) / 2.0  # 1 mm
px_co = bore_inner_r_co * 0.5  # 0.5 mm

print(f"\n=== CoolantOutletManifold ===")
print(f"  boss_OD={cfg_co.coolant_manifold_inlet_OD*1e3:.0f}mm  "
      f"bore_inner_r={bore_inner_r_co*1e3:.1f}mm  "
      f"wall_t={t_co*1e3:.0f}mm  probe_x={px_co*1e3:.1f}mm")
print(f"  r_outer={r_out_o*1e3:.2f}mm  cavity_r_outer={(r_out_o-t_co)*1e3:.2f}mm")
probe(sdf_co, [
    ("A boss_tip    y=r_out+bp-1mm",  px_co,  r_out_o + bp - 0.001, z_ctr_o),
    ("B junction    y=r_out",         px_co,  r_out_o,               z_ctr_o),
    ("C 2mm in wall y=r_out-0.002",   px_co,  r_out_o - 0.002,       z_ctr_o),
    ("D 7mm in      y=r_out-0.007",   px_co,  r_out_o - 0.007,       z_ctr_o),
])

# -----------------------------------------------------------------------
# CoolantInletManifold — radial boss pointing +y from (0, r_outer, z_ctr)
# -----------------------------------------------------------------------
ci = design.coolant_inlet_manifold
gi = ci._geom()
cfg_ci = ci.config
sdf_ci = ci.to_sdf(blend_config=blend)
r_out_i = gi["r_outer"]
z_ctr_i = gi["z_ctr"]
t_ci = cfg_ci.wall_thickness
bore_inner_r_ci = (cfg_ci.coolant_manifold_inlet_OD - 2 * t_ci) / 2.0  # 1 mm
px_ci = bore_inner_r_ci * 0.5  # 0.5 mm

print(f"\n=== CoolantInletManifold ===")
print(f"  boss_OD={cfg_ci.coolant_manifold_inlet_OD*1e3:.0f}mm  "
      f"bore_inner_r={bore_inner_r_ci*1e3:.1f}mm  "
      f"wall_t={t_ci*1e3:.0f}mm  probe_x={px_ci*1e3:.1f}mm")
print(f"  r_outer={r_out_i*1e3:.2f}mm  cavity_r_outer={(r_out_i-t_ci)*1e3:.2f}mm")
probe(sdf_ci, [
    ("A boss_tip    y=r_out+bp-1mm",  px_ci,  r_out_i + bp - 0.001, z_ctr_i),
    ("B junction    y=r_out",         px_ci,  r_out_i,               z_ctr_i),
    ("C 2mm in wall y=r_out-0.002",   px_ci,  r_out_i - 0.002,       z_ctr_i),
    ("D 7mm in      y=r_out-0.007",   px_ci,  r_out_i - 0.007,       z_ctr_i),
])
