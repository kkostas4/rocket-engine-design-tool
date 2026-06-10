"""
test_manifold_connectivity.py
=============================

Verify that each manifold's flow_path_sdf() creates a void that physically
connects the manifold cavity to the engine flow path it feeds.

A void is defined as SDF > 0.  A continuous flow path exists when every
sample point along a path through the void returns SDF > 0.

Tests work against the reference 1000-lbf methalox engine design to use
fully-consistent geometry parameters.

Sources
-------
Dressler & Bauer (2000) AIAA 2000-3871 — LOX dome geometry.
NASA SP-8087 (1977) §3.3 — coolant manifold geometry.
NASA SP-8089 (1976) §4.2 — fuel manifold geometry.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from noyron import Engine, EngineRequirements, METHALOX
from noyron.rp.geometry.manifolds import (
    LoxDome,
    FuelManifold,
    CoolantOutletManifold,
    CoolantInletManifold,
    ManifoldConfig,
)


# ---------------------------------------------------------------------------
# Reference engine fixture — instantiated once per module
# ---------------------------------------------------------------------------

_REQ = EngineRequirements(
    thrust=4448.22,
    chamber_pressure=6.8948e6,
    mixture_ratio=3.60,
    propellant=METHALOX,
    expansion_ratio=10.605,
)


@pytest.fixture(scope="module")
def design():
    return Engine(_REQ).design()


# ---------------------------------------------------------------------------
# Test 1: LoxDome flow path connects cavity to LOX annulus
# ---------------------------------------------------------------------------


def test_lox_dome_connects_to_annulus(design):
    """
    Sample the LoxDome SDF along a path from well inside the dome cavity
    through the flow-path hole (at z = z_hi = face_plate_z) into the LOX
    annulus region.  All points must be void (SDF > 0).
    """
    cfg = ManifoldConfig()
    dome = LoxDome(nozzle=design.nozzle, bilayer=design.bilayer,
                   injector=design.injector, config=cfg)
    sdf = dome.to_sdf()
    g = dome._geom()

    # Radial probe position: midpoint of LOX annulus gap
    r_probe = (design.injector.pintle_OD / 2.0 + design.injector.lox_annulus_OD / 2.0) / 2.0
    # Axial path: from near the dome floor up through the face (z_hi = face plate upstream face)
    z_lo_cav = g["z_lo"] + cfg.wall_thickness + 1e-4  # just inside cavity floor
    z_hi_annulus = g["z_hi"] + 2e-3                    # 2 mm past the dome into the annulus

    z_samples = np.linspace(z_lo_cav, z_hi_annulus, 12)
    x_probe = np.full_like(z_samples, r_probe)
    y_probe = np.zeros_like(z_samples)

    vals = sdf(x_probe, y_probe, z_samples)
    assert np.all(vals > 0), (
        f"LoxDome flow path not fully void along axial probe at r={r_probe*1e3:.2f} mm.\n"
        f"SDF values: {vals}"
    )


# ---------------------------------------------------------------------------
# Test 2: FuelManifold flow path connects cavity to orifices
# ---------------------------------------------------------------------------


def test_fuel_manifold_connects_to_orifices(design):
    """
    For orifice #0 (θ=0, along +x), sample radially from inside the
    fuel manifold cavity inward to just past the orifice centre.
    All points must be void.
    """
    cfg = ManifoldConfig()
    fm = FuelManifold(nozzle=design.nozzle, bilayer=design.bilayer,
                      injector=design.injector, config=cfg)
    sdf = fm.to_sdf()
    g = fm._geom()

    # z = face plate midplane  (same as flow_path_sdf uses)
    z_mid = g["z_hi"] + 8e-3 / 2.0

    r_inner_face = design.injector.lox_annulus_OD / 2.0
    r_orifice = r_inner_face + 2e-3           # orifice centre
    r_manifold_cav = g["r_inner"] + cfg.wall_thickness + 1e-4  # just inside cavity

    # Radial probe along +x axis from manifold cavity to orifice
    r_samples = np.linspace(r_manifold_cav, r_orifice, 12)
    x_probe = r_samples
    y_probe = np.zeros_like(r_samples)
    z_probe = np.full_like(r_samples, z_mid)

    vals = sdf(x_probe, y_probe, z_probe)
    assert np.all(vals > 0), (
        f"FuelManifold flow path not fully void along radial probe at z={z_mid*1e3:.1f} mm.\n"
        f"SDF values: {vals}"
    )


# ---------------------------------------------------------------------------
# Test 3: CoolantOutletManifold flow path connects cavity to channel exits
# ---------------------------------------------------------------------------


def test_coolant_outlet_connects_to_channels(design):
    """
    Sample radially from inside the CoolantOutletManifold cavity inward
    through the annular slot into the region where channel exits are.
    All points must be void.
    """
    cfg = ManifoldConfig()
    com = CoolantOutletManifold(nozzle=design.nozzle, bilayer=design.bilayer, config=cfg)
    sdf = com.to_sdf()
    g = com._geom()

    z_mid = g["z_ctr"]
    t = cfg.wall_thickness
    r_cav_inner = g["r_inner"] + t + 1e-4   # just inside manifold cavity
    r_channel = g["r_inner"] - 1e-4         # just inside the jacket (channel zone)

    # Radial probe from cavity inward through the slot
    r_samples = np.linspace(r_cav_inner, r_channel, 12)
    x_probe = r_samples
    y_probe = np.zeros_like(r_samples)
    z_probe = np.full_like(r_samples, z_mid)

    vals = sdf(x_probe, y_probe, z_probe)
    assert np.all(vals > 0), (
        f"CoolantOutletManifold flow path not fully void along radial probe.\n"
        f"SDF values: {vals}"
    )


# ---------------------------------------------------------------------------
# Test 4: CoolantInletManifold flow path connects cavity to channel inlets
# ---------------------------------------------------------------------------


def test_coolant_inlet_connects_to_channels(design):
    """
    Mirror of test_coolant_outlet_connects_to_channels at the nozzle exit end.
    """
    cfg = ManifoldConfig()
    cim = CoolantInletManifold(nozzle=design.nozzle, bilayer=design.bilayer, config=cfg)
    sdf = cim.to_sdf()
    g = cim._geom()

    z_mid = g["z_ctr"]
    t = cfg.wall_thickness
    r_cav_inner = g["r_inner"] + t + 1e-4
    r_channel = g["r_inner"] - 1e-4

    r_samples = np.linspace(r_cav_inner, r_channel, 12)
    x_probe = r_samples
    y_probe = np.zeros_like(r_samples)
    z_probe = np.full_like(r_samples, z_mid)

    vals = sdf(x_probe, y_probe, z_probe)
    assert np.all(vals > 0), (
        f"CoolantInletManifold flow path not fully void along radial probe.\n"
        f"SDF values: {vals}"
    )


# ---------------------------------------------------------------------------
# Test 5: No obvious leaks between LOX and fuel circuits
# ---------------------------------------------------------------------------


def test_no_obvious_leaks(design):
    """
    Verify that the LOX dome flow path and the fuel manifold flow path
    are spatially disjoint circuits.

    Each manifold exposes a flow_path_sdf() that models its passage as a
    signed-distance primitive with SDF < 0 inside the passage volume.

    Two checks:
    1. A probe inside the LOX annulus passage (dome flow_path SDF < 0) must
       be outside the fuel channel passage (fuel flow_path SDF > 0).
    2. A probe inside a fuel radial channel (fuel flow_path SDF < 0) must
       be outside the LOX annulus passage (dome flow_path SDF > 0).

    Source: Dressler & Bauer (2000) AIAA 2000-3871 §3; NASA SP-8089 §4.2.
    """
    cfg = ManifoldConfig()
    dome = LoxDome(nozzle=design.nozzle, bilayer=design.bilayer,
                   injector=design.injector, config=cfg)
    fm = FuelManifold(nozzle=design.nozzle, bilayer=design.bilayer,
                      injector=design.injector, config=cfg)

    dome_fp = dome.flow_path_sdf()
    fuel_fp = fm.flow_path_sdf()

    dome_g = dome._geom()
    fm_g = fm._geom()

    # --- Check 1: point inside LOX dome passage (r < lox_annulus_OD/2, inside dome z range) ---
    # LOX annulus is the gap between pintle OD/2 and lox_annulus_OD/2.
    # cylindrical_boss_sdf for dome flow_path has OD = lox_annulus_OD, so r_cylinder = lox_annulus_OD/2.
    # SDF < 0 inside the cylinder.  Pick r midway between pintle surface and annulus OD.
    r_lox = (design.injector.pintle_OD / 2.0 + design.injector.lox_annulus_OD / 2.0) / 2.0
    z_lox = (dome_g["z_lo"] + dome_g["z_hi"]) / 2.0

    d_lox = dome_fp(r_lox, 0.0, z_lox)
    f_lox = fuel_fp(r_lox, 0.0, z_lox)

    assert d_lox < 0, (
        f"Probe (r={r_lox*1e3:.2f} mm, z={z_lox*1e3:.2f} mm) expected to be inside "
        f"LOX dome flow path (SDF < 0) but got {d_lox:.6f}"
    )
    assert f_lox > 0, (
        f"LOX passage probe at (r={r_lox*1e3:.2f} mm, z={z_lox*1e3:.2f} mm) is also "
        f"inside fuel manifold flow path (fuel_fp = {f_lox:.6f} < 0). "
        f"LOX and fuel circuits must not share passage volume."
    )

    # --- Check 2: point inside a fuel radial channel at face-plate midplane ---
    # Channels run from r_inner (jacket OD) inward to r_orifice_center
    r_inner_face = design.injector.lox_annulus_OD / 2.0
    r_orifice_center = r_inner_face + 2e-3      # _ORIFICE_RADIAL_MARGIN
    r_fuel_ch = (fm_g["r_inner"] + r_orifice_center) / 2.0  # mid-channel radially
    z_fuel_ch = fm_g["z_hi"] + 8e-3 / 2.0       # _FACE_THICKNESS / 2 past manifold face

    d_fuel = dome_fp(r_fuel_ch, 0.0, z_fuel_ch)
    f_fuel = fuel_fp(r_fuel_ch, 0.0, z_fuel_ch)

    assert f_fuel < 0, (
        f"Probe (r={r_fuel_ch*1e3:.2f} mm, z={z_fuel_ch*1e3:.2f} mm) expected to be inside "
        f"fuel channel flow path (SDF < 0) but got {f_fuel:.6f}"
    )
    assert d_fuel > 0, (
        f"Fuel channel probe at (r={r_fuel_ch*1e3:.2f} mm, z={z_fuel_ch*1e3:.2f} mm) is also "
        f"inside LOX dome flow path (dome_fp = {d_fuel:.6f} < 0). "
        f"LOX and fuel circuits must not share passage volume."
    )
