"""
test_injector_geometry.py
=========================

Verification tests for geometry/components/injector.py.

Reference engine (CLAUDE.md):
  - Throat diameter  : 22.313 mm  → r_throat = 11.157 mm
  - Chamber diameter : 44.625 mm  → r_ch = 22.313 mm
  - Chamber length   : 125 mm
  - Expansion ratio  : 10.605
  - Pintle OD        : 30 mm
  - LOX annulus OD   : ~30.47 mm  (computed from reference dP/flow)
  - 20 × 0.819 mm fuel orifices
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from geometry.components.injector import injector_sdf
from physics.injector.pintle import PintleInjector
from physics.nozzle.rao import RaoNozzle
from physics.regen.coolant import GRCOP42, CoolingChannelGeometry
from physics.regen.sizing import ChannelSizingProblem, size_channels
from physics.structural.bilayer import (
    BilayerWall,
    RibGeometry,
    WallLayer,
    size_jacket_thickness,
)
from physics.structural.materials import GRCOP84_PROXY, INCONEL718_STRUCT
from physics.regen.bartz import CombustionGas
from physics.regen.coolant import CoolantInlet


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_NOZZLE = RaoNozzle(r_throat=0.022313 / 2.0, epsilon=10.605)

_PINTLE = PintleInjector.from_uniform_dP(
    dP_inj=0.45 * 6.8948e6,   # 45 % of Pc = reference engine
    m_dot_ox=1.1144,
    m_dot_fuel=0.3096,
    rho_ox=1116.0,
    rho_fuel=386.0,
    Cd_ox=0.6,
    Cd_fuel=0.6,
    pintle_OD=0.030,
    n_fuel_orifices=20,
)


@pytest.fixture(scope="module")
def bilayer_and_geometry():
    """Build a minimal BilayerWall from the reference engine channel sizing."""
    gas = CombustionGas(
        Tc=3561.2,
        Pc=6.8948e6,
        gamma=1.15,
        MW=21.9,
        cp=2000.0,
        mu=8.0e-5,
        Pr=0.60,
        c_star=1893.3,
    )
    coolant = CoolantInlet(T=150.0, P=12.0e6, m_dot=0.3096, fluid="Methane")
    import warnings
    problem = ChannelSizingProblem(
        nozzle=_NOZZLE,
        gas=gas,
        coolant_inlet=coolant,
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
        geom = size_channels(problem)

    r_exit = _NOZZLE.r_throat * math.sqrt(_NOZZLE.epsilon)
    r_jacket_exit = r_exit + 0.5e-3 + geom.height(_NOZZLE.x_exit)
    t_jacket = max(
        1.5e-3,
        math.ceil(
            size_jacket_thickness(
                P_coolant_max=12.0e6,
                r_jacket=r_jacket_exit,
                material=INCONEL718_STRUCT,
                T_op=200.0,
                FoS=1.4,
            )
            / 0.5e-3
        ) * 0.5e-3,
    )
    bl = BilayerWall(
        inner_liner=WallLayer(thickness=geom.wall_thickness, material=GRCOP84_PROXY),
        outer_jacket=WallLayer(thickness=lambda x, _t=t_jacket: _t, material=INCONEL718_STRUCT),
        ribs=RibGeometry(
            n_ribs=geom.n_channels,
            land_width=geom.land_thickness,
            height=geom.height,
            material=GRCOP84_PROXY,
        ),
        channel_geometry=geom,
    )
    return bl, geom


@pytest.fixture(scope="module")
def inj_sdf_and_info(bilayer_and_geometry):
    bl, _ = bilayer_and_geometry
    sdf, info = injector_sdf(pintle=_PINTLE, nozzle=_NOZZLE, bilayer=bl)
    return sdf, info


# ---------------------------------------------------------------------------
# Test 1: pintle centred on axis is inside the solid (f < 0)
# ---------------------------------------------------------------------------

def test_pintle_centered_on_axis(inj_sdf_and_info):
    """SDF at (0, 0, z_within_pintle) is negative — inside the pintle post."""
    sdf, info = inj_sdf_and_info
    z_mid = info["z_face"] + info["pintle_length"] / 2.0
    val = float(sdf(0.0, 0.0, z_mid))
    assert val < 0.0, (
        f"Expected SDF < 0 inside pintle at (0, 0, {z_mid*1e3:.1f} mm), got {val:.4e}"
    )


# ---------------------------------------------------------------------------
# Test 2: LOX annulus gap is void (f > 0 at gap midpoint)
# ---------------------------------------------------------------------------

def test_lox_annulus_gap_correct(inj_sdf_and_info):
    """SDF at the radial midpoint of the LOX annulus gap is positive (void)."""
    sdf, info = inj_sdf_and_info
    # Midpoint between pintle OD and LOX annulus OD
    r_gap_mid = info["r_pintle"] + (_PINTLE.lox_annulus_gap / 2.0)
    z_mid = info["z_face"] + info["face_thickness"] / 2.0
    val = float(sdf(r_gap_mid, 0.0, z_mid))
    assert val > 0.0, (
        f"Expected SDF > 0 in LOX annulus gap at r={r_gap_mid*1e3:.2f} mm, "
        f"z={z_mid*1e3:.1f} mm, got {val:.4e}"
    )


# ---------------------------------------------------------------------------
# Test 3: fuel orifice count — 2×N sign transitions around the circle
# ---------------------------------------------------------------------------

def test_fuel_orifice_count(inj_sdf_and_info):
    """
    Sampling the SDF at r_orifice_center, z = face mid-thickness yields exactly
    2×N_orifices sign transitions as theta sweeps 0 to 2π.
    """
    sdf, info = inj_sdf_and_info
    N = info["n_fuel_orifices"]
    r = info["r_orifice_center"]
    z = info["z_face"] + info["face_thickness"] / 2.0

    # Dense angular sampling (100 points per orifice)
    n_theta = N * 100
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    xs = r * np.cos(theta)
    ys = r * np.sin(theta)
    zs = np.full(n_theta, z)

    vals = sdf(xs, ys, zs)
    signs = np.sign(vals)

    # Count transitions (sign changes between consecutive samples, wrapping around)
    transitions = int(np.sum(signs[:-1] != signs[1:]))
    # Handle wrap-around
    if signs[-1] != signs[0]:
        transitions += 1

    assert transitions == 2 * N, (
        f"Expected {2*N} sign transitions (2 per orifice, N={N}), "
        f"got {transitions}"
    )


# ---------------------------------------------------------------------------
# Test 4: assembly watertight at 0.5 mm voxel
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_assembly_watertight(bilayer_and_geometry, tmp_path):
    """
    Full engine assembly STL at 0.5 mm voxels is watertight (manifold mesh).
    """
    import warnings
    from geometry.components.injector import engine_assembly_sdf
    from geometry.mesh.chunked_extractor import chunked_sdf_to_mesh
    from geometry.mesh.extractor import adaptive_bbox
    from geometry.mesh.stl_export import is_watertight

    bl, geom = bilayer_and_geometry
    sdf, profile = engine_assembly_sdf(
        nozzle=_NOZZLE,
        bilayer=bl,
        channel_geometry=geom,
        pintle=_PINTLE,
    )

    bbox = adaptive_bbox(
        nozzle_x_min=profile["z_min"],
        nozzle_x_max=profile["z_max"],
        r_outer_max=profile["r_outer_max"],
        radial_margin=3e-3,
        axial_margin=3e-3,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        verts, faces = chunked_sdf_to_mesh(
            sdf,
            bbox,
            voxel_size=5e-4,   # 0.5 mm
            slab_axial_steps=25,
            verbose=False,
        )

    assert len(faces) > 0, "Mesh extraction produced no faces"
    watertight = is_watertight(verts, faces)
    assert watertight, (
        f"Assembly mesh is not watertight: {len(verts):,} verts / "
        f"{len(faces):,} faces"
    )


# ---------------------------------------------------------------------------
# Test 5: face plate outer radius matches the cylindrical chamber inner radius
# ---------------------------------------------------------------------------

def test_face_plate_outer_radius_matches_chamber(inj_sdf_and_info):
    """
    Face plate OD equals r_throat * sqrt(Ac/At).

    The reference engine has Ac/At = 4.0 (CLAUDE.md), giving a chamber inner
    radius of r_throat * 2 = 22.313 mm.  injector_sdf() derives r_ch_inner
    from this relation rather than from nozzle.r(x_inlet), which only gives
    the Rao convergent-arc inlet radius (~1.2*R_t ≈ 13.4 mm) and is too small
    for the pintle to fit.
    """
    _, info = inj_sdf_and_info
    _AC_AT = 4.0   # reference engine contraction ratio (CLAUDE.md)
    r_expected = _NOZZLE.r_throat * math.sqrt(_AC_AT)
    r_face_od = info["r_ch_inner"]

    # Allow sub-micron floating-point tolerance
    tol = 1e-9
    assert abs(r_face_od - r_expected) < tol, (
        f"Face plate outer radius {r_face_od*1e3:.4f} mm differs from "
        f"r_throat*sqrt(Ac/At) = {r_expected*1e3:.4f} mm by "
        f"{abs(r_face_od - r_expected)*1e3:.6f} mm"
    )
