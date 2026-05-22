"""
injector.py
===========

3D SDF geometry for the pintle injector face plate and post, and the full
engine assembly (chamber + nozzle + injector).

Geometry model
--------------
The injector sits at the upstream end of the cylindrical combustion chamber
(z = z_chamber_start = z_conv_start - L_chamber).

Three components are modelled:

  1. Face plate — annular disc, outer radius = r_ch_inner (chamber inner wall),
     inner radius = lox_annulus_OD / 2 (LOX flow-through hole), thickness = 8 mm.
     N fuel orifices (diameter = fuel_orifice_diameter) are drilled through the
     plate, centred at r_inner_face + 2 mm radially, equally spaced at 2π/N
     around the circumference.

  2. Pintle post — solid cylinder, diameter = pintle_OD, centred on the z-axis,
     extending downstream from the face plate by `pintle_protrusion` (default
     1.5 × pintle_OD).

  3. LOX annulus — the radial gap between the pintle OD and the face plate inner
     hole is the natural flow path for the LOX; no additional geometry is needed.

Coordinate system: z = 0 at the throat; z_chamber_start < 0 (upstream).

Sources
-------
Dressler, G.A. and Bauer, J.M. (2000). "TRW Pintle Engine Heritage and
    Performance Characteristics." AIAA 2000-3871. §3 — geometry and TMR.

NASA SP-8089 (1976). "Liquid Rocket Engine Injectors." §4.2 — face plate.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np

from noyron.kernel.geometry.sdf.operations import difference, union
from noyron.kernel.geometry.sdf.primitives import cylinder_z
from noyron.kernel.geometry.sdf.revolution import revolve
from noyron.kernel.geometry.sdf.transforms import rotate_z

if TYPE_CHECKING:
    from noyron.rp.injector.pintle import PintleInjector
    from noyron.rp.nozzle.rao import RaoNozzle
    from noyron.rp.structural.bilayer import BilayerWall
    from noyron.rp.regen.coolant import CoolingChannelGeometry


_ORIFICE_RADIAL_MARGIN = 2e-3  # m  orifice centre offset inward from LOX annulus OD


def injector_sdf(
    *,
    pintle: "PintleInjector",
    nozzle: "RaoNozzle | None" = None,
    bilayer: "BilayerWall | None" = None,
    face_thickness: float = 8e-3,
    pintle_protrusion: float | None = None,
) -> tuple[Callable, dict]:
    """
    Build the 3D SDF of the injector face plate + pintle post.

    The SDF is negative inside solid material and positive in void regions
    (LOX annulus gap, fuel orifice bores, surrounding space).

    Parameters
    ----------
    pintle : PintleInjector
        Sized injector (provides pintle_OD, lox_annulus_OD, fuel_orifice_diameter,
        n_fuel_orifices).
    nozzle : RaoNozzle or None
        Nozzle contour (provides x_chamber_start for face plate position and
        r_chamber for the cylindrical chamber inner radius).  When None,
        local coordinates are used: z_face = 0 and r_ch_inner = lox_annulus_OD.
    bilayer : BilayerWall or None
        Wall layers (accepted for API consistency; not used for geometry here).
    face_thickness : float
        Axial thickness of the injector face plate [m].  Default 8 mm.
    pintle_protrusion : float or None
        How far the pintle post extends downstream from the face plate
        downstream face [m].  Default = 1.5 × pintle_OD.

    Returns
    -------
    sdf : callable f(x, y, z) -> ndarray
        3D SDF.  Negative inside solid, positive in void.
    geom_info : dict
        Auxiliary geometry data:
          'z_face'           : float  upstream face of face plate [m]
          'z_face_downstream': float  downstream face of face plate [m]
          'r_ch_inner'       : float  chamber inner radius (= face plate OD) [m]
          'r_inner_face'     : float  face plate inner hole radius (= LOX annulus OD/2) [m]
          'r_orifice_center' : float  radial position of fuel orifice centres [m]
          'r_pintle'         : float  pintle post radius [m]
          'z_pintle_tip'     : float  axial position of pintle tip [m]
          'pintle_length'    : float  total pintle length [m]
          'face_thickness'   : float  face plate axial thickness [m]
          'n_fuel_orifices'  : int    number of fuel orifices
          'r_hole'           : float  fuel orifice radius [m]
    """
    # ------------------------------------------------------------------
    # 1.  Geometry parameters
    # ------------------------------------------------------------------
    if nozzle is not None:
        z_face = float(nozzle.x_chamber_start)
        r_ch_inner = float(nozzle.r_chamber)
    else:
        z_face = 0.0
        r_ch_inner = pintle.lox_annulus_OD
    z_face_down = z_face + face_thickness

    r_pintle = pintle.pintle_OD / 2.0
    r_inner_face = pintle.lox_annulus_OD / 2.0      # hole through which LOX flows
    r_hole = pintle.fuel_orifice_diameter / 2.0
    r_orifice_center = r_inner_face + _ORIFICE_RADIAL_MARGIN

    _pintle_protrusion = (
        pintle_protrusion
        if pintle_protrusion is not None
        else 1.5 * pintle.pintle_OD
    )
    pintle_length = face_thickness + _pintle_protrusion
    z_pintle_tip = z_face + pintle_length

    # ------------------------------------------------------------------
    # 2.  Face plate: annular disc (outer solid minus inner LOX hole)
    # ------------------------------------------------------------------
    _r_outer = np.float32(r_ch_inner)
    _r_inner = np.float32(r_inner_face)

    outer_disc = revolve(
        lambda z_arr, _r=_r_outer: np.full(np.shape(z_arr), _r, dtype=np.float32),
        z_face, z_face_down,
    )
    lox_hole = revolve(
        lambda z_arr, _r=_r_inner: np.full(np.shape(z_arr), _r, dtype=np.float32),
        z_face, z_face_down,
    )
    face_plate_ring = difference(outer_disc, lox_hole)

    # ------------------------------------------------------------------
    # 3.  Fuel orifices: N z-aligned cylinders at r_orifice_center, evenly
    #     distributed around the circumference.  rotate_z replicates one
    #     prototype orifice at each angular position.
    # ------------------------------------------------------------------
    N = pintle.n_fuel_orifices
    one_orifice = cylinder_z(center_xy=(r_orifice_center, 0.0), radius=r_hole)

    orifices: Callable = one_orifice
    for i in range(1, N):
        angle = i * 2.0 * math.pi / N
        orifices = union(orifices, rotate_z(one_orifice, angle))

    face_plate = difference(face_plate_ring, orifices)

    # ------------------------------------------------------------------
    # 4.  Pintle post: solid cylinder from z_face to z_pintle_tip
    # ------------------------------------------------------------------
    _r_pin = np.float32(r_pintle)
    pintle_post = revolve(
        lambda z_arr, _r=_r_pin: np.full(np.shape(z_arr), _r, dtype=np.float32),
        z_face, z_pintle_tip,
    )

    # ------------------------------------------------------------------
    # 5.  Full injector = face plate ∪ pintle post
    # ------------------------------------------------------------------
    sdf = union(face_plate, pintle_post)

    geom_info: dict = {
        "z_face": z_face,
        "z_face_downstream": z_face_down,
        "r_ch_inner": r_ch_inner,
        "r_inner_face": r_inner_face,
        "r_orifice_center": r_orifice_center,
        "r_pintle": r_pintle,
        "z_pintle_tip": z_pintle_tip,
        "pintle_length": pintle_length,
        "face_thickness": face_thickness,
        "n_fuel_orifices": N,
        "r_hole": r_hole,
    }
    return sdf, geom_info


def engine_assembly_sdf(
    *,
    nozzle: "RaoNozzle",
    bilayer: "BilayerWall",
    channel_geometry: "CoolingChannelGeometry",
    pintle: "PintleInjector",
    face_thickness: float = 8e-3,
    n_profile_points: int = 500,
) -> tuple[Callable, dict]:
    """
    Build the 3D SDF of the complete engine assembly.

    Combines the channelled chamber+nozzle wall with the injector face plate
    and pintle post via SDF union.

    Parameters
    ----------
    nozzle : RaoNozzle
        Owns all chamber geometry (r_chamber, L_chamber, x_chamber_start).
    bilayer : BilayerWall
    channel_geometry : CoolingChannelGeometry
    pintle : PintleInjector
    face_thickness : float
        Face plate axial thickness [m].  Default 8 mm.
    n_profile_points : int
        Axial sample count for the chamber+nozzle profile interpolants.

    Returns
    -------
    sdf : callable f(x, y, z) -> ndarray
        Assembly SDF.
    profile : dict
        Combined profile with keys from chamber_with_channels_sdf plus:
          'z_face'       : float  upstream face of face plate [m]
          'z_min'        : float  updated z_min = z_face [m]
          'r_outer_max'  : float  max outer radius across the whole assembly [m]
          'injector_info': dict   geom_info from injector_sdf()
    """
    from noyron.rp.geometry.cooling_channels import chamber_with_channels_sdf

    # Trim chamber domain by face_thickness so the face-plate zone is left
    # entirely to f_inj — otherwise the revolve solid buries orifice voids.
    _z_face = float(nozzle.x_chamber_start)
    _L_trimmed = float(nozzle.x_inlet) - _z_face - face_thickness

    f_main, profile = chamber_with_channels_sdf(
        rao_nozzle=nozzle,
        bilayer=bilayer,
        n_profile_points=n_profile_points,
        L_chamber=max(_L_trimmed, 0.0),
    )

    f_inj, inj_info = injector_sdf(
        pintle=pintle,
        nozzle=nozzle,
        bilayer=bilayer,
        face_thickness=face_thickness,
    )

    f_assembly = union(f_main, f_inj)

    # z_face == nozzle.x_chamber_start, which matches z_min from chamber profile.
    z_face = float(inj_info["z_face"])
    r_outer_max = max(float(profile["r_outer_max"]), float(inj_info["r_ch_inner"]))

    combined_profile = dict(profile)
    combined_profile["z_face"] = z_face
    combined_profile["z_min"] = min(float(profile["z_min"]), z_face)
    combined_profile["r_outer_max"] = r_outer_max
    combined_profile["injector_info"] = inj_info

    return f_assembly, combined_profile
