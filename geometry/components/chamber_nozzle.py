"""
chamber_nozzle.py
=================

3D SDF of the chamber + nozzle outer jacket surface (jacket OD).

Geometry
--------
The model covers three axial zones, revolved about the z-axis
(z = axial direction; x, y = radial):

  1. Cylindrical combustion chamber  (z_ch_start  to  z_conv_start)
     Constant outer radius: r_outer(z) = r_ch_outer

  2. Convergent section  (z_conv_start  to  z_throat)
     r_inner(z) from the RaoNozzle convergent arc.
     r_outer(z) = r_inner(z) + t_liner(z) + h_channel(z) + t_jacket(z)

  3. Divergent bell  (z_throat  to  z_exit)
     Same formula as zone 2.

Note: "z" here is the geometry-kernel axial coordinate; it maps
one-to-one to the nozzle's internal "x" coordinate (x=0 at throat).

Wall layers included
--------------------
  r_outer(z) = r_nozzle_inner(z)
             + t_liner(z)       [inner copper liner]
             + h_channel(z)     [coolant channel height]
             + t_jacket(z)      [outer structural jacket]

The coolant channels are NOT cut out — this produces a smooth outer bell
shape, not a channelled wall.  Channel geometry is a separate future task.

Out-of-scope (flagged for follow-up)
-------------------------------------
  - Coolant channel grooves (requires SDF for swept curves along contour)
  - Inlet/outlet manifold bosses
  - Mounting flanges and closeout ring
  - Injector plate (separate placeholder: injector_plate.py)

Sources
-------
Rao, G.V.R. (1958). "Exhaust Nozzle Contour for Optimum Thrust."
    Jet Propulsion, Vol. 28, No. 6.

NASA SP-8087 (1977) §3.3 -- bilayer wall construction.

Huzel & Huang (1992) p. 76 -- chamber geometry (L*, Ac/At, L_chamber).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import interp1d  # type: ignore[import-untyped]

from geometry.sdf.revolution import revolve

if TYPE_CHECKING:
    from physics.nozzle.rao import RaoNozzle
    from physics.structural.bilayer import BilayerWall


# Reference engine chamber length from CLAUDE.md
_L_CHAMBER_DEFAULT = 0.125  # m  (L_chamber, convergent start to injector face)


def chamber_nozzle_sdf(
    rao_nozzle: "RaoNozzle",
    bilayer: "BilayerWall",
    L_chamber: float = _L_CHAMBER_DEFAULT,
    n_profile_points: int = 500,
):
    """
    Build the 3D SDF of the outer jacket (jacket OD) of the chamber + nozzle.

    The SDF is a body of revolution: f < 0 inside the solid jacket, f > 0
    outside.  The zero-crossing is the outer jacket surface.

    The profile is sampled at ``n_profile_points`` axial stations from
    z_chamber_start to z_exit and stored as a numpy interpolant.  This
    allows the 3D SDF to be evaluated on large numpy arrays efficiently
    (vectorized r_of_z lookup).

    Parameters
    ----------
    rao_nozzle : RaoNozzle
        Nozzle contour and geometry.
    bilayer : BilayerWall
        Wall layer descriptions (t_liner, h_channel, t_jacket functions of x).
    L_chamber : float
        Cylindrical chamber length upstream of the convergent section [m].
        Default 0.125 m (reference engine per CLAUDE.md).
    n_profile_points : int
        Number of axial samples for the outer profile interpolant.

    Returns
    -------
    sdf : callable f(x, y, z) -> ndarray
        3D SDF of the outer jacket surface.
    profile : dict
        Auxiliary data:
          'z_ax'    : 1D array, axial stations [m] (z = nozzle x)
          'r_inner' : 1D array, gas-wall inner radius at each station [m]
          'r_outer' : 1D array, outer jacket OD at each station [m]
          'z_min'   : float, axial start of the profile [m]
          'z_max'   : float, axial end of the profile [m]
          'r_outer_max' : float, maximum outer radius [m]
    """
    # ------------------------------------------------------------------
    # 1.  Build the axial profile from z_chamber_start to z_exit
    # ------------------------------------------------------------------
    z_conv_start = float(rao_nozzle.x_inlet)   # convergent section starts here
    z_exit = float(rao_nozzle.x_exit)
    z_chamber_start = z_conv_start - L_chamber

    # Dense axial stations across nozzle domain (convergent + divergent)
    z_nozzle = np.linspace(z_conv_start, z_exit, n_profile_points)

    # Gas-side inner radius from the Rao contour at each nozzle station
    r_nozzle_inner = np.array([rao_nozzle.r(float(z)) for z in z_nozzle])

    # Outer radius = inner + all wall layers
    geom = bilayer.channel_geometry
    r_outer_nozzle = np.array([
        r_nozzle_inner[k]
        + bilayer.inner_liner.thickness(float(z_nozzle[k]))
        + geom.height(float(z_nozzle[k]))
        + bilayer.outer_jacket.thickness(float(z_nozzle[k]))
        for k in range(n_profile_points)
    ])

    # Chamber section: constant r_outer evaluated at z_conv_start
    r_ch_inner = float(rao_nozzle.r(z_conv_start))
    r_ch_outer = (
        r_ch_inner
        + bilayer.inner_liner.thickness(z_conv_start)
        + geom.height(z_conv_start)
        + bilayer.outer_jacket.thickness(z_conv_start)
    )

    # Chamber stations (upstream of convergent)
    z_chamber = np.linspace(z_chamber_start, z_conv_start, max(20, n_profile_points // 10))
    r_outer_chamber = np.full_like(z_chamber, r_ch_outer)
    r_inner_chamber = np.full_like(z_chamber, r_ch_inner)

    # Concatenate, dropping the duplicate junction point at z_conv_start
    z_all = np.concatenate([z_chamber[:-1], z_nozzle])
    r_outer_all = np.concatenate([r_outer_chamber[:-1], r_outer_nozzle])
    r_inner_all = np.concatenate([r_inner_chamber[:-1], r_nozzle_inner])

    z_min = float(z_all[0])
    z_max = float(z_all[-1])
    r_outer_max = float(r_outer_all.max())

    # ------------------------------------------------------------------
    # 2.  Build vectorised interpolant for the outer profile
    # ------------------------------------------------------------------
    _r_outer_interp = interp1d(
        z_all, r_outer_all,
        kind="linear",
        bounds_error=False,
        fill_value=(float(r_outer_all[0]), float(r_outer_all[-1])),
    )

    def r_of_z(z_arr: np.ndarray) -> np.ndarray:
        return _r_outer_interp(z_arr).astype(np.float32)

    # ------------------------------------------------------------------
    # 3.  Return the revolution SDF and profile data
    # ------------------------------------------------------------------
    sdf = revolve(r_of_z, z_min, z_max)

    profile = {
        "z_ax": z_all,
        "r_inner": r_inner_all,
        "r_outer": r_outer_all,
        "z_min": z_min,
        "z_max": z_max,
        "r_outer_max": r_outer_max,
    }
    return sdf, profile


def sdf_to_stl(
    sdf,
    bbox: tuple[float, float, float, float, float, float],
    voxel_size: float = 5e-4,
    output: str = "chamber.stl",
    scale_to_mm: bool = True,
) -> str:
    """
    Convenience wrapper: extract mesh from SDF and write to STL.

    Parameters
    ----------
    sdf : callable f(x, y, z) -> ndarray
    bbox : (x_min, x_max, y_min, y_max, z_min, z_max) [m]
    voxel_size : float   Grid spacing [m].  Default 0.5 mm.
    output : str         Output STL path.
    scale_to_mm : bool   Write STL in mm (default True — most CAD tools expect mm).

    Returns
    -------
    str   Path to the written STL file.
    """
    from geometry.mesh.extractor import sdf_to_mesh
    from geometry.mesh.stl_export import write_stl

    verts, faces = sdf_to_mesh(sdf, bbox, voxel_size=voxel_size)
    write_stl(verts, faces, output, scale_to_mm=scale_to_mm)
    return output
