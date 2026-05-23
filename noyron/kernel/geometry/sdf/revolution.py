"""
revolution.py
=============

Revolve a 2D radial profile r(z_ax) about the z-axis to produce a 3D SDF.

The coordinate system used throughout the geometry kernel:
  - z : axial direction (maps to the nozzle's internal x coordinate)
  - x, y : transverse (radial) directions
  - rho = sqrt(x^2 + y^2) : radial distance from the z-axis

SDF convention: f < 0 inside the solid, f > 0 outside.

Notes on exactness
------------------
The SDF returned by ``revolve()`` is **exact** along purely radial directions
(constant z), and is an **approximation** near the endcap planes (z = z_min
and z = z_max) and wherever the profile has high curvature in z.

At the z-endcaps the returned value is the axial distance to the cap plane,
not the true distance to the rim edge.  This means the isosurface (f = 0) is
exact (the zero-crossing is the true surface), but the gradient magnitude may
differ from 1 near the rims.  For marching-cubes mesh extraction the only
quantity that matters for surface location is the sign and zero-crossing, so
this approximation is harmless.

For exact SDF along caps as well, one would solve for the nearest point on
the 2D profile curve, which requires a root-find at every query point.  That
is unnecessary for this application.

Source: Quilez, I. (2023). "Signed Distance Functions — Revolution."
        iquilezles.org/articles/distfunctions.  (The exact-rim variant uses
        a 2D SDF on the profile curve; the approximate variant used here is
        the radial-distance shorthand.)
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def hollow_revolve(
    r_outer_of_z: Callable[[np.ndarray], np.ndarray],
    r_inner_of_z: Callable[[np.ndarray], np.ndarray],
    z_min: float,
    z_max: float,
) -> Callable:
    """
    Build a 3D SDF for a hollow capped shell of revolution.

    Equivalent to ``difference(revolve(r_outer_of_z, z_min, z_max),
    revolve(r_inner_of_z, z_min, z_max))`` but ~3× faster: computes rho,
    z_clamped, f_bot, and f_top once instead of twice, and eliminates the
    difference closure layer.

    The solid occupies the annular shell:
        { (x,y,z) : z_min <= z <= z_max
                    AND r_inner_of_z(z) <= sqrt(x^2+y^2) <= r_outer_of_z(z) }

    SDF value:
        f_outer = max(rho - r_outer(z_c), z_min - z, z - z_max)
        f_inner = max(rho - r_inner(z_c), z_min - z, z - z_max)
        f       = max(f_outer, -f_inner)

    This gives f < 0 inside the shell and f > 0 outside (gas path or exterior).

    Parameters
    ----------
    r_outer_of_z : Callable[[ndarray], ndarray]
        Outer radial profile [m]. Called with z_clamped.
    r_inner_of_z : Callable[[ndarray], ndarray]
        Inner radial profile [m]. Called with z_clamped.
    z_min : float
        Axial start of the shell [m].
    z_max : float
        Axial end of the shell [m].

    Returns
    -------
    Callable f(x, y, z) -> ndarray
        Signed distance function.

    Source: Quilez, I. (2023). "Signed Distance Functions — Subtraction."
            iquilezles.org/articles/distfunctions.
    """
    if z_max <= z_min:
        raise ValueError(f"z_max ({z_max}) must be > z_min ({z_min})")

    def _f(x, y, z):
        x_ = np.asarray(x, dtype=np.float32)
        y_ = np.asarray(y, dtype=np.float32)
        z_ = np.asarray(z, dtype=np.float32)

        rho = np.sqrt(x_ * x_ + y_ * y_)
        z_c = np.clip(z_, z_min, z_max)
        z_c64 = z_c.astype(float)

        r_out = r_outer_of_z(z_c64).astype(np.float32)
        r_in  = r_inner_of_z(z_c64).astype(np.float32)

        f_bot = np.float32(z_min) - z_
        f_top = z_ - np.float32(z_max)

        f_outer = np.maximum(np.maximum(rho - r_out, f_bot), f_top)
        f_inner = np.maximum(np.maximum(rho - r_in,  f_bot), f_top)

        return np.maximum(f_outer, -f_inner)

    return _f


def revolve(
    r_of_z: Callable[[np.ndarray], np.ndarray],
    z_min: float,
    z_max: float,
) -> Callable:
    """
    Build a 3D SDF for a capped solid of revolution.

    The solid occupies:
        { (x, y, z) : z_min <= z <= z_max  AND  sqrt(x^2+y^2) <= r_of_z(z) }

    The SDF value is:
        f(x,y,z) = max(rho - r_of_z(z_clamped),  z_min - z,  z - z_max)

    where z_clamped = clip(z, z_min, z_max).  This correctly classifies
    all points as inside/outside and gives the correct surface location.

    Parameters
    ----------
    r_of_z : Callable[[ndarray], ndarray]
        Radial profile r(z_ax) [m].  Must accept a numpy array argument and
        return a numpy array of the same shape.  Will be called with
        z_clamped = clip(z, z_min, z_max) so no out-of-range z values are
        ever passed in.
    z_min : float
        Axial start of the solid [m].
    z_max : float
        Axial end of the solid [m].

    Returns
    -------
    Callable f(x, y, z) -> ndarray
        Signed distance function.
    """
    if z_max <= z_min:
        raise ValueError(f"z_max ({z_max}) must be > z_min ({z_min})")

    def _f(x, y, z):
        x_ = np.asarray(x, dtype=np.float32)
        y_ = np.asarray(y, dtype=np.float32)
        z_ = np.asarray(z, dtype=np.float32)

        rho = np.sqrt(x_ * x_ + y_ * y_)
        z_c = np.clip(z_, z_min, z_max)
        r = r_of_z(z_c.astype(float))  # profile may need float64

        f_wall = rho - r.astype(np.float32)
        f_bot = np.float32(z_min) - z_
        f_top = z_ - np.float32(z_max)

        return np.maximum(np.maximum(f_wall, f_bot), f_top)

    return _f
