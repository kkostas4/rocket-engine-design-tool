"""
primitives.py
=============

Exact signed distance functions (SDFs) for primitive shapes.

Convention: f(x, y, z) < 0 inside, > 0 outside, = 0 on the surface.

All functions accept numpy arrays for x, y, z (broadcast-compatible) and
return a numpy array of the same shape.  Scalar inputs also work.

Sources
-------
Quilez, I. (2023). "Signed Distance Functions." iquilezles.org/articles/distfunctions.
    Canonical reference for SDF primitives; all formulas below are derived
    from or consistent with this reference.
Timoshenko & Goodier (1951) §131 -- cylinder thin-shell (geometry only).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Sphere
# ---------------------------------------------------------------------------


def sphere(
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 1.0,
):
    """
    Exact SDF for a sphere.

        f(x, y, z) = sqrt((x-cx)^2 + (y-cy)^2 + (z-cz)^2) - radius

    Parameters
    ----------
    center : (cx, cy, cz)
    radius : float > 0

    Returns
    -------
    Callable f(x, y, z) -> ndarray
    """
    cx, cy, cz = center

    def _f(x, y, z):
        x_ = np.asarray(x, dtype=float)
        y_ = np.asarray(y, dtype=float)
        z_ = np.asarray(z, dtype=float)
        return np.sqrt((x_ - cx) ** 2 + (y_ - cy) ** 2 + (z_ - cz) ** 2) - radius

    return _f


# ---------------------------------------------------------------------------
# Infinite cylinder (axis aligned with z, centred at (cx, cy))
# ---------------------------------------------------------------------------


def cylinder_z(
    center_xy: tuple[float, float] = (0.0, 0.0),
    radius: float = 1.0,
):
    """
    Exact SDF for an infinite cylinder aligned with the z-axis.

        f(x, y, z) = sqrt((x-cx)^2 + (y-cy)^2) - radius

    Use operations.intersection with halfspaces to cap it.

    Parameters
    ----------
    center_xy : (cx, cy) -- axis passes through this point in the xy-plane
    radius : float > 0
    """
    cx, cy = center_xy

    def _f(x, y, z):
        x_ = np.asarray(x, dtype=float)
        y_ = np.asarray(y, dtype=float)
        return np.sqrt((x_ - cx) ** 2 + (y_ - cy) ** 2) - radius

    return _f


# ---------------------------------------------------------------------------
# Box
# ---------------------------------------------------------------------------


def box(
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    half_extents: tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    """
    Exact SDF for an axis-aligned box.

    Parameters
    ----------
    center : (cx, cy, cz)
    half_extents : (hx, hy, hz)  -- half-side lengths along each axis
    """
    cx, cy, cz = center
    hx, hy, hz = half_extents

    def _f(x, y, z):
        dx = np.abs(np.asarray(x, dtype=float) - cx) - hx
        dy = np.abs(np.asarray(y, dtype=float) - cy) - hy
        dz = np.abs(np.asarray(z, dtype=float) - cz) - hz
        outside = np.sqrt(
            np.maximum(dx, 0.0) ** 2
            + np.maximum(dy, 0.0) ** 2
            + np.maximum(dz, 0.0) ** 2
        )
        inside = np.minimum(np.maximum(dx, np.maximum(dy, dz)), 0.0)
        return outside + inside

    return _f


# ---------------------------------------------------------------------------
# Halfspace  (plane — one side is inside)
# ---------------------------------------------------------------------------


def halfspace(
    normal: tuple[float, float, float],
    offset: float,
):
    """
    Signed distance to a plane: f = n·p - d.

    The region n·p < d is inside (negative).

    Parameters
    ----------
    normal : unit normal vector (nx, ny, nz) pointing *outward*
    offset : d such that the plane is n·p = d
    """
    nx, ny, nz = normal
    mag = (nx**2 + ny**2 + nz**2) ** 0.5
    nx, ny, nz = nx / mag, ny / mag, nz / mag

    def _f(x, y, z):
        x_ = np.asarray(x, dtype=float)
        y_ = np.asarray(y, dtype=float)
        z_ = np.asarray(z, dtype=float)
        return nx * x_ + ny * y_ + nz * z_ - offset

    return _f
