"""
transforms.py
=============

Rigid-body and scaling transforms for SDF callables.

All transforms return a new callable f(x, y, z) -> ndarray.

Note on scaling: scaling a SDF by factor s requires dividing the returned
distance by |s| to keep the SDF metric (Quilez 2023).  Only uniform scaling
preserves the exact SDF property.

Source: Quilez, I. (2023). "Signed Distance Functions — Transforms."
        iquilezles.org/articles/distfunctions.
"""

from __future__ import annotations

import numpy as np


def translate(sdf, offset: tuple[float, float, float]):
    """
    Translate an SDF by a vector offset (tx, ty, tz).

        f_translated(x, y, z) = f(x - tx, y - ty, z - tz)

    Parameters
    ----------
    sdf : callable
    offset : (tx, ty, tz) [m]
    """
    tx, ty, tz = offset
    return lambda x, y, z: sdf(
        np.asarray(x, dtype=float) - tx,
        np.asarray(y, dtype=float) - ty,
        np.asarray(z, dtype=float) - tz,
    )


def scale_uniform(sdf, factor: float):
    """
    Uniformly scale an SDF about the origin.

        f_scaled(x, y, z) = f(x/s, y/s, z/s) * s

    Dividing by s gives the correct metric distance in scaled space.

    Parameters
    ----------
    factor : float > 0
    """
    if factor <= 0.0:
        raise ValueError(f"scale factor must be positive, got {factor}")
    return lambda x, y, z: sdf(
        np.asarray(x, dtype=float) / factor,
        np.asarray(y, dtype=float) / factor,
        np.asarray(z, dtype=float) / factor,
    ) * factor


def rotate_z(sdf, angle_rad: float):
    """
    Rotate an SDF about the z-axis by angle_rad [rad].

    Applies the inverse rotation to the query point (standard SDF transform
    approach: transform the point, not the shape).

    Parameters
    ----------
    angle_rad : float   Rotation angle [radians], right-hand positive.
    """
    c = float(np.cos(-angle_rad))
    s = float(np.sin(-angle_rad))

    def _f(x, y, z):
        x_ = np.asarray(x, dtype=float)
        y_ = np.asarray(y, dtype=float)
        return sdf(c * x_ - s * y_, s * x_ + c * y_, z)

    return _f
