"""
operations.py
=============

Boolean and blending operations on SDF callables.

All operations return a new callable f(x, y, z) -> ndarray.

Sources
-------
Quilez, I. (2023). "Signed Distance Functions." iquilezles.org/articles/distfunctions.
    Standard SDF boolean algebra and smooth-blend formulas.
"""

from __future__ import annotations

import numpy as np


def union(a, b):
    """
    SDF union (logical OR): the set of points inside either a or b.

        f = min(a(p), b(p))

    Source: Quilez (2023), union.
    """
    return lambda x, y, z: np.minimum(a(x, y, z), b(x, y, z))


def intersection(a, b):
    """
    SDF intersection (logical AND): the set of points inside both a and b.

        f = max(a(p), b(p))

    Source: Quilez (2023), intersection.
    """
    return lambda x, y, z: np.maximum(a(x, y, z), b(x, y, z))


def difference(a, b):
    """
    SDF difference (a minus b): points inside a but not inside b.

        f = max(a(p), -b(p))

    Source: Quilez (2023), subtraction.
    """
    return lambda x, y, z: np.maximum(a(x, y, z), -b(x, y, z))


def smooth_union(a, b, k: float = 0.01):
    """
    Smooth SDF union with blend radius k [m].

    Blends the two shapes over a transition zone of radius k, avoiding the
    C0 discontinuity of the hard union.  Useful for fillet-like blending.

        h = clip(0.5 + 0.5*(b-a)/k, 0, 1)
        f = mix(b, a, h) - k*h*(1-h)

    Source: Quilez (2023), smooth minimum.

    Parameters
    ----------
    k : float
        Blend radius [m].  Larger k = smoother but more distorted SDF.
    """

    def _f(x, y, z):
        da = a(x, y, z)
        db = b(x, y, z)
        h = np.clip(0.5 + 0.5 * (db - da) / k, 0.0, 1.0)
        return db * (1.0 - h) + da * h - k * h * (1.0 - h)

    return _f


def smooth_difference(a, b, k: float = 0.01):
    """
    Smooth SDF difference (a minus b) with blend radius k.

    Source: Quilez (2023), smooth subtraction.
    """

    def _f(x, y, z):
        da = a(x, y, z)
        db = -b(x, y, z)
        h = np.clip(0.5 - 0.5 * (da + db) / k, 0.0, 1.0)
        return da * (1.0 - h) + db * h + k * h * (1.0 - h)

    return _f
