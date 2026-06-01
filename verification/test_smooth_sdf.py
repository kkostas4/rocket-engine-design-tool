"""
test_smooth_sdf.py
==================

Verification tests for smooth-union / intersection / difference SDF ops.

Sources
-------
Quilez, I. (2023). "Smooth Minimum." iquilezles.org/articles/smin
    — defines the polynomial smooth-min blend and the expected properties.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from noyron.kernel.geometry.sdf.operations import (
    difference,
    smooth_difference,
    smooth_intersection,
    smooth_union,
    union,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sphere(cx: float, cy: float, cz: float, r: float):
    """Exact SDF of a sphere."""
    def _f(x, y, z):
        x_ = np.asarray(x, dtype=float)
        y_ = np.asarray(y, dtype=float)
        z_ = np.asarray(z, dtype=float)
        return np.sqrt((x_ - cx)**2 + (y_ - cy)**2 + (z_ - cz)**2) - r
    return _f


# ---------------------------------------------------------------------------
# Test 1: k=0 smooth_union equals hard union
# ---------------------------------------------------------------------------

def test_smooth_union_k0_equals_hard_union():
    """smooth_union(k=0) must be identical to union."""
    a = _sphere(0.0, 0.0, 0.0, 0.05)
    b = _sphere(0.03, 0.0, 0.0, 0.03)

    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.1, 0.1, size=(200, 3))
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    hard = union(a, b)(x, y, z)
    soft = smooth_union(a, b, k=0.0)(x, y, z)
    np.testing.assert_array_equal(hard, soft)


# ---------------------------------------------------------------------------
# Test 2: k=0 smooth_difference equals hard difference
# ---------------------------------------------------------------------------

def test_smooth_difference_k0_equals_hard_difference():
    """smooth_difference(k=0) must be identical to difference."""
    a = _sphere(0.0, 0.0, 0.0, 0.05)
    b = _sphere(0.0, 0.0, 0.0, 0.02)

    rng = np.random.default_rng(1)
    pts = rng.uniform(-0.1, 0.1, size=(200, 3))
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    hard = difference(a, b)(x, y, z)
    soft = smooth_difference(a, b, k=0.0)(x, y, z)
    np.testing.assert_array_equal(hard, soft)


# ---------------------------------------------------------------------------
# Test 3: smooth_union is continuous across the blend interface
# ---------------------------------------------------------------------------

def test_smooth_union_continuous():
    """smooth_union SDF must not jump discontinuously across the interface."""
    a = _sphere(0.0, 0.0, 0.0, 0.04)
    b = _sphere(0.06, 0.0, 0.0, 0.04)
    k = 0.005

    f = smooth_union(a, b, k=k)

    # Sample along the x-axis through the blend zone
    x = np.linspace(-0.1, 0.16, 1000)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    vals = f(x, y, z)

    # Max step between adjacent samples should be small (< 2 × sample spacing × 2)
    dx = x[1] - x[0]
    max_jump = np.max(np.abs(np.diff(vals)))
    assert max_jump < 2.0 * dx * 2.0, (
        f"smooth_union is discontinuous: max jump = {max_jump:.6f} m "
        f"(sample spacing = {dx:.6f} m)"
    )


# ---------------------------------------------------------------------------
# Test 4: smooth_union encloses strictly more volume than hard union
# ---------------------------------------------------------------------------

def test_smooth_union_adds_material_in_blend_region():
    """smooth_union with k>0 must add material vs hard union in the gap.

    Geometry: two r=0.03 spheres with centres 0.08 m apart, gap = 0.02 m.
    The midpoint (x=0.04) is equidistant (0.01 m) from both sphere surfaces,
    so hard_union = 0.01 > 0.  smooth_union fills the gap and is < 0.01.
    """
    a = _sphere(0.0, 0.0, 0.0, 0.03)
    b = _sphere(0.08, 0.0, 0.0, 0.03)
    k = 0.008

    # Midpoint of the gap between the two sphere surfaces
    x_mid = np.array([0.04])
    y_mid = np.zeros(1)
    z_mid = np.zeros(1)

    hard_val = union(a, b)(x_mid, y_mid, z_mid)[0]
    smooth_val = smooth_union(a, b, k=k)(x_mid, y_mid, z_mid)[0]

    assert hard_val > 0.0, f"midpoint should be outside both spheres; got {hard_val}"
    assert smooth_val < hard_val, (
        f"smooth_union ({smooth_val:.6f}) should be < hard union ({hard_val:.6f}) "
        "in the blend region"
    )


# ---------------------------------------------------------------------------
# Test 5: orifice cutouts are SHARP (k=0 smooth_difference preserves void)
# ---------------------------------------------------------------------------

def test_orifice_bores_preserved_sharp():
    """
    When channel_difference/orifice_difference = 0, the void boundary must
    match the hard-boolean result exactly.  This guards against accidentally
    using k>0 for those interfaces.
    """
    # Face plate ring (solid annulus)
    outer = _sphere(0.0, 0.0, 0.0, 0.022)  # ~44 mm chamber radius proxy

    # Orifice bore (small sphere standing in for a cylinder)
    orifice = _sphere(0.015, 0.0, 0.0, 0.00041)  # 0.82 mm diameter ≈ r=0.41 mm

    k_orifice = 0.0  # MUST be sharp

    hard = difference(outer, orifice)
    soft = smooth_difference(outer, orifice, k=k_orifice)

    rng = np.random.default_rng(2)
    pts = rng.uniform(-0.025, 0.025, size=(500, 3))
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    np.testing.assert_array_equal(
        hard(x, y, z), soft(x, y, z),
        err_msg="smooth_difference(k=0) diverged from hard difference — orifice bore would be rounded",
    )


# ---------------------------------------------------------------------------
# Test 6: smooth ops are isolated to noyron.kernel (no rp imports needed)
# ---------------------------------------------------------------------------

def test_smooth_ops_importable_from_kernel_only():
    """
    The operations module must be importable without any rp sub-package.
    This guards against circular-import regressions.
    """
    import importlib
    mod = importlib.import_module("noyron.kernel.geometry.sdf.operations")
    assert hasattr(mod, "smooth_union")
    assert hasattr(mod, "smooth_intersection")
    assert hasattr(mod, "smooth_difference")
    assert callable(mod.smooth_union)
    assert callable(mod.smooth_intersection)
    assert callable(mod.smooth_difference)
