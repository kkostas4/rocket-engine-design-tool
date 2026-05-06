"""
extractor.py
============

Sample an SDF on a regular voxel grid and extract the zero-isosurface
using the Marching Cubes algorithm (Lorensen & Cline 1987).

Memory budget
-------------
A grid of N_x × N_y × N_z float32 values costs 4 * N_x * N_y * N_z bytes.
For a 200 mm long chamber with 0.5 mm voxels across a 60 mm diameter:

    N_x = N_y ≈ 60 mm / 0.5 mm = 120
    N_z ≈ 200 mm / 0.5 mm = 400
    Grid = 4 × 120 × 120 × 400 ≈ 23 MB  (float32)

This is well within typical available memory.  Double the voxel count in
each dimension (0.25 mm) → 8× memory = 184 MB.  Flag if going below 0.25 mm.

Performance
-----------
Grid evaluation is vectorized (single numpy call).  Marching cubes from
scikit-image is a compiled C extension and is fast for grids < 500^3.
For very high resolutions (> 1000^3) consider dual contouring or surface
nets, but those are out of scope for this first pass.

Sources
-------
Lorensen, W.E. and Cline, H.E. (1987). "Marching Cubes: A High Resolution
    3D Surface Construction Algorithm." SIGGRAPH '87 Proceedings,
    Vol. 21, No. 4, pp. 163-169.

scikit-image marching_cubes:
    van der Walt, S. et al. (2014). "scikit-image: image processing in
    Python." PeerJ, 2:e453.  https://doi.org/10.7717/peerj.453.
"""

from __future__ import annotations

import warnings

import numpy as np
from skimage.measure import marching_cubes  # type: ignore[import-untyped]


def sdf_to_mesh(
    sdf_func,
    bbox: tuple[float, float, float, float, float, float],
    voxel_size: float = 5e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the zero-isosurface of an SDF as a triangle mesh.

    Samples the SDF on a regular voxel grid covering ``bbox``, then runs
    Marching Cubes at level = 0 to extract the surface.

    Parameters
    ----------
    sdf_func : callable
        f(x, y, z) -> ndarray.  Must accept broadcastable numpy arrays.
        Convention: negative inside, positive outside.
    bbox : (x_min, x_max, y_min, y_max, z_min, z_max)
        Bounding box of the region to evaluate [m].
        The SDF must be negative at least somewhere inside this box.
    voxel_size : float
        Grid spacing [m].  Default 0.5 mm.
        Memory cost: (dx/ds) × (dy/ds) × (dz/ds) × 4 bytes (float32).
    Returns
    -------
    verts : ndarray, shape (N, 3), float64  [m]
        Vertex positions in world coordinates.
    faces : ndarray, shape (M, 3), int
        Triangle indices into verts.

    Raises
    ------
    ValueError
        If marching cubes finds no surface (SDF never crosses zero in the box).
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bbox

    # Build the grid.  Use arange so the spacing is exactly voxel_size.
    xs = np.arange(x_min, x_max + voxel_size * 0.5, voxel_size, dtype=np.float32)
    ys = np.arange(y_min, y_max + voxel_size * 0.5, voxel_size, dtype=np.float32)
    zs = np.arange(z_min, z_max + voxel_size * 0.5, voxel_size, dtype=np.float32)

    nx, ny, nz = len(xs), len(ys), len(zs)
    mem_mb = nx * ny * nz * 4 / 1e6
    if mem_mb > 500:
        warnings.warn(
            f"SDF grid is {nx}×{ny}×{nz} = {mem_mb:.0f} MB (float32). "
            f"Consider increasing voxel_size.",
            ResourceWarning,
            stacklevel=2,
        )

    # Evaluate SDF on the full grid in one vectorized call.
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    grid = sdf_func(X, Y, Z).astype(np.float32)

    if grid.min() >= 0.0:
        raise ValueError(
            "SDF is non-negative everywhere in the bounding box — "
            "no surface found.  Check the bbox or the SDF."
        )
    if grid.max() <= 0.0:
        raise ValueError(
            "SDF is non-positive everywhere in the bounding box — "
            "the entire box is inside the solid.  Expand the bbox."
        )

    # Run marching cubes.  spacing=(ds, ds, ds) converts voxel indices to metres.
    verts_vox, faces, _, _ = marching_cubes(
        grid,
        level=0.0,
        spacing=(voxel_size, voxel_size, voxel_size),
    )

    # Translate from voxel-space origin to world-space origin.
    verts = verts_vox + np.array([x_min, y_min, z_min], dtype=np.float64)

    return verts.astype(np.float64), faces.astype(np.int32)


def adaptive_bbox(
    nozzle_x_min: float,
    nozzle_x_max: float,
    r_outer_max: float,
    radial_margin: float = 5e-3,
    axial_margin: float = 5e-3,
) -> tuple[float, float, float, float, float, float]:
    """
    Build a bounding box that snugly wraps the chamber+nozzle outer surface.

    The bounding box is:
        x ∈ [-(r_outer_max + margin), +(r_outer_max + margin)]
        y ∈ [-(r_outer_max + margin), +(r_outer_max + margin)]
        z ∈ [nozzle_x_min - margin, nozzle_x_max + margin]

    In the geometry kernel the axial axis is z; the radial axes are x, y.

    Parameters
    ----------
    nozzle_x_min : float   Start of the axial domain [m] (typically < 0).
    nozzle_x_max : float   End of the axial domain [m].
    r_outer_max : float    Maximum outer radius over the full profile [m].
    radial_margin : float  Padding in x/y [m].  Default 5 mm.
    axial_margin : float   Padding in z [m].  Default 5 mm.

    Returns
    -------
    (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    rp = r_outer_max + radial_margin
    return (
        -rp, rp,
        -rp, rp,
        nozzle_x_min - axial_margin,
        nozzle_x_max + axial_margin,
    )
