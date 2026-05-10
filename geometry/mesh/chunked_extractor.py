"""
chunked_extractor.py
====================

Chunked marching-cubes extraction that processes the SDF in axial slabs
to keep peak memory within bounds.

Motivation
----------
At 0.1 mm voxels the full chamber + nozzle envelope is roughly:

    Radial: 2 × 30 mm / 0.1 mm ≈ 600 voxels per radial axis
    Axial:  220 mm / 0.1 mm     ≈ 2200 voxels

    Single-pass grid: 600 × 600 × 2200 × 4 bytes ≈ 3.2 GB  (too large)

This module splits the axial range into ``slab_axial_steps`` steps, extracts
each slab independently (peak ~320 MB per slab), then concatenates the partial
meshes and welds the seam vertices with trimesh.

Slab overlap
------------
Adjacent slabs overlap by one voxel so the marching-cubes triangles at the
seam are identical on both sides.  The trimesh ``process=True`` flag merges
duplicate vertices and removes degenerate faces after concatenation.

Sources
-------
Lorensen, W.E. and Cline, H.E. (1987). "Marching Cubes." SIGGRAPH '87.

trimesh: Dawson-Haggerty, M. et al. (2019). trimesh v3.
    github.com/mikedh/trimesh

scikit-image marching_cubes:
    van der Walt, S. et al. (2014). PeerJ, 2:e453.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np

from geometry.mesh.extractor import sdf_to_mesh


def chunked_sdf_to_mesh(
    sdf: Callable,
    bbox: tuple[float, float, float, float, float, float],
    voxel_size: float = 1e-4,
    slab_axial_steps: int = 100,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the zero-isosurface of an SDF using axial slab decomposition.

    Splits the bounding box into ``n_slabs`` axial chunks, runs
    ``sdf_to_mesh`` on each slab, then concatenates and welds the results
    with trimesh.

    Parameters
    ----------
    sdf : callable
        f(x, y, z) -> ndarray.  Same convention as ``sdf_to_mesh``.
    bbox : (x_min, x_max, y_min, y_max, z_min, z_max)
        Full bounding box of the volume [m].
    voxel_size : float
        Grid spacing [m].  Default 0.1 mm.
    slab_axial_steps : int
        Number of axial voxels per slab.  Controls memory:
        peak ≈ (Nx × Ny × slab_axial_steps) × 4 bytes.
        Default 100 (≈ 10 mm at 0.1 mm voxels).
    verbose : bool
        If True, print progress per slab.

    Returns
    -------
    verts : ndarray, shape (N, 3), float64  [m]
    faces : ndarray, shape (M, 3), int32

    Notes
    -----
    Slabs overlap by one voxel on the shared boundary.  After concatenation,
    trimesh ``process=True`` merges duplicate seam vertices and removes
    zero-area triangles, producing a connected mesh.
    """
    import trimesh  # type: ignore[import-untyped]

    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    slab_thickness = voxel_size * slab_axial_steps

    # Build slab z-boundaries with one-voxel overlap at each seam
    z_starts = np.arange(z_min, z_max, slab_thickness)
    n_slabs = len(z_starts)

    if verbose:
        nx = round((x_max - x_min) / voxel_size)
        ny = round((y_max - y_min) / voxel_size)
        mem_mb = nx * ny * (slab_axial_steps + 1) * 4 / 1e6
        print(
            f"  Chunked extraction: {n_slabs} slabs × {slab_axial_steps} axial "
            f"steps × {voxel_size*1e3:.2f} mm voxels  "
            f"(~{mem_mb:.0f} MB peak per slab)"
        )

    partial_meshes = []

    for i, z_s in enumerate(z_starts):
        z_e = min(z_s + slab_thickness + voxel_size, z_max)  # one-voxel overlap
        slab_bbox = (x_min, x_max, y_min, y_max, z_s, z_e)

        if verbose:
            print(
                f"  Slab {i + 1}/{n_slabs}:  z = [{z_s*1e3:.1f}, {z_e*1e3:.1f}] mm",
                end="",
                flush=True,
            )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                verts_s, faces_s = sdf_to_mesh(sdf, slab_bbox, voxel_size=voxel_size)

            partial_meshes.append(
                trimesh.Trimesh(vertices=verts_s, faces=faces_s, process=False)
            )
            if verbose:
                print(f"  → {len(verts_s):,} verts, {len(faces_s):,} faces")

        except ValueError as exc:
            # Slab entirely inside solid or entirely outside — skip silently
            if verbose:
                print(f"  (skipped: {exc})")
            continue

    if not partial_meshes:
        raise ValueError(
            "No surface found in any slab.  Check the SDF and bounding box."
        )

    if verbose:
        print("  Welding slab seams ...", flush=True)

    combined = trimesh.util.concatenate(partial_meshes)
    combined = combined.process()  # merge duplicate verts, remove degenerate faces

    verts_out = np.array(combined.vertices, dtype=np.float64)
    faces_out = np.array(combined.faces, dtype=np.int32)

    if verbose:
        print(
            f"  Final mesh: {len(verts_out):,} verts, {len(faces_out):,} faces  "
            f"(watertight={combined.is_watertight})"
        )

    return verts_out, faces_out
