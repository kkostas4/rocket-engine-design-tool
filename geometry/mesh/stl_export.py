"""
stl_export.py
=============

Write triangle meshes to binary STL files and read them back.

STL library choice: trimesh
----------------------------
trimesh was chosen over numpy-stl because:
  1. trimesh.Trimesh.is_watertight() is needed for manifold verification in
     the geometry test suite.
  2. trimesh's STL writer produces standard binary STL with correct normals.
  3. trimesh is actively maintained and handles degenerate meshes gracefully.

numpy-stl is still installed (pulled in with trimesh) and is an alternative
for environments where trimesh is not available.

Sources
-------
STL format:
    StereoLithography Interface Specification (3D Systems, 1989).
    Binary STL: 80-byte header, uint32 triangle count, then per-triangle:
    3 float32 normal, 9 float32 vertices, uint16 attribute byte count.

trimesh:
    Dawson-Haggerty, M. et al. (2019). trimesh. github.com/mikedh/trimesh.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh  # type: ignore[import-untyped]


def write_stl(
    verts: np.ndarray,
    faces: np.ndarray,
    path: str | Path,
    scale_to_mm: bool = False,
) -> Path:
    """
    Write a triangle mesh to a binary STL file.

    Parameters
    ----------
    verts : (N, 3) float array   Vertex positions [m] (or [mm] if already scaled).
    faces : (M, 3) int array     Triangle vertex indices.
    path : str or Path           Output file path.  Parent directory must exist.
    scale_to_mm : bool           If True, multiply all coordinates by 1000 before
                                 writing (converts metres to millimetres).
                                 Most FEA and slicing tools expect mm.

    Returns
    -------
    Path   Resolved path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int32)

    if scale_to_mm:
        v = v * 1000.0

    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    mesh.export(str(path), file_type="stl")
    return path


def read_stl(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a binary or ASCII STL file.

    Returns
    -------
    verts : (N, 3) float64  Vertex positions (in whatever units the file uses).
    faces : (M, 3) int32    Triangle vertex indices.
    """
    mesh = trimesh.load(str(path), force="mesh")
    return (
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int32),
    )


def is_watertight(verts: np.ndarray, faces: np.ndarray) -> bool:
    """
    Return True if the mesh is a closed manifold (watertight).

    A watertight mesh has no boundary edges and every edge is shared by
    exactly two triangles.  This is a necessary condition for a valid
    solid for printing or FEA meshing.

    Uses trimesh.Trimesh.is_watertight.
    """
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    return bool(mesh.is_watertight)
