"""
test_geometry.py
================

Verification tests for geometry/ -- SDF kernel, mesh extraction,
STL export, and chamber+nozzle surface generation.

Sources
-------
    Quilez, I. (2023). "Signed Distance Functions."
        iquilezles.org/articles/distfunctions.
    Lorensen & Cline (1987). "Marching Cubes." SIGGRAPH '87.
    Rao (1958). Parabolic bell nozzle.
"""

from __future__ import annotations

import math
import os
import tempfile
import warnings

import numpy as np

from noyron.kernel.geometry.sdf.operations import difference, intersection, union
from noyron.kernel.geometry.sdf.primitives import cylinder_z, sphere
from noyron.kernel.geometry.sdf.revolution import revolve


# ---------------------------------------------------------------------------
# 1. Sphere SDF — exact distance
# ---------------------------------------------------------------------------
class TestSphereSDF:
    def test_distance_to_origin(self) -> None:
        """
        sphere at origin: f(p) = |p| - r.

        Source: Quilez (2023), exact sphere SDF.
        """
        f = sphere(center=(0.0, 0.0, 0.0), radius=1.0)
        # Point at radius 1.5 -> distance = 0.5
        assert abs(float(f(1.5, 0.0, 0.0)) - 0.5) < 1e-9
        # Point at radius 0.5 -> distance = -0.5 (inside)
        assert abs(float(f(0.5, 0.0, 0.0)) - (-0.5)) < 1e-9
        # Point on surface -> distance = 0
        assert abs(float(f(1.0, 0.0, 0.0))) < 1e-9

    def test_distance_from_offset_center(self) -> None:
        """sphere at (1,2,3): distance from (1,2,4) = 1 - r."""
        f = sphere(center=(1.0, 2.0, 3.0), radius=0.5)
        d = float(f(1.0, 2.0, 4.0))
        assert abs(d - 0.5) < 1e-9

    def test_numpy_array_input(self) -> None:
        """sphere SDF accepts numpy array inputs."""
        f = sphere(center=(0.0, 0.0, 0.0), radius=1.0)
        x = np.array([0.0, 1.5, -2.0])
        y = np.zeros(3)
        z = np.zeros(3)
        result = f(x, y, z)
        expected = np.array([-1.0, 0.5, 1.0])
        np.testing.assert_allclose(result, expected, atol=1e-9)

    def test_inside_negative(self) -> None:
        """Points inside the sphere have negative SDF."""
        f = sphere(radius=1.0)
        assert float(f(0.0, 0.0, 0.0)) < 0.0

    def test_outside_positive(self) -> None:
        """Points outside the sphere have positive SDF."""
        f = sphere(radius=1.0)
        assert float(f(2.0, 0.0, 0.0)) > 0.0


# ---------------------------------------------------------------------------
# 2. SDF operations — union, intersection, difference
# ---------------------------------------------------------------------------
class TestSDFOperations:
    def test_union_inside_either(self) -> None:
        """Union: point inside either sphere is inside the union."""
        a = sphere(center=(-1.0, 0.0, 0.0), radius=0.8)
        b = sphere(center=(+1.0, 0.0, 0.0), radius=0.8)
        ab = union(a, b)
        # Point at (-1,0,0): inside a, outside b -> inside union
        assert float(ab(-1.0, 0.0, 0.0)) < 0.0
        assert float(ab(+1.0, 0.0, 0.0)) < 0.0

    def test_intersection_inside_both(self) -> None:
        """Intersection: point is inside only when inside both shapes.

        a = sphere at origin r=1.5; b = sphere at (1,0,0) r=1.5.
        (-1,0,0): inside a (dist=1 < 1.5), outside b (dist=2 > 1.5) -> OUTSIDE.
        (0.5,0,0): inside a (dist=0.5), inside b (dist=0.5)           -> INSIDE.
        """
        a = sphere(center=(0.0, 0.0, 0.0), radius=1.5)
        b = sphere(center=(1.0, 0.0, 0.0), radius=1.5)
        ab = intersection(a, b)
        # (-1, 0, 0): inside a, outside b -> outside intersection
        assert float(ab(-1.0, 0.0, 0.0)) > 0.0, (
            "(-1,0,0) is inside a but outside b; intersection must be positive"
        )
        # (0.5, 0, 0): inside both -> inside intersection
        assert float(ab(0.5, 0.0, 0.0)) < 0.0, (
            "(0.5,0,0) is inside both spheres; intersection must be negative"
        )

    def test_difference_removes_b_from_a(self) -> None:
        """Difference: point inside a but outside b is inside a-b."""
        a = sphere(center=(0.0, 0.0, 0.0), radius=2.0)
        b = sphere(center=(0.0, 0.0, 0.0), radius=1.0)
        a_minus_b = difference(a, b)
        # (1.5, 0, 0): inside a, outside b -> inside difference
        assert float(a_minus_b(1.5, 0.0, 0.0)) < 0.0
        # (0.0, 0, 0): inside both -> outside difference
        assert float(a_minus_b(0.0, 0.0, 0.0)) > 0.0


# ---------------------------------------------------------------------------
# 3. revolution.revolve recovers exact cylinder SDF on the lateral surface
# ---------------------------------------------------------------------------
class TestRevolveCylinder:
    def test_revolution_recovers_cylinder(self) -> None:
        """
        Revolving a constant r(z) = R profile produces a cylinder whose
        SDF on the lateral surface matches cylinder_z() exactly.

        Source: revolve() docstring; cylinder_z() definition.
        The revolution SDF for constant r is f_wall = rho - R, which is the
        same as the infinite cylinder SDF from cylinder_z().
        """
        R = 0.05  # 50 mm radius
        z_min, z_max = -0.1, 0.1

        r_const = lambda z: np.full_like(np.asarray(z, dtype=float), R)
        f_rev = revolve(r_const, z_min, z_max)
        f_cyl = cylinder_z(center_xy=(0.0, 0.0), radius=R)

        # Test points strictly inside z range -> f_wall governs
        test_rho = np.array([0.0, 0.03, 0.05, 0.07])
        test_z = np.zeros_like(test_rho)
        test_x = test_rho
        test_y = np.zeros_like(test_rho)

        rev_vals = f_rev(test_x, test_y, test_z)
        cyl_vals = f_cyl(test_x, test_y, test_z)

        np.testing.assert_allclose(rev_vals, cyl_vals, atol=1e-6, err_msg=(
            "Revolution of constant r(z) should match cylinder_z SDF "
            "on the lateral surface"
        ))

    def test_revolution_caps_correctly(self) -> None:
        """
        Points outside the z range are classified as outside the solid.
        """
        R = 0.05
        z_min, z_max = 0.0, 0.1
        r_const = lambda z: np.full_like(np.asarray(z, dtype=float), R)
        f = revolve(r_const, z_min, z_max)

        # Point directly below the bottom cap, on the axis
        assert float(f(0.0, 0.0, -0.01)) > 0.0
        # Point directly above the top cap
        assert float(f(0.0, 0.0, 0.11)) > 0.0
        # Point on axis, inside z range -> inside solid
        assert float(f(0.0, 0.0, 0.05)) < 0.0


# ---------------------------------------------------------------------------
# 4. Marching cubes on a sphere — vertex distances close to radius
# ---------------------------------------------------------------------------
class TestMarchingCubesSphere:
    def test_marching_cubes_sphere_vertices(self) -> None:
        """
        Extract a sphere of radius r = 0.05 m.
        All vertices should be at distance r ± 1 voxel from the centre.

        Source: Lorensen & Cline (1987) — marching cubes preserves surface
        location to within ~1 voxel.
        """
        from noyron.kernel.geometry.mesh.extractor import sdf_to_mesh

        r = 0.05
        voxel = 0.005  # 5 mm voxels for speed in CI
        margin = 0.01

        f = sphere(center=(0.0, 0.0, 0.0), radius=r)
        bbox = (-(r + margin), r + margin,
                -(r + margin), r + margin,
                -(r + margin), r + margin)

        verts, faces = sdf_to_mesh(f, bbox, voxel_size=voxel)
        dists = np.linalg.norm(verts, axis=1)

        tol = 1.5 * voxel  # allow 1.5 voxel tolerance
        assert dists.min() > r - tol, (
            f"Min vertex distance {dists.min():.4f} < r - tol ({r - tol:.4f})"
        )
        assert dists.max() < r + tol, (
            f"Max vertex distance {dists.max():.4f} > r + tol ({r + tol:.4f})"
        )

    def test_marching_cubes_sphere_connectivity(self) -> None:
        """Extracted sphere mesh has positive vertex and face counts."""
        from noyron.kernel.geometry.mesh.extractor import sdf_to_mesh

        r = 0.05
        voxel = 0.005
        f = sphere(radius=r)
        bbox = (-0.07, 0.07, -0.07, 0.07, -0.07, 0.07)
        verts, faces = sdf_to_mesh(f, bbox, voxel_size=voxel)

        assert len(verts) > 0, "No vertices extracted"
        assert len(faces) > 0, "No faces extracted"
        assert faces.shape[1] == 3, "Faces must be triangles"


# ---------------------------------------------------------------------------
# 5. Chamber+nozzle volume sanity check
# ---------------------------------------------------------------------------
class TestChamberNozzleVolumeSanity:
    def test_chamber_nozzle_volume_within_5pct(self) -> None:
        """
        Extract a simplified chamber+nozzle at low resolution, compare
        the mesh volume to the analytical volume of revolution.

        Tolerance 5 % accounts for marching-cubes staircasing error.
        Uses a small test nozzle (R_t=10 mm, eps=4) to keep runtime short.

        Source: Pappus centroid theorem — volume of revolution
            V = pi * integral(r(z)^2, z_min, z_max).
        """
        from geometry.components.chamber_nozzle import chamber_nozzle_sdf
        from noyron.kernel.geometry.mesh.extractor import sdf_to_mesh, adaptive_bbox
        from physics.nozzle.rao import RaoNozzle
        from physics.structural.bilayer import (
            BilayerWall, WallLayer, RibGeometry,
        )
        from physics.structural.materials import GRCOP84_PROXY, INCONEL718_STRUCT
        from physics.regen.coolant import CoolingChannelGeometry

        # Small nozzle for speed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nozzle = RaoNozzle(r_throat=0.010, epsilon=4.0)

        # Thin constant wall (keeps geometry simple)
        t_lin = 0.5e-3
        h_ch = 1.0e-3
        t_jkt = 1.0e-3

        geom = CoolingChannelGeometry(
            n_channels=40,
            width=lambda x: 0.5e-3,
            height=lambda x: h_ch,
            wall_thickness=lambda x: t_lin,
            land_thickness=lambda x: 0.3e-3,
        )
        bilayer = BilayerWall(
            inner_liner=WallLayer(
                thickness=lambda x: t_lin,
                material=GRCOP84_PROXY,
            ),
            outer_jacket=WallLayer(
                thickness=lambda x: t_jkt,
                material=INCONEL718_STRUCT,
            ),
            ribs=RibGeometry(
                n_ribs=40,
                land_width=lambda x: 0.3e-3,
                height=lambda x: h_ch,
                material=GRCOP84_PROXY,
            ),
            channel_geometry=geom,
        )

        L_ch = 0.05   # 50 mm chamber

        sdf, profile = chamber_nozzle_sdf(
            rao_nozzle=nozzle,
            bilayer=bilayer,
            L_chamber=L_ch,
        )

        # Analytical volume = pi * integral(r_outer^2, z_min, z_max)
        z_ax = profile["z_ax"]
        r_out = profile["r_outer"]
        V_analytical = math.pi * float(np.trapezoid(r_out**2, z_ax))

        # Extract mesh at coarse resolution for speed
        voxel = 1.0e-3  # 1 mm
        bbox = adaptive_bbox(
            nozzle_x_min=profile["z_min"],
            nozzle_x_max=profile["z_max"],
            r_outer_max=profile["r_outer_max"],
            radial_margin=2e-3,
            axial_margin=2e-3,
        )
        verts, faces = sdf_to_mesh(sdf, bbox, voxel_size=voxel)

        import trimesh  # type: ignore[import-untyped]
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
        V_mesh = float(abs(mesh.volume))

        rel_err = abs(V_mesh - V_analytical) / V_analytical
        assert rel_err < 0.05, (
            f"Mesh volume {V_mesh*1e6:.1f} cm³ vs analytical {V_analytical*1e6:.1f} cm³ "
            f"(rel err = {rel_err*100:.1f} % > 5 %)"
        )


# ---------------------------------------------------------------------------
# 6. STL export roundtrip
# ---------------------------------------------------------------------------
class TestSTLExportRoundtrip:
    def test_stl_export_roundtrip(self) -> None:
        """
        Write a sphere mesh to STL, read it back, confirm same vertex count.

        Source: STL binary format (3D Systems 1989); trimesh I/O.
        """
        from noyron.kernel.geometry.mesh.extractor import sdf_to_mesh
        from noyron.kernel.geometry.mesh.stl_export import read_stl, write_stl

        f = sphere(radius=0.05)
        bbox = (-0.07, 0.07, -0.07, 0.07, -0.07, 0.07)
        verts, faces = sdf_to_mesh(f, bbox, voxel_size=0.005)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sphere.stl")
            write_stl(verts, faces, path, scale_to_mm=False)
            assert os.path.isfile(path), "STL file not created"
            assert os.path.getsize(path) > 0, "STL file is empty"

            v2, f2 = read_stl(path)
            # trimesh may merge duplicate vertices; check face count matches
            assert len(f2) == len(faces), (
                f"Round-trip face count: wrote {len(faces)}, read {len(f2)}"
            )

    def test_stl_scale_to_mm(self) -> None:
        """
        With scale_to_mm=True, vertex coordinates are 1000× larger.
        """
        from noyron.kernel.geometry.mesh.extractor import sdf_to_mesh
        from noyron.kernel.geometry.mesh.stl_export import read_stl, write_stl

        r = 0.05  # metres
        f = sphere(radius=r)
        bbox = (-0.07, 0.07, -0.07, 0.07, -0.07, 0.07)
        verts_m, faces = sdf_to_mesh(f, bbox, voxel_size=0.005)

        with tempfile.TemporaryDirectory() as tmp:
            path_mm = os.path.join(tmp, "sphere_mm.stl")
            write_stl(verts_m, faces, path_mm, scale_to_mm=True)
            v_mm, _ = read_stl(path_mm)

            # Max vertex distance should be ~50 mm (= 0.05 m × 1000)
            dists_mm = np.linalg.norm(v_mm, axis=1)
            assert dists_mm.max() > 45.0, (
                f"Max vertex dist = {dists_mm.max():.1f} mm, expected ~50 mm"
            )
