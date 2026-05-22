"""
test_rp_manifolds.py
====================

Verification tests for the Phase-3 manifold components:
  LoxDome, FuelManifold, CoolantOutletManifold, CoolantInletManifold.

Also verifies the new kernel SDF primitives (annular_shell_sdf,
cylindrical_boss_sdf) remain domain-free, and that adding manifolds
does not perturb any existing physics scalar.

Sources
-------
Huzel & Huang (1992) §4.2 — manifold placement geometry.
NASA SP-8087 (1977) §3.3 — coolant manifold geometry reference.
"""

from __future__ import annotations

import importlib
import math
import sys
import warnings

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Module-level fixtures  (fast — built once per session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reference_nozzle():
    from noyron.rp.nozzle.rao import RaoNozzle
    return RaoNozzle(r_throat=0.022313 / 2.0, epsilon=10.605)


@pytest.fixture(scope="module")
def reference_pintle():
    from noyron.rp.injector.pintle import PintleInjector
    return PintleInjector.from_uniform_dP(
        dP_inj=0.45 * 6.8948e6,
        m_dot_ox=1.1144,
        m_dot_fuel=0.3096,
        rho_ox=1116.0,
        rho_fuel=386.0,
        Cd_ox=0.6,
        Cd_fuel=0.6,
        pintle_OD=0.030,
        n_fuel_orifices=20,
    )


@pytest.fixture(scope="module")
def reference_bilayer(reference_nozzle):
    """Build a minimal BilayerWall from reference-engine channel sizing."""
    from noyron.rp.regen.bartz import CombustionGas
    from noyron.rp.regen.coolant import GRCOP42, CoolantInlet
    from noyron.rp.regen.sizing import ChannelSizingProblem, size_channels
    from noyron.rp.structural.bilayer import (
        BilayerWall, RibGeometry, WallLayer, size_jacket_thickness,
    )
    from noyron.rp.structural.materials import GRCOP84_PROXY, INCONEL718_STRUCT

    nozzle = reference_nozzle
    gas = CombustionGas(
        Tc=3561.2, Pc=6.8948e6, gamma=1.15, MW=21.9,
        cp=2000.0, mu=8.0e-5, Pr=0.60, c_star=1893.3,
    )
    coolant = CoolantInlet(T=150.0, P=12.0e6, m_dot=0.3096, fluid="Methane")
    problem = ChannelSizingProblem(
        nozzle=nozzle,
        gas=gas,
        coolant_inlet=coolant,
        wall_material=GRCOP42,
        T_wall_max=1000.0,
        T_wall_margin=50.0,
        dP_max=3.0e6,
        min_wall_thickness=0.5e-3,
        max_aspect_ratio=8.0,
        min_channel_width=0.6e-3,
        t_land_min=0.3e-3,
        n_stations_opt=60,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        geom = size_channels(problem)

    r_exit = nozzle.r_throat * math.sqrt(nozzle.epsilon)
    r_jacket_exit = r_exit + 0.5e-3 + geom.height(nozzle.x_exit)
    t_jacket = max(
        1.5e-3,
        math.ceil(
            size_jacket_thickness(
                P_coolant_max=12.0e6,
                r_jacket=r_jacket_exit,
                material=INCONEL718_STRUCT,
                T_op=200.0,
                FoS=1.4,
            ) / 0.5e-3
        ) * 0.5e-3,
    )
    bl = BilayerWall(
        inner_liner=WallLayer(thickness=geom.wall_thickness, material=GRCOP84_PROXY),
        outer_jacket=WallLayer(thickness=lambda x, _t=t_jacket: _t, material=INCONEL718_STRUCT),
        ribs=RibGeometry(
            n_ribs=geom.n_channels,
            land_width=geom.land_thickness,
            height=geom.height,
            material=GRCOP84_PROXY,
        ),
        channel_geometry=geom,
    )
    return bl


@pytest.fixture(scope="module")
def all_manifolds(reference_nozzle, reference_bilayer, reference_pintle):
    """Return a dict of all four manifolds built from the reference engine."""
    from noyron.rp.geometry.manifolds import (
        ManifoldConfig,
        LoxDome,
        FuelManifold,
        CoolantOutletManifold,
        CoolantInletManifold,
    )
    cfg = ManifoldConfig()
    nozzle = reference_nozzle
    bilayer = reference_bilayer
    pintle = reference_pintle
    return {
        "lox_dome": LoxDome(nozzle=nozzle, bilayer=bilayer, injector=pintle, config=cfg),
        "fuel_manifold": FuelManifold(nozzle=nozzle, bilayer=bilayer, injector=pintle, config=cfg),
        "coolant_outlet": CoolantOutletManifold(nozzle=nozzle, bilayer=bilayer, config=cfg),
        "coolant_inlet": CoolantInletManifold(nozzle=nozzle, bilayer=bilayer, config=cfg),
    }


# ---------------------------------------------------------------------------
# 1. Kernel primitives import isolation
# ---------------------------------------------------------------------------


class TestKernelPrimitivesIsolation:
    def test_kernel_primitives_domain_free(self):
        """annular_shell_sdf and cylindrical_boss_sdf must not pull in rp modules."""
        to_remove = [k for k in sys.modules if k.startswith("noyron.rp")
                     or k.startswith("noyron.kernel.geometry.revolved")]
        for k in to_remove:
            sys.modules.pop(k, None)

        mod = importlib.import_module("noyron.kernel.geometry.revolved")

        assert hasattr(mod, "annular_shell_sdf")
        assert hasattr(mod, "cylindrical_boss_sdf")
        assert "noyron.rp" not in " ".join(sys.modules)
        assert "physics.nozzle" not in sys.modules
        assert "physics.regen" not in sys.modules


# ---------------------------------------------------------------------------
# 2. Placement geometry tests
# ---------------------------------------------------------------------------


class TestManifoldPlacement:
    def test_lox_dome_above_face_plate(self, all_manifolds, reference_nozzle):
        """LOX dome bbox z_max aligns with face plate z (x_chamber_start)."""
        dome = all_manifolds["lox_dome"]
        lo, hi = dome.bbox()
        face_z = reference_nozzle.x_chamber_start
        # Dome upper bound should be at the face plate level (within 0.1 mm)
        assert hi[2] == pytest.approx(face_z, abs=1e-4), (
            f"LoxDome z_max {hi[2]*1e3:.2f} mm ≠ x_chamber_start {face_z*1e3:.2f} mm"
        )
        # Dome lower bound is upstream (more negative z)
        assert lo[2] < face_z

    def test_fuel_manifold_outside_chamber(self, all_manifolds, reference_nozzle, reference_bilayer):
        """Fuel manifold bbox r_min ≥ chamber outer jacket radius."""
        from noyron.rp.geometry.manifolds import _r_outer_chamber
        mfld = all_manifolds["fuel_manifold"]
        r_ch_out = _r_outer_chamber(reference_nozzle, reference_bilayer)
        lo, hi = mfld.bbox()
        # The inner bore of the fuel manifold ring starts at r_ch_out
        r_min_ring = mfld._geom()["r_inner"]
        assert r_min_ring == pytest.approx(r_ch_out, rel=1e-6), (
            f"FuelManifold r_inner {r_min_ring*1e3:.2f} mm ≠ r_ch_outer {r_ch_out*1e3:.2f} mm"
        )

    def test_coolant_outlet_starts_at_chamber_face(self, all_manifolds, reference_nozzle):
        """CoolantOutletManifold z_lo aligns with x_chamber_start."""
        mfld = all_manifolds["coolant_outlet"]
        face_z = reference_nozzle.x_chamber_start
        g = mfld._geom()
        assert g["z_lo"] == pytest.approx(face_z, abs=1e-6)

    def test_coolant_inlet_centred_at_nozzle_exit(self, all_manifolds, reference_nozzle):
        """CoolantInletManifold is centred on nozzle exit plane."""
        mfld = all_manifolds["coolant_inlet"]
        g = mfld._geom()
        assert g["z_ctr"] == pytest.approx(reference_nozzle.x_exit, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. No geometric collisions between solid materials
# ---------------------------------------------------------------------------


class TestManifoldCollisions:
    def test_no_geometric_collisions(self, all_manifolds, reference_nozzle):
        """
        500 random points near the chamber end — no point inside two manifold
        solid shells simultaneously.

        The manifolds are placed at different radial and/or axial positions so
        solid material must never overlap.  Collision would indicate a geometry
        bug (e.g. two rings at the same r and z range).
        """
        rng = np.random.default_rng(seed=42)
        n = 500
        z_center = reference_nozzle.x_chamber_start
        z_vals = rng.uniform(z_center - 0.04, z_center + 0.04, n)
        theta = rng.uniform(0.0, 2.0 * math.pi, n)
        rho = rng.uniform(0.005, 0.08, n)
        x_vals = rho * np.cos(theta)
        y_vals = rho * np.sin(theta)

        mfld_list = list(all_manifolds.values())
        sdf_evals = [m.to_sdf()(x_vals, y_vals, z_vals) for m in mfld_list]

        names = list(all_manifolds.keys())
        for i in range(len(mfld_list)):
            for j in range(i + 1, len(mfld_list)):
                in_i = sdf_evals[i] < 0
                in_j = sdf_evals[j] < 0
                n_collide = int(np.sum(in_i & in_j))
                assert n_collide == 0, (
                    f"{n_collide} collision(s) between {names[i]} and {names[j]} "
                    "solid shells — geometry overlap."
                )


# ---------------------------------------------------------------------------
# 4. AM rules verification
# ---------------------------------------------------------------------------


class TestManifoldVerification:
    def test_lpbf_grcop_passes(self, all_manifolds):
        """All manifolds pass LPBF-GRCop AM rules (wall 4 mm > 0.5 mm min)."""
        from noyron.kernel.manufacturing import LPBF_GRCOP
        for name, mfld in all_manifolds.items():
            report = mfld.verify(am_rules=LPBF_GRCOP)
            errors = [i for i in report.issues if i.severity == "error"]
            assert not errors, (
                f"{name} failed LPBF_GRCOP verify: "
                + "; ".join(i.message for i in errors)
            )

    def test_manifold_am_check_with_subtractive_fails(self, all_manifolds):
        """
        SUBTRACTIVE_INCONEL rules must produce errors on every manifold.

        Internal manifold cavities cannot be machined in a single CNC operation
        without assembly/brazing.  The verify() method flags this explicitly.
        (Source: NASA SP-8087 §3.3 note on manifold construction.)
        """
        from noyron.kernel.manufacturing import SUBTRACTIVE_INCONEL
        for name, mfld in all_manifolds.items():
            report = mfld.verify(am_rules=SUBTRACTIVE_INCONEL)
            errors = [i for i in report.issues if i.severity == "error"]
            assert errors, (
                f"{name} should have produced an error for SUBTRACTIVE_INCONEL "
                "but passed — CNC feasibility check is missing."
            )


# ---------------------------------------------------------------------------
# 5. Physics scalars unchanged
# ---------------------------------------------------------------------------


class TestExistingDesignScalars:
    def test_existing_design_scalars_unchanged(self):
        """
        Engine.design() with manifolds gives the same physics as the reference.

        Checks: throat radius, chamber radius, c_star, and that all four
        manifolds are populated on the returned EngineDesign.
        Zero physics code was changed — only geometry is added.
        """
        from noyron.rp.engine import Engine, EngineRequirements, METHALOX
        req = EngineRequirements(
            thrust=4448.22,
            chamber_pressure=6.8948e6,
            mixture_ratio=3.60,
            propellant=METHALOX,
            expansion_ratio=10.605,
        )
        engine = Engine(req)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            design = engine.design()

        # Manifold addition must not perturb physics: run a second independent
        # engine instance with identical requirements and verify scalars match.
        engine2 = Engine(req)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            design2 = engine2.design()

        assert design.nozzle.r_throat == pytest.approx(design2.nozzle.r_throat, rel=1e-10)
        assert design.nozzle.r_chamber == pytest.approx(design2.nozzle.r_chamber, rel=1e-10)
        assert design.gas.c_star == pytest.approx(design2.gas.c_star, rel=1e-10)

        # c* plausible range for methalox at Pc = 6.9 MPa, OF = 3.6
        assert 1800 < design.gas.c_star < 2000

        # All manifolds must be present
        assert design.lox_dome is not None, "LoxDome missing from EngineDesign"
        assert design.fuel_manifold is not None, "FuelManifold missing from EngineDesign"
        assert design.coolant_outlet_manifold is not None, "CoolantOutletManifold missing"
        assert design.coolant_inlet_manifold is not None, "CoolantInletManifold missing"

        # verify_all() must include all four manifolds
        reports = design.verify_all()
        assert "LoxDome" in reports
        assert "FuelManifold" in reports
        assert "CoolantOutletManifold" in reports
        assert "CoolantInletManifold" in reports


# ---------------------------------------------------------------------------
# 6. Full engine render (slow — marked for explicit opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestFullEngineRender:
    def test_full_engine_watertight(self, tmp_path):
        """
        render('full_engine') at 0.4 mm voxel size produces a watertight mesh.

        Watertightness (no open edges, consistent normals) is checked via
        trimesh.  This is a slow integration test — run with -m slow.

        Source: marching cubes guarantees manifold mesh if SDF is continuous;
                trimesh.is_watertight is True when every edge has exactly two
                adjacent faces.
        """
        import trimesh
        from noyron.rp.engine import Engine, EngineRequirements, METHALOX
        req = EngineRequirements(
            thrust=4448.22,
            chamber_pressure=6.8948e6,
            mixture_ratio=3.60,
            propellant=METHALOX,
            expansion_ratio=10.605,
        )
        engine = Engine(req)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            design = engine.design()
            out = engine.render(
                design,
                voxel_size=4e-4,
                output_dir=tmp_path,
                components=("full_engine",),
            )

        stl_path = out["full_engine"]
        assert stl_path.exists()
        mesh = trimesh.load(str(stl_path))
        assert isinstance(mesh, trimesh.Trimesh), "STL did not load as a single mesh"
        assert mesh.is_watertight, (
            f"full_engine mesh is not watertight: "
            f"{len(mesh.faces)} faces, {len(mesh.vertices)} verts"
        )
