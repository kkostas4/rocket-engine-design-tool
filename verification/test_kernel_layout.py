"""
test_kernel_layout.py
=====================

Phase 1 + Phase 2 regression tests:
  - Phase 1: kernel modules in noyron/kernel/; old geometry/sdf, geometry/mesh,
    physics/heat_exchanger, physics/util paths emit DeprecationWarning shims.
  - Phase 2: rp modules in noyron/rp/; old physics/* and geometry/components
    paths emit DeprecationWarning shims.
  - Hot path: importing noyron.rp.engine fires no DeprecationWarnings.
"""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest


# ---------------------------------------------------------------------------
# 1. Kernel import isolation
# ---------------------------------------------------------------------------


class TestKernelImportIsolation:
    """Importing noyron.kernel submodules must NOT pull in rocket physics."""

    def _fresh_import(self, module_name: str):
        """Import module_name in a context where physics.* is not yet loaded."""
        to_remove = [
            k for k in sys.modules
            if k.startswith("physics") or k.startswith("noyron.kernel")
        ]
        for k in to_remove:
            sys.modules.pop(k, None)
        return importlib.import_module(module_name)

    def test_kernel_hx_base_no_rocket_leakage(self):
        mod = self._fresh_import("noyron.kernel.heat_exchanger.base")
        assert hasattr(mod, "HotSideModel")
        assert "physics.nozzle" not in sys.modules
        assert "physics.regen" not in sys.modules
        assert "physics.thermochem" not in sys.modules

    def test_kernel_hx_solver_no_rocket_leakage(self):
        mod = self._fresh_import("noyron.kernel.heat_exchanger.solver")
        assert hasattr(mod, "solve_heat_exchanger")
        assert "physics.nozzle" not in sys.modules

    def test_kernel_sdf_no_rocket_leakage(self):
        mod = self._fresh_import("noyron.kernel.geometry.sdf.revolution")
        assert hasattr(mod, "revolve")
        assert "physics.nozzle" not in sys.modules
        assert "physics.regen" not in sys.modules

    def test_kernel_util_no_rocket_leakage(self):
        mod = self._fresh_import("noyron.kernel.util.units")
        assert hasattr(mod, "PSI_TO_PA")
        assert "physics.nozzle" not in sys.modules


# ---------------------------------------------------------------------------
# 2. Deprecation shims work
# ---------------------------------------------------------------------------


class TestDeprecationShims:
    """Old import paths still work but emit DeprecationWarning."""

    def test_geometry_sdf_shim_works(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from geometry.sdf.revolution import revolve  # noqa: F401
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert dep_warnings, "expected DeprecationWarning from geometry.sdf.revolution shim"

    def test_geometry_mesh_shim_works(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # use importlib to avoid caching from prior imports
            mod = importlib.import_module("geometry.mesh.extractor")
        assert hasattr(mod, "sdf_to_mesh")
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert dep_warnings, "expected DeprecationWarning from geometry.mesh.extractor shim"

    def test_physics_heat_exchanger_shim_works(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mod = importlib.import_module("physics.heat_exchanger.base")
        assert hasattr(mod, "HotSideModel")
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert dep_warnings, "expected DeprecationWarning from physics.heat_exchanger.base shim"

    def test_physics_util_units_shim_works(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mod = importlib.import_module("physics.util.units")
        assert hasattr(mod, "PSI_TO_PA")
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert dep_warnings, "expected DeprecationWarning from physics.util.units shim"

    def test_physics_heat_exchanger_rocket_regen_shim_works(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mod = importlib.import_module("physics.heat_exchanger.rocket_regen")
        assert hasattr(mod, "solve_rocket_regen")
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert dep_warnings, "expected DeprecationWarning from physics.heat_exchanger.rocket_regen shim"

    # --- Phase 2 shims ---

    def test_physics_nozzle_shim_works(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mod = importlib.import_module("physics.nozzle.rao")
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert dep_warnings, "expected DeprecationWarning from physics.nozzle.rao shim"

    def test_physics_regen_shim_works(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mod = importlib.import_module("physics.regen.bartz")
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert dep_warnings, "expected DeprecationWarning from physics.regen.bartz shim"

    def test_physics_structural_shim_works(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mod = importlib.import_module("physics.structural.bilayer")
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert dep_warnings, "expected DeprecationWarning from physics.structural.bilayer shim"

    def test_geometry_components_shim_works(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mod = importlib.import_module("geometry.components.chamber_nozzle")
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert dep_warnings, "expected DeprecationWarning from geometry.components.chamber_nozzle shim"

    def test_noyron_hx_importable(self):
        """noyron.hx stub is importable and exports PlateFinHeatExchanger."""
        from noyron.hx import PlateFinHeatExchanger  # noqa: F401
        assert hasattr(PlateFinHeatExchanger, "hydraulic_diameter_cold")


# ---------------------------------------------------------------------------
# 3. Hot path: no DeprecationWarnings on normal engine import
# ---------------------------------------------------------------------------


class TestNoInternalDeprecationWarnings:
    """Internal code uses new paths — importing the engine emits no DeprecationWarnings."""

    def test_engine_import_no_deprecation(self):
        """noyron.rp.engine (and all its internal imports) must not fire DeprecationWarning."""
        # Strip engine + physics from module cache so re-import is fresh
        to_remove = [
            k for k in sys.modules
            if k.startswith("noyron") or k.startswith("physics") or k.startswith("geometry")
        ]
        for k in to_remove:
            sys.modules.pop(k, None)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            importlib.import_module("noyron.rp.engine")

        shim_paths = [
            "geometry/sdf", "geometry/mesh",
            "physics/heat_exchanger", "physics/util",
            "physics/nozzle", "physics/regen", "physics/structural",
            "physics/injector", "physics/thermochem", "geometry/components",
        ]
        dep_warnings = [
            x for x in w
            if issubclass(x.category, DeprecationWarning)
            and "deprecated" in str(x.message).lower()
            and any(path in str(x.filename).replace("\\", "/") for path in shim_paths)
        ]
        assert not dep_warnings, (
            f"noyron.rp.engine import triggered {len(dep_warnings)} DeprecationWarning(s):\n"
            + "\n".join(f"  {x.filename}:{x.lineno}: {x.message}" for x in dep_warnings)
        )
