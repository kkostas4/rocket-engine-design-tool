"""
test_kernel_layout.py
=====================

Phase 1 regression tests: kernel modules are in noyron/kernel/; old paths
are backward-compat shims that emit DeprecationWarning; all internal imports
use the new paths so there are no DeprecationWarnings on the hot path.
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


# ---------------------------------------------------------------------------
# 3. Hot path: no DeprecationWarnings on normal engine import
# ---------------------------------------------------------------------------


class TestNoInternalDeprecationWarnings:
    """Internal code uses new paths — importing the engine emits no DeprecationWarnings."""

    def test_engine_import_no_deprecation(self):
        """noyron.engine (and all its internal imports) must not fire DeprecationWarning."""
        # Strip engine + physics from module cache so re-import is fresh
        to_remove = [
            k for k in sys.modules
            if k.startswith("noyron") or k.startswith("physics") or k.startswith("geometry")
        ]
        for k in to_remove:
            sys.modules.pop(k, None)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            importlib.import_module("noyron.engine")

        dep_warnings = [
            x for x in w
            if issubclass(x.category, DeprecationWarning)
            and "deprecated" in str(x.message).lower()
            # Only flag our own shim warnings, not third-party
            and any(
                path in str(x.filename)
                for path in ["geometry/sdf", "geometry/mesh", "physics/heat_exchanger", "physics/util"]
            )
        ]
        assert not dep_warnings, (
            f"noyron.engine import triggered {len(dep_warnings)} DeprecationWarning(s):\n"
            + "\n".join(f"  {w.filename}:{w.lineno}: {w.message}" for w in dep_warnings)
        )
