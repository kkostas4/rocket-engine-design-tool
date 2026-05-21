"""
test_components.py
==================

Verification tests for the Component / Renderable / Sizable interface
hierarchy (noyron/kernel/component.py).

Tests
-----
1. test_kernel_import_isolation
   Importing noyron.kernel.component must NOT pull physics.nozzle into
   sys.modules.  The kernel is domain-free by construction.

2. test_verification_report_frozen
   VerificationReport.add() returns a new instance; the original is unchanged.
   Frozen-dataclass semantics hold: direct attribute mutation raises.

3. test_raonozzle_verify_passes
   Reference-engine RaoNozzle (r_throat=11.157 mm, eps=10.605, Ac_At=4.0)
   passes all checks with no issues.

4. test_raonozzle_warns_bad_ac_at
   RaoNozzle with Ac_At=9.0 (outside [3, 8]) fires exactly one warning
   about the contraction ratio; overall report still passes (warning ≠ error).

5. test_pintleinjector_warns_tmr
   Reference-engine PintleInjector has TMR ≈ 0.47 < 0.50 lower bound
   (Dressler & Bauer 2000).  verify() fires exactly one warning on TMR;
   the report still passes.

6. test_to_sdf_lazy_import
   PintleInjector.to_sdf() with no context (nozzle=None) calls
   injector_sdf in local-coordinate mode and returns a callable.
   Verifies that the lazy-import path completes without error.

7. test_verify_all_reference_engine  (slow)
   Full Engine.design() on the reference engine, then verify_all().
   Must return a dict containing at least RaoNozzle, PintleInjector, and
   BilayerWall; TMR warning must appear in the PintleInjector report.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Reference constants (CLAUDE.md)
# ---------------------------------------------------------------------------

_R_THROAT = 0.022313 / 2        # m  — 22.313 mm throat diameter
_EPS = 10.605                    # expansion ratio
_MDOT_OX = 1.1144                # kg/s LOX
_MDOT_FUEL = 0.3096              # kg/s LCH4
_PC = 6.8948e6                   # Pa   chamber pressure
_DP_INJ = 0.45 * _PC             # Pa   uniform-ΔP injector drop


# ---------------------------------------------------------------------------
# Test 1 — kernel import isolation
# ---------------------------------------------------------------------------


def test_kernel_import_isolation():
    """Importing noyron.kernel.component must not load physics.nozzle."""
    # Remove any existing cached imports so the test is deterministic.
    for key in list(sys.modules):
        if key.startswith("physics.nozzle"):
            del sys.modules[key]

    import importlib
    importlib.import_module("noyron.kernel.component")

    loaded = [k for k in sys.modules if k.startswith("physics.nozzle")]
    assert loaded == [], (
        f"Kernel import leaked physics.nozzle modules: {loaded}"
    )


# ---------------------------------------------------------------------------
# Test 2 — VerificationReport frozen semantics
# ---------------------------------------------------------------------------


def test_verification_report_frozen():
    from noyron.kernel.component import VerificationReport

    r0 = VerificationReport.ok("MyComp")
    r1 = r0.add("warning", "a warning", metric="x", value=1.0, threshold=0.5)
    r2 = r1.add("error", "a fatal issue", metric="y", value=-1.0, threshold=0.0)

    # add() returns new instance; originals unchanged
    assert r0.passed is True
    assert len(r0.issues) == 0

    assert r1.passed is True           # warning does not fail
    assert len(r1.issues) == 1
    assert r1.issues[0].severity == "warning"

    assert r2.passed is False          # error fails
    assert len(r2.issues) == 2

    # Direct attribute mutation raises (frozen dataclass)
    with pytest.raises((TypeError, AttributeError)):
        r0.passed = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test 3 — RaoNozzle.verify() passes on reference engine
# ---------------------------------------------------------------------------


def test_raonozzle_verify_passes():
    import warnings
    from physics.nozzle.rao import RaoNozzle

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nozzle = RaoNozzle(r_throat=_R_THROAT, epsilon=_EPS, Ac_At=4.0)

    report = nozzle.verify()
    assert report.passed is True, (
        f"Reference RaoNozzle should pass verify(); issues: {report.issues}"
    )
    assert report.issues == [], (
        f"Reference RaoNozzle should have no issues; got: {report.issues}"
    )


# ---------------------------------------------------------------------------
# Test 4 — RaoNozzle warns when Ac_At is outside [3, 8]
# ---------------------------------------------------------------------------


def test_raonozzle_warns_bad_ac_at():
    from physics.nozzle.rao import RaoNozzle

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nozzle = RaoNozzle(r_throat=0.020, epsilon=10.0, Ac_At=9.0)

    report = nozzle.verify()

    warnings_found = [i for i in report.issues if i.severity == "warning"]
    assert len(warnings_found) == 1, (
        f"Expected exactly 1 Ac_At warning; got {warnings_found}"
    )
    assert "Ac_At" in warnings_found[0].message
    assert report.passed is True, "A warning must not set passed=False"


# ---------------------------------------------------------------------------
# Test 5 — PintleInjector warns on TMR < 0.50
# ---------------------------------------------------------------------------


def test_pintleinjector_warns_tmr():
    from physics.injector.pintle import PintleInjector

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inj = PintleInjector.from_uniform_dP(
            dP_inj=_DP_INJ,
            m_dot_ox=_MDOT_OX,
            m_dot_fuel=_MDOT_FUEL,
            rho_ox=1116.0,
            rho_fuel=386.0,
            Cd_ox=0.6,
            Cd_fuel=0.6,
            pintle_OD=0.030,
            n_fuel_orifices=20,
        )

    assert inj.TMR < 0.5, (
        f"Reference TMR should be < 0.5; got {inj.TMR:.4f}"
    )

    report = inj.verify()

    tmr_warnings = [
        i for i in report.issues
        if i.severity == "warning" and i.metric == "TMR"
    ]
    assert len(tmr_warnings) == 1, (
        f"Expected 1 TMR warning; got {report.issues}"
    )
    assert tmr_warnings[0].value is not None
    assert tmr_warnings[0].value < 0.5
    assert report.passed is True


# ---------------------------------------------------------------------------
# Test 6 — to_sdf() lazy import works without context
# ---------------------------------------------------------------------------


def test_to_sdf_lazy_import():
    """PintleInjector.to_sdf() with no nozzle uses local-coordinate fallback."""
    from physics.injector.pintle import PintleInjector

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inj = PintleInjector.from_uniform_dP(
            dP_inj=_DP_INJ,
            m_dot_ox=_MDOT_OX,
            m_dot_fuel=_MDOT_FUEL,
            rho_ox=1116.0,
            rho_fuel=386.0,
            Cd_ox=0.6,
            Cd_fuel=0.6,
            pintle_OD=0.030,
            n_fuel_orifices=20,
        )

    sdf = inj.to_sdf()   # no context → nozzle=None, bilayer=None
    assert callable(sdf), "to_sdf() must return a callable"


# ---------------------------------------------------------------------------
# Test 7 — verify_all() on the full reference-engine design  (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_verify_all_reference_engine():
    from noyron.engine import Engine, EngineRequirements, METHALOX

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

    reports = design.verify_all()

    assert "RaoNozzle" in reports, "verify_all() must include RaoNozzle"
    assert "PintleInjector" in reports, "verify_all() must include PintleInjector"
    assert "BilayerWall" in reports, "verify_all() must include BilayerWall"

    # RaoNozzle and BilayerWall must pass cleanly
    assert reports["RaoNozzle"].passed is True
    assert reports["BilayerWall"].passed is True

    # PintleInjector must pass but carry exactly one TMR warning
    assert reports["PintleInjector"].passed is True
    tmr_warns = [
        i for i in reports["PintleInjector"].issues
        if i.severity == "warning" and i.metric == "TMR"
    ]
    assert len(tmr_warns) == 1, (
        f"Expected 1 TMR warning in full design; got {reports['PintleInjector'].issues}"
    )
