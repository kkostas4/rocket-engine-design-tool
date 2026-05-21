"""
test_manufacturing.py
=====================

Verification tests for noyron/kernel/manufacturing.py (AMRules).

Tests
-----
1. test_rules_are_frozen
2. test_default_design_passes_lpbf_grcop  (slow)
3. test_design_fails_subtractive_for_channels  (slow)
4. test_binder_jet_allows_finer_channels
5. test_existing_scalar_results_unchanged  (slow)
6. test_summary_shows_active_rules  (slow)
"""

from __future__ import annotations

import warnings

import pytest


# ---------------------------------------------------------------------------
# Test 1 — AMRules is frozen (mutation raises)
# ---------------------------------------------------------------------------


def test_rules_are_frozen():
    """Frozen dataclass: attribute mutation must raise TypeError."""
    from noyron.kernel.manufacturing import LPBF_GRCOP

    with pytest.raises((TypeError, AttributeError)):
        LPBF_GRCOP.min_wall_thickness = 0.1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test 2 — Default design passes LPBF_GRCOP rules (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_default_design_passes_lpbf_grcop():
    """Reference engine with LPBF_GRCOP (default) must have no errors in verify_all()."""
    from noyron.engine import Engine, EngineRequirements, METHALOX

    req = EngineRequirements(
        thrust=4448.22,
        chamber_pressure=6.8948e6,
        mixture_ratio=3.60,
        propellant=METHALOX,
        expansion_ratio=10.605,
        # am_rules defaults to LPBF_GRCOP
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        design = Engine(req).design()

    reports = design.verify_all()
    for comp_name, report in reports.items():
        errors = [i for i in report.issues if i.severity == "error"]
        assert errors == [], (
            f"{comp_name} has errors with LPBF_GRCOP rules: {errors}"
        )


# ---------------------------------------------------------------------------
# Test 3 — SUBTRACTIVE_INCONEL produces errors for AM-sized channels (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_design_fails_subtractive_for_channels():
    """Reference engine with SUBTRACTIVE_INCONEL rules must fail sizing or produce errors.

    SUBTRACTIVE_INCONEL has min_channel_width=2.0mm and min_feature_size=0.5mm.
    For the reference engine throat diameter (~22mm), the channels literally cannot
    fit around the circumference (arc spacing ~0.9mm < 2.5mm minimum).
    This makes sizing infeasible, which is the correct outcome.

    Accept either:
    - ChannelSizingError / SizingInfeasibleError raised during design(), OR
    - The design succeeds but verify_all() finds errors
    """
    from noyron.engine import Engine, EngineRequirements, METHALOX
    from noyron.kernel.manufacturing import SUBTRACTIVE_INCONEL

    req = EngineRequirements(
        thrust=4448.22,
        chamber_pressure=6.8948e6,
        mixture_ratio=3.60,
        propellant=METHALOX,
        expansion_ratio=10.605,
        am_rules=SUBTRACTIVE_INCONEL,
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            design = Engine(req).design()
        # If design succeeded, check for errors in verify_all()
        reports = design.verify_all()
        has_errors = any(
            any(i.severity == "error" for i in r.issues)
            for r in reports.values()
        )
        assert has_errors, (
            "SUBTRACTIVE_INCONEL rules should produce errors for AM-sized channels "
            f"(channel width ~0.6mm < 2.0mm minimum). Got: {reports}"
        )
    except Exception as exc:
        # ChannelSizingError or SizingInfeasibleError: also acceptable —
        # SUBTRACTIVE_INCONEL min_channel_width=2mm is too wide to fit at throat
        exc_type = type(exc).__name__
        assert "Sizing" in exc_type or "Channel" in exc_type or "infeasible" in str(exc).lower(), (
            f"Unexpected exception type {exc_type}: {exc}"
        )


# ---------------------------------------------------------------------------
# Test 4 — BINDER_JET allows channels that LPBF_GRCOP rejects
# ---------------------------------------------------------------------------


def test_binder_jet_allows_finer_channels():
    """0.45 mm channel width: passes BINDER_JET_INCONEL (min 0.4mm), fails LPBF_GRCOP (min 0.6mm)."""
    from physics.regen.coolant import CoolingChannelGeometry
    from noyron.kernel.manufacturing import BINDER_JET_INCONEL, LPBF_GRCOP

    # 0.45 mm channel width — between BINDER_JET min (0.4mm) and LPBF_GRCOP min (0.6mm)
    w = 0.45e-3

    geom = CoolingChannelGeometry(
        n_channels=60,
        width=lambda x: w,
        height=lambda x: 1.0e-3,
        wall_thickness=lambda x: 0.5e-3,
        land_thickness=lambda x: 0.3e-3,
    )

    report_bj = geom.verify(am_rules=BINDER_JET_INCONEL)
    report_lpbf = geom.verify(am_rules=LPBF_GRCOP)

    # BINDER_JET_INCONEL: 0.45 >= 0.4 → no channel width error
    channel_width_errors_bj = [
        i for i in report_bj.issues
        if i.severity == "error" and "width" in i.message.lower()
    ]
    assert channel_width_errors_bj == [], (
        f"BINDER_JET_INCONEL should allow 0.45mm channels; "
        f"got errors: {[i.message for i in channel_width_errors_bj]}"
    )

    # LPBF_GRCOP: 0.45 < 0.6 → channel width error
    channel_width_errors_lpbf = [
        i for i in report_lpbf.issues
        if i.severity == "error" and "width" in i.message.lower()
    ]
    assert len(channel_width_errors_lpbf) >= 1, (
        f"LPBF_GRCOP should reject 0.45mm channels; "
        f"got issues: {[i.message for i in report_lpbf.issues]}"
    )


# ---------------------------------------------------------------------------
# Test 5 — Existing scalar results unchanged (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_existing_scalar_results_unchanged():
    """Adding AMRules must not change throat radius or other geometry scalars."""
    from noyron.engine import Engine, EngineRequirements, METHALOX

    # Reference values from CLAUDE.md
    R_THROAT_REF = 0.022313 / 2   # m — 22.313 mm throat diameter

    req = EngineRequirements(
        thrust=4448.22,
        chamber_pressure=6.8948e6,
        mixture_ratio=3.60,
        propellant=METHALOX,
        expansion_ratio=10.605,
        # default am_rules = LPBF_GRCOP (same values as old hardcoded defaults)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        design = Engine(req).design()

    # Throat radius must be within 0.01% of reference
    assert abs(design.nozzle.r_throat - R_THROAT_REF) / R_THROAT_REF < 1e-4, (
        f"r_throat changed: {design.nozzle.r_throat:.6f} m vs {R_THROAT_REF:.6f} m"
    )


# ---------------------------------------------------------------------------
# Test 6 — summary() shows active manufacturing rules (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_summary_shows_active_rules():
    """Engine.summary() must include the active manufacturing process name."""
    from noyron.engine import Engine, EngineRequirements, METHALOX

    req = EngineRequirements(
        thrust=4448.22,
        chamber_pressure=6.8948e6,
        mixture_ratio=3.60,
        propellant=METHALOX,
        expansion_ratio=10.605,
        # default am_rules = LPBF_GRCOP
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        design = Engine(req).design()

    summary = Engine(req).summary(design)

    assert "Manufacturing process" in summary, (
        "summary() must show 'Manufacturing process' line"
    )
    assert "LPBF" in summary, (
        "summary() must show 'LPBF' for default LPBF_GRCOP rules"
    )
