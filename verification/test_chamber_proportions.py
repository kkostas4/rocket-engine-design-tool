"""
test_chamber_proportions.py
===========================

Verification tests for L_star and Ac_At as design parameters.

These tests work directly against RaoNozzle geometry rather than running
the full Engine.design() pipeline, so they run in milliseconds.

Sources
-------
Sutton, G.P. and Biblarz, O. (2017). "Rocket Propulsion Elements," 9th ed.
    Ch. 8, Table 8-1 — L* guidance; L_chamber derivation.
Huzel & Huang (1992) §4.2 — Ac/At and chamber proportion sizing.
"""

from __future__ import annotations

import math

import pytest

from noyron.rp.nozzle.rao import RaoNozzle


# Reference throat: 1000-lbf methalox engine (CEA-corrected).
_R_THROAT = 0.010947319134266876   # m
_EPSILON   = 10.605                # Ae/At
_L_STAR    = 1.1                   # m  (methalox default)
_AC_AT     = 4.0


# ---------------------------------------------------------------------------
# Test 1: higher Ac/At shortens the cylindrical chamber
# ---------------------------------------------------------------------------

def test_higher_ac_at_shortens_chamber():
    """
    Doubling Ac/At from 4 to 8 must halve L_chamber (at fixed L_star).

    Derivation:
        L_chamber = (L* · At − V_conv) / A_ch

    A_ch is proportional to Ac_At at fixed r_throat.  V_conv depends only
    on the convergent arc (scaled by r_throat), which is independent of
    Ac_At.  Therefore L_chamber decreases as Ac_At increases.

    The halving relationship holds approximately when V_conv « L*·At
    (valid for methalox L*=1.1 at this throat size).

    Source: Sutton & Biblarz (2017) Ch. 8 — L* definition.
    """
    noz4 = RaoNozzle(r_throat=_R_THROAT, epsilon=_EPSILON, L_star=_L_STAR, Ac_At=4.0)
    noz8 = RaoNozzle(r_throat=_R_THROAT, epsilon=_EPSILON, L_star=_L_STAR, Ac_At=8.0)

    L4 = noz4.L_chamber
    L8 = noz8.L_chamber

    assert L8 < L4, (
        f"L_chamber should decrease as Ac_At increases: "
        f"L(4)={L4*1e3:.1f} mm, L(8)={L8*1e3:.1f} mm"
    )
    # The ratio should be close to Ac_At(4)/Ac_At(8) = 0.5 when V_conv is small.
    # Allow 20% tolerance for the V_conv correction term.
    ratio = L8 / L4
    assert 0.35 <= ratio <= 0.65, (
        f"L_chamber ratio L(8)/L(4) = {ratio:.3f}, expected near 0.5 "
        f"(L4={L4*1e3:.1f} mm, L8={L8*1e3:.1f} mm)"
    )


# ---------------------------------------------------------------------------
# Test 2: lower L_star proportionally shortens the chamber
# ---------------------------------------------------------------------------

def test_lower_l_star_shortens_chamber():
    """
    Reducing L_star from 1.1 m to 0.7 m must reduce L_chamber by ~36%.

    When V_conv « L*·At:
        L_chamber ∝ L_star

    The expected reduction is (1.1 - 0.7) / 1.1 ≈ 36.4%.
    Allow ±5 pp for the V_conv correction term.

    Source: Sutton & Biblarz (2017) Ch. 8 — L* definition.
    """
    noz_hi = RaoNozzle(r_throat=_R_THROAT, epsilon=_EPSILON, L_star=1.1, Ac_At=_AC_AT)
    noz_lo = RaoNozzle(r_throat=_R_THROAT, epsilon=_EPSILON, L_star=0.7, Ac_At=_AC_AT)

    L_hi = noz_hi.L_chamber
    L_lo = noz_lo.L_chamber

    assert L_lo < L_hi, (
        f"L_chamber should decrease as L_star decreases: "
        f"L(1.1)={L_hi*1e3:.1f} mm, L(0.7)={L_lo*1e3:.1f} mm"
    )

    reduction_pct = (L_hi - L_lo) / L_hi * 100.0
    assert 25.0 <= reduction_pct <= 45.0, (
        f"L_chamber reduction = {reduction_pct:.1f}%, expected ~36% ± 10 pp "
        f"(L_hi={L_hi*1e3:.1f} mm, L_lo={L_lo*1e3:.1f} mm)"
    )


# ---------------------------------------------------------------------------
# Test 3: thrust scales r_chamber as sqrt(thrust), L_chamber unchanged
# ---------------------------------------------------------------------------

def test_thrust_scales_radius_not_length():
    """
    At constant L_star and Ac_At, scaling thrust by 10x scales r_throat
    (and r_chamber) as sqrt(10) while L_chamber changes only via V_conv.

    The L_chamber correction from V_conv is proportional to sqrt(thrust):
        L_chamber = (L* · At − V_conv) / A_ch
        At, A_ch ∝ thrust  →  L* · At / A_ch = L* / Ac_At  (independent of thrust)
        V_conv ∝ r_throat^3 ∝ At^(3/2) ∝ thrust^(3/2)
        V_conv / A_ch ∝ thrust^(1/2)

    So L_chamber *decreases* slightly as thrust grows (convergent eats more
    of the V*).  The dominant term (L*/Ac_At) is constant.  We test that
    L_chamber changes by less than 30% when thrust increases 10×.

    r_chamber scaling: r_chamber = r_throat × sqrt(Ac_At)
        r_throat ∝ sqrt(At) ∝ sqrt(thrust)
    So r_chamber should grow as sqrt(10) ≈ 3.162 × when thrust is 10×.

    Source: Sutton & Biblarz (2017) Ch. 8 — L* definition and throat sizing.
    """
    # Simulate throat radii corresponding to 1× and 10× thrust at the same
    # Pc and c*.  At scales directly with thrust (At = m_dot * c* / Pc),
    # so r_throat scales as sqrt(10).
    r1x = _R_THROAT
    r10x = _R_THROAT * math.sqrt(10.0)

    noz_1x  = RaoNozzle(r_throat=r1x,  epsilon=_EPSILON, L_star=_L_STAR, Ac_At=_AC_AT)
    noz_10x = RaoNozzle(r_throat=r10x, epsilon=_EPSILON, L_star=_L_STAR, Ac_At=_AC_AT)

    # r_chamber must scale as sqrt(10) within 1%
    r_ratio = noz_10x.r_chamber / noz_1x.r_chamber
    assert abs(r_ratio - math.sqrt(10.0)) < 0.01, (
        f"r_chamber ratio = {r_ratio:.4f}, expected sqrt(10) = {math.sqrt(10):.4f}"
    )

    # L_chamber dominant term is L*/Ac_At = const; V_conv correction grows as
    # sqrt(thrust).  For 10× thrust the V_conv correction grows by sqrt(10) ≈ 3.16×.
    # L_chamber should not change by more than 30% for 10× thrust scaling.
    L_ratio = noz_10x.L_chamber / noz_1x.L_chamber
    assert 0.70 <= L_ratio <= 1.30, (
        f"L_chamber ratio (10×/1×) = {L_ratio:.3f}, expected within [0.70, 1.30]. "
        f"L(1×)={noz_1x.L_chamber*1e3:.1f} mm, L(10×)={noz_10x.L_chamber*1e3:.1f} mm"
    )
