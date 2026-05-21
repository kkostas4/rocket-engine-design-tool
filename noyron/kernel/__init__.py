"""
noyron.kernel — domain-free interface primitives.

Importing this package does NOT pull in physics.* or geometry.*.
"""

from noyron.kernel.component import (
    Component,
    Renderable,
    Sizable,
    VerificationIssue,
    VerificationReport,
)
from noyron.kernel.manufacturing import (
    AMRules,
    LPBF_GRCOP,
    LPBF_INCONEL,
    BINDER_JET_INCONEL,
    SUBTRACTIVE_INCONEL,
    AM_RULES_PRESETS,
)

__all__ = [
    "Component",
    "Renderable",
    "Sizable",
    "VerificationIssue",
    "VerificationReport",
    "AMRules",
    "LPBF_GRCOP",
    "LPBF_INCONEL",
    "BINDER_JET_INCONEL",
    "SUBTRACTIVE_INCONEL",
    "AM_RULES_PRESETS",
]
