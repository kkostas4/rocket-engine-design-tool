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

__all__ = [
    "Component",
    "Renderable",
    "Sizable",
    "VerificationIssue",
    "VerificationReport",
]
