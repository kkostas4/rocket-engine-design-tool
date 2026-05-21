"""
component.py
============

Abstract interface hierarchy for engine design components.

Mirrors the Noyron HX/RP separation pattern (Coding for Engineers, ch. 6):
small ABCs, each with one job, mixed in per-component.

No physics or geometry imports — the kernel layer is domain-free by
construction.

Interfaces
----------
VerificationIssue   Frozen dataclass — one diagnosed condition.
VerificationReport  Frozen dataclass — immutable collection of issues.
Component           Anything named + self-verifiable.
Renderable          Anything that contributes 3-D geometry (SDF + bbox).
Sizable             Anything that can be sized from EngineRequirements.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from noyron.kernel.manufacturing import AMRules

import numpy as np

# SDF: 3-D signed-distance function f(x, y, z) -> ndarray, negative inside solid.
SDF = Callable[..., Any]


@dataclasses.dataclass(frozen=True)
class VerificationIssue:
    """One diagnosed condition from a component's self-check."""

    severity: Literal["error", "warning", "info"]
    message: str
    metric: str | None = None
    value: float | None = None
    threshold: float | None = None


@dataclasses.dataclass(frozen=True)
class VerificationReport:
    """Immutable collection of VerificationIssues for one component.

    Build via .ok() + .add() to accumulate issues without mutation:

        report = VerificationReport.ok("MyComponent")
        if x < 0:
            report = report.add("error", "x must be positive", metric="x", value=x)
    """

    component_name: str
    passed: bool   # False if any issue has severity == "error"
    issues: list[VerificationIssue] = dataclasses.field(
        default_factory=list, hash=False
    )

    @classmethod
    def ok(cls, name: str) -> "VerificationReport":
        """Return an empty passing report for the named component."""
        return cls(component_name=name, passed=True)

    def add(
        self,
        severity: Literal["error", "warning", "info"],
        message: str,
        **kwargs: Any,
    ) -> "VerificationReport":
        """Return a new VerificationReport with one additional issue.

        Parameters
        ----------
        severity : "error" | "warning" | "info"
        message : str
        **kwargs
            Optional VerificationIssue fields: metric, value, threshold.

        Returns
        -------
        VerificationReport
            New frozen instance — the original is unchanged.
        """
        new_issue = VerificationIssue(severity=severity, message=message, **kwargs)
        new_issues = list(self.issues) + [new_issue]
        new_passed = self.passed and (severity != "error")
        return VerificationReport(
            component_name=self.component_name,
            passed=new_passed,
            issues=new_issues,
        )


class Component(ABC):
    """Anything that is a named part of an engineering design."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable component name, unique within a design."""
        ...

    @abstractmethod
    def verify(self, am_rules: "AMRules | None" = None) -> VerificationReport:
        """Run self-checks and return a VerificationReport.

        Parameters
        ----------
        am_rules : AMRules or None
            When provided, adds manufacturability checks against the process
            envelope.  When None, only kernel-level physics checks are run.
        """
        ...


class Renderable(ABC):
    """A component that contributes 3-D geometry."""

    @abstractmethod
    def to_sdf(self, **context: Any) -> SDF:
        """Return a signed-distance function for this component.

        Parameters
        ----------
        **context
            Optional geometry context consumed by implementations that
            lack self-contained geometry.  Common keys: ``nozzle``
            (RaoNozzle), ``bilayer`` (BilayerWall).
            Self-contained implementations (e.g. RaoNozzle) ignore context.

        Returns
        -------
        SDF
            Callable ``f(x, y, z) -> ndarray``.  Negative inside solid,
            positive outside.
        """
        ...

    @abstractmethod
    def bbox(self, **context: Any) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box of this component.

        Parameters
        ----------
        **context
            Same optional keys as to_sdf().

        Returns
        -------
        (min_corner, max_corner)
            Two length-3 ndarrays ``[x, y, z]`` in metres.
        """
        ...


class Sizable(ABC):
    """A component that can be sized from top-level requirements."""

    @classmethod
    @abstractmethod
    def size_from(cls, requirements: Any, **context: Any) -> "Sizable":
        """Build a sized instance from EngineRequirements.

        Parameters
        ----------
        requirements : EngineRequirements (or similar)
            Top-level requirements object.
        **context
            Upstream sized components required by this component's sizing.
            Examples: ``gas`` (CombustionGas), ``m_dot`` (float),
                      ``m_dot_ox`` / ``m_dot_fuel`` (float),
                      ``nozzle`` (RaoNozzle).

        Returns
        -------
        Sizable
            A fully-constructed sized instance of this class.
        """
        ...
