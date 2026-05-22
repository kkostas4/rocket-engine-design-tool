"""
manifolds.py
============

LOX dome, fuel manifold, coolant inlet and coolant outlet manifolds as
first-class Component + Renderable objects.

Each manifold builds its geometry from one annular_shell_sdf (ring body)
combined with one cylindrical_boss_sdf (inlet/outlet port stub) via
SDF union.

Coordinate system (inherits from geometry kernel):
  z = 0 at the throat; z < 0 upstream (injector / chamber end);
  z > 0 downstream (nozzle exit end).  ρ = √(x²+y²) is radial distance.

Sources
-------
Huzel, D.K. and Huang, D.H. (1992). "Modern Engineering for Design of
    Liquid-Propellant Rocket Engines." AIAA Progress in Astronautics
    and Aeronautics, Vol. 147. §4.2 — manifold placement and sizing.

NASA SP-8087 (1977). "Liquid Rocket Engine Fluid-Cooled Combustion
    Chambers." §3.3 — wall construction and manifold geometry.

Dressler, G.A. and Bauer, J.M. (2000). "TRW Pintle Engine Heritage and
    Performance Characteristics." AIAA 2000-3871. §3 — pintle LOX dome.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from noyron.kernel.component import (
    Component,
    Renderable,
    VerificationReport,
)
from noyron.kernel.geometry.revolved import annular_shell_sdf, cylindrical_boss_sdf
from noyron.kernel.geometry.sdf.operations import union

if TYPE_CHECKING:
    from noyron.kernel.component import SDF
    from noyron.kernel.manufacturing import AMRules
    from noyron.rp.nozzle.rao import RaoNozzle
    from noyron.rp.structural.bilayer import BilayerWall
    from noyron.rp.injector.pintle import PintleInjector


# ---------------------------------------------------------------------------
# ManifoldConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifoldConfig:
    """
    Frozen sizing parameters for all four manifolds.

    All dimensions SI (metres).  Defaults sized for the 1000-lbf methalox
    reference engine (chamber r = 22.3 mm, jacket OD ≈ 28 mm).

    Source: Huzel & Huang (1992) §4.2; engineering judgment for boss sizing.
    """

    lox_dome_height_ratio: float = 0.5
    lox_dome_inlet_OD: float = 12e-3
    fuel_manifold_radial_width: float = 15e-3
    fuel_manifold_inlet_OD: float = 10e-3
    coolant_manifold_radial_width: float = 12e-3
    coolant_manifold_axial_extent: float = 20e-3
    coolant_manifold_inlet_OD: float = 10e-3
    boss_protrusion: float = 15e-3
    wall_thickness: float = 4e-3


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------


def _r_outer_chamber(nozzle: "RaoNozzle", bilayer: "BilayerWall") -> float:
    """Outer jacket OD at the cylindrical chamber section [m]."""
    z = nozzle.x_inlet
    geom = bilayer.channel_geometry
    return (
        nozzle.r_chamber
        + bilayer.inner_liner.thickness(z)
        + geom.height(z)
        + bilayer.outer_jacket.thickness(z)
    )


def _r_outer_exit(nozzle: "RaoNozzle", bilayer: "BilayerWall") -> float:
    """Outer jacket OD at the nozzle exit plane [m]."""
    z = nozzle.x_exit
    r_exit_inner = nozzle.r_throat * math.sqrt(nozzle.epsilon)
    geom = bilayer.channel_geometry
    return (
        r_exit_inner
        + bilayer.inner_liner.thickness(z)
        + geom.height(z)
        + bilayer.outer_jacket.thickness(z)
    )


def _am_process_error(
    report: VerificationReport, am_rules: "AMRules", manifold_name: str
) -> VerificationReport:
    """Add an error if the process cannot produce internal manifold cavities."""
    if "CNC" in am_rules.process_name:
        report = report.add(
            "error",
            f"Internal manifold cavity of {manifold_name} is not achievable by "
            f"{am_rules.process_name} without assembly and brazing. "
            "Use LPBF or binder-jet AM instead. "
            "(Source: NASA SP-8087 §3.3 note on manifold construction.)",
        )
    return report


# ---------------------------------------------------------------------------
# LoxDome
# ---------------------------------------------------------------------------


@dataclass
class LoxDome(Component, Renderable):
    """
    LOX distribution dome upstream of the injector face plate.

    Geometry: hollow annular shell whose z_hi aligns with the upstream
    face of the combustion chamber (nozzle.x_chamber_start).  The shell
    inner bore matches the pintle post OD; the outer envelope matches the
    chamber jacket OD.  An axial inlet boss protrudes further upstream at
    the engine centre-line for the LOX feed line.

    Source: Dressler & Bauer (2000) AIAA 2000-3871 §3 — LOX dome geometry.
            Huzel & Huang (1992) §4.2 — dome height sizing.
    """

    nozzle: "RaoNozzle"
    bilayer: "BilayerWall"
    injector: "PintleInjector"
    config: ManifoldConfig = field(default_factory=ManifoldConfig)

    @property
    def name(self) -> str:
        return "LoxDome"

    def _geom(self) -> dict:
        cfg = self.config
        dome_height = cfg.lox_dome_height_ratio * 2.0 * self.nozzle.r_chamber
        r_inner = self.injector.pintle_OD / 2.0
        r_outer = _r_outer_chamber(self.nozzle, self.bilayer)
        z_hi = float(self.nozzle.x_chamber_start)
        z_lo = z_hi - dome_height
        return {
            "z_lo": z_lo, "z_hi": z_hi,
            "r_inner": r_inner, "r_outer": r_outer,
            "dome_height": dome_height,
        }

    def to_sdf(self, **context: Any) -> "SDF":
        cfg = self.config
        g = self._geom()
        t = cfg.wall_thickness
        shell = annular_shell_sdf(
            r_inner=g["r_inner"],
            r_outer=g["r_outer"],
            z_lo=g["z_lo"],
            z_hi=g["z_hi"],
            wall_thickness=t,
        )
        boss_OD = cfg.lox_dome_inlet_OD
        boss_inner_OD = boss_OD - 2.0 * t
        boss = cylindrical_boss_sdf(
            axis_origin=(0.0, 0.0, g["z_lo"]),
            axis_direction=(0.0, 0.0, -1.0),
            OD=boss_OD,
            length=cfg.boss_protrusion,
            inner_OD=boss_inner_OD if boss_inner_OD > 1e-9 else None,
        )
        return union(shell, boss)

    def bbox(self, **context: Any) -> tuple[np.ndarray, np.ndarray]:
        g = self._geom()
        r = g["r_outer"]
        return (
            np.array([-r, -r, g["z_lo"] - self.config.boss_protrusion]),
            np.array([r, r, g["z_hi"]]),
        )

    def verify(self, am_rules: "AMRules | None" = None) -> VerificationReport:
        report = VerificationReport.ok(self.name)
        g = self._geom()
        if g["r_outer"] <= g["r_inner"]:
            report = report.add(
                "error",
                f"r_outer ({g['r_outer']*1e3:.1f} mm) ≤ r_inner ({g['r_inner']*1e3:.1f} mm); "
                "increase r_outer or reduce pintle_OD.",
            )
        if am_rules is not None:
            t = self.config.wall_thickness
            ok, msg = am_rules.check_wall_thickness(t, "LOX dome wall")
            if not ok:
                report = report.add(
                    "warning", msg, metric="wall_thickness", value=t,
                    threshold=am_rules.min_wall_thickness,
                )
            ok2, msg2 = am_rules.check_hole(
                self.config.lox_dome_inlet_OD, "LOX dome inlet boss"
            )
            if not ok2:
                report = report.add(
                    "warning", msg2, metric="lox_dome_inlet_OD",
                    value=self.config.lox_dome_inlet_OD,
                    threshold=am_rules.min_hole_diameter,
                )
            report = _am_process_error(report, am_rules, self.name)
        return report


# ---------------------------------------------------------------------------
# FuelManifold
# ---------------------------------------------------------------------------


@dataclass
class FuelManifold(Component, Renderable):
    """
    Fuel distribution manifold at the outer rim of the injector face plate.

    Geometry: annular shell wrapping the chamber outer jacket at the face
    plate axial station (z ∈ [x_chamber_start − axial_extent, x_chamber_start]).
    A radial inlet boss at θ=0 (positive x-axis) connects the fuel feed line.

    Source: Huzel & Huang (1992) §4.2 — fuel manifold placement.
            NASA SP-8089 (1976) §4.2 — face plate and manifold sizing.
    """

    nozzle: "RaoNozzle"
    bilayer: "BilayerWall"
    injector: "PintleInjector"
    config: ManifoldConfig = field(default_factory=ManifoldConfig)

    @property
    def name(self) -> str:
        return "FuelManifold"

    def _geom(self) -> dict:
        cfg = self.config
        r_inner = _r_outer_chamber(self.nozzle, self.bilayer)
        r_outer = r_inner + cfg.fuel_manifold_radial_width
        z_hi = float(self.nozzle.x_chamber_start)
        z_lo = z_hi - cfg.coolant_manifold_axial_extent
        z_ctr = 0.5 * (z_lo + z_hi)
        return {
            "z_lo": z_lo, "z_hi": z_hi, "z_ctr": z_ctr,
            "r_inner": r_inner, "r_outer": r_outer,
        }

    def to_sdf(self, **context: Any) -> "SDF":
        cfg = self.config
        g = self._geom()
        t = cfg.wall_thickness
        shell = annular_shell_sdf(
            r_inner=g["r_inner"],
            r_outer=g["r_outer"],
            z_lo=g["z_lo"],
            z_hi=g["z_hi"],
            wall_thickness=t,
        )
        boss_OD = cfg.fuel_manifold_inlet_OD
        boss_inner_OD = boss_OD - 2.0 * t
        boss = cylindrical_boss_sdf(
            axis_origin=(g["r_outer"], 0.0, g["z_ctr"]),
            axis_direction=(1.0, 0.0, 0.0),
            OD=boss_OD,
            length=cfg.boss_protrusion,
            inner_OD=boss_inner_OD if boss_inner_OD > 1e-9 else None,
        )
        return union(shell, boss)

    def bbox(self, **context: Any) -> tuple[np.ndarray, np.ndarray]:
        g = self._geom()
        r = g["r_outer"]
        return (
            np.array([-r, -r, g["z_lo"]]),
            np.array([r + self.config.boss_protrusion, r, g["z_hi"]]),
        )

    def verify(self, am_rules: "AMRules | None" = None) -> VerificationReport:
        report = VerificationReport.ok(self.name)
        if am_rules is not None:
            t = self.config.wall_thickness
            ok, msg = am_rules.check_wall_thickness(t, "fuel manifold wall")
            if not ok:
                report = report.add(
                    "warning", msg, metric="wall_thickness", value=t,
                    threshold=am_rules.min_wall_thickness,
                )
            ok2, msg2 = am_rules.check_hole(
                self.config.fuel_manifold_inlet_OD, "fuel manifold inlet boss"
            )
            if not ok2:
                report = report.add(
                    "warning", msg2, metric="fuel_manifold_inlet_OD",
                    value=self.config.fuel_manifold_inlet_OD,
                    threshold=am_rules.min_hole_diameter,
                )
            report = _am_process_error(report, am_rules, self.name)
        return report


# ---------------------------------------------------------------------------
# CoolantOutletManifold
# ---------------------------------------------------------------------------


@dataclass
class CoolantOutletManifold(Component, Renderable):
    """
    Coolant outlet manifold at the upstream (chamber) end of the jacket.

    In a counterflow regen cooling scheme coolant exits the channels at
    the upstream end after absorbing heat from the chamber wall.  This
    manifold collects the heated coolant before it enters the injector.

    Geometry: annular shell wrapped around the chamber jacket OD at
    z ∈ [x_chamber_start, x_chamber_start + axial_extent].  A radial
    outlet boss at θ = π/2 (positive y-axis) connects the coolant exit line.

    Source: NASA SP-8087 (1977) §3.3 — coolant outlet manifold geometry.
    """

    nozzle: "RaoNozzle"
    bilayer: "BilayerWall"
    config: ManifoldConfig = field(default_factory=ManifoldConfig)

    @property
    def name(self) -> str:
        return "CoolantOutletManifold"

    def _geom(self) -> dict:
        cfg = self.config
        r_inner = _r_outer_chamber(self.nozzle, self.bilayer)
        r_outer = r_inner + cfg.coolant_manifold_radial_width
        z_lo = float(self.nozzle.x_chamber_start)
        z_hi = z_lo + cfg.coolant_manifold_axial_extent
        z_ctr = 0.5 * (z_lo + z_hi)
        return {
            "z_lo": z_lo, "z_hi": z_hi, "z_ctr": z_ctr,
            "r_inner": r_inner, "r_outer": r_outer,
        }

    def to_sdf(self, **context: Any) -> "SDF":
        cfg = self.config
        g = self._geom()
        t = cfg.wall_thickness
        shell = annular_shell_sdf(
            r_inner=g["r_inner"],
            r_outer=g["r_outer"],
            z_lo=g["z_lo"],
            z_hi=g["z_hi"],
            wall_thickness=t,
        )
        boss_OD = cfg.coolant_manifold_inlet_OD
        boss_inner_OD = boss_OD - 2.0 * t
        boss = cylindrical_boss_sdf(
            axis_origin=(0.0, g["r_outer"], g["z_ctr"]),
            axis_direction=(0.0, 1.0, 0.0),
            OD=boss_OD,
            length=cfg.boss_protrusion,
            inner_OD=boss_inner_OD if boss_inner_OD > 1e-9 else None,
        )
        return union(shell, boss)

    def bbox(self, **context: Any) -> tuple[np.ndarray, np.ndarray]:
        g = self._geom()
        r = g["r_outer"]
        return (
            np.array([-r, -r, g["z_lo"]]),
            np.array([r, r + self.config.boss_protrusion, g["z_hi"]]),
        )

    def verify(self, am_rules: "AMRules | None" = None) -> VerificationReport:
        report = VerificationReport.ok(self.name)
        if am_rules is not None:
            t = self.config.wall_thickness
            ok, msg = am_rules.check_wall_thickness(t, "coolant outlet manifold wall")
            if not ok:
                report = report.add(
                    "warning", msg, metric="wall_thickness", value=t,
                    threshold=am_rules.min_wall_thickness,
                )
            ok2, msg2 = am_rules.check_hole(
                self.config.coolant_manifold_inlet_OD, "coolant outlet boss"
            )
            if not ok2:
                report = report.add(
                    "warning", msg2, metric="coolant_manifold_inlet_OD",
                    value=self.config.coolant_manifold_inlet_OD,
                    threshold=am_rules.min_hole_diameter,
                )
            report = _am_process_error(report, am_rules, self.name)
        return report


# ---------------------------------------------------------------------------
# CoolantInletManifold
# ---------------------------------------------------------------------------


@dataclass
class CoolantInletManifold(Component, Renderable):
    """
    Coolant inlet manifold at the downstream (nozzle exit) end of the jacket.

    In a counterflow regen cooling scheme coolant enters the channels at
    the nozzle exit end (the coldest station) and flows upstream.  This
    manifold distributes incoming coolant to all channels simultaneously.

    Geometry: annular shell wrapped around the nozzle exit jacket OD at
    z ∈ [x_exit − axial_extent/2, x_exit + axial_extent/2].  A radial inlet
    boss at θ = π/2 (positive y-axis) connects the coolant feed line.

    Source: NASA SP-8087 (1977) §3.3 — coolant inlet manifold geometry.
    """

    nozzle: "RaoNozzle"
    bilayer: "BilayerWall"
    config: ManifoldConfig = field(default_factory=ManifoldConfig)

    @property
    def name(self) -> str:
        return "CoolantInletManifold"

    def _geom(self) -> dict:
        cfg = self.config
        r_inner = _r_outer_exit(self.nozzle, self.bilayer)
        r_outer = r_inner + cfg.coolant_manifold_radial_width
        half = cfg.coolant_manifold_axial_extent / 2.0
        z_ctr = float(self.nozzle.x_exit)
        z_lo = z_ctr - half
        z_hi = z_ctr + half
        return {
            "z_lo": z_lo, "z_hi": z_hi, "z_ctr": z_ctr,
            "r_inner": r_inner, "r_outer": r_outer,
        }

    def to_sdf(self, **context: Any) -> "SDF":
        cfg = self.config
        g = self._geom()
        t = cfg.wall_thickness
        shell = annular_shell_sdf(
            r_inner=g["r_inner"],
            r_outer=g["r_outer"],
            z_lo=g["z_lo"],
            z_hi=g["z_hi"],
            wall_thickness=t,
        )
        boss_OD = cfg.coolant_manifold_inlet_OD
        boss_inner_OD = boss_OD - 2.0 * t
        boss = cylindrical_boss_sdf(
            axis_origin=(0.0, g["r_outer"], g["z_ctr"]),
            axis_direction=(0.0, 1.0, 0.0),
            OD=boss_OD,
            length=cfg.boss_protrusion,
            inner_OD=boss_inner_OD if boss_inner_OD > 1e-9 else None,
        )
        return union(shell, boss)

    def bbox(self, **context: Any) -> tuple[np.ndarray, np.ndarray]:
        g = self._geom()
        r = g["r_outer"]
        return (
            np.array([-r, -r, g["z_lo"]]),
            np.array([r, r + self.config.boss_protrusion, g["z_hi"]]),
        )

    def verify(self, am_rules: "AMRules | None" = None) -> VerificationReport:
        report = VerificationReport.ok(self.name)
        if am_rules is not None:
            t = self.config.wall_thickness
            ok, msg = am_rules.check_wall_thickness(t, "coolant inlet manifold wall")
            if not ok:
                report = report.add(
                    "warning", msg, metric="wall_thickness", value=t,
                    threshold=am_rules.min_wall_thickness,
                )
            ok2, msg2 = am_rules.check_hole(
                self.config.coolant_manifold_inlet_OD, "coolant inlet boss"
            )
            if not ok2:
                report = report.add(
                    "warning", msg2, metric="coolant_manifold_inlet_OD",
                    value=self.config.coolant_manifold_inlet_OD,
                    threshold=am_rules.min_hole_diameter,
                )
            report = _am_process_error(report, am_rules, self.name)
        return report
