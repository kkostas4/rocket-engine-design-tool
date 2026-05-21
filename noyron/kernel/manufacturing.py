"""
manufacturing.py
================

Manufacturing process rules as a named component.  Every constraint is a
named, traceable, frozen object rather than a scattered magic number.
Implements Noyron's "manufacturing process rules" principle.

References
----------
Gradl, P., Tinker, D.C., et al. (2022). "Robust Metal Additive Manufacturing
    Process Selection and Development for Aerospace Components." J. Mater. Eng.
    Perform., 31, 6013-6044.  https://doi.org/10.1007/s11665-022-06850-0

Gradl, P.R. and Protz, C.S. (2020). "Technology Advancements for Channel Wall
    Nozzle Manufacturing and Hot-fire Testing in Support of Liquid Rocket
    Engines." Acta Astronautica, 174, 148-158.
    https://doi.org/10.1016/j.actaastro.2020.04.067

ASTM F3055-14(2022). "Standard Specification for Additive Manufacturing Nickel
    Alloy (UNS N07718) with Powder Bed Fusion."

Smid, P. (2008). "CNC Programming Handbook," 3rd ed. Industrial Press, Ch. 8.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class AMRules:
    """
    Manufacturing process envelope for one process + material combination.

    Components call check_wall_thickness / check_channel / check_feature /
    check_hole inside verify() when am_rules is not None.

    All dimensions SI (metres).  Angles in degrees.  source must be non-empty.
    """

    process_name: str
    material_family: str

    # Minimum feature sizes (m)
    min_wall_thickness: float
    min_feature_size: float
    min_channel_width: float
    min_channel_height: float
    min_hole_diameter: float

    # Geometric limits
    max_internal_aspect_ratio: float
    max_overhang_angle_no_support: float   # degrees from horizontal
    max_unsupported_span: float            # m

    # Surface finish
    surface_roughness_Ra: float            # m, as-printed/as-machined

    # Citation -- must be non-empty
    source: str

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError(
                "AMRules.source must be non-empty -- every rule must be traceable"
            )

    # ------------------------------------------------------------------
    # Check helpers -- return (ok: bool, message: str) or list thereof
    # ------------------------------------------------------------------

    def check_wall_thickness(self, t: float, where: str = "") -> tuple[bool, str]:
        """Return (ok, message) for a wall of thickness t [m]."""
        loc = f" ({where})" if where else ""
        if t < self.min_wall_thickness:
            return (
                False,
                f"Wall thickness {t * 1e3:.3f} mm{loc} < "
                f"{self.min_wall_thickness * 1e3:.3f} mm minimum "
                f"[{self.process_name}, {self.material_family}]",
            )
        return (True, "")

    def check_channel(
        self, w: float, h: float, where: str = ""
    ) -> list[tuple[bool, str]]:
        """Return list of (ok, message) for a channel of width w and height h [m]."""
        issues: list[tuple[bool, str]] = []
        loc = f" ({where})" if where else ""
        if w < self.min_channel_width:
            issues.append(
                (
                    False,
                    f"Channel width {w * 1e3:.3f} mm{loc} < "
                    f"{self.min_channel_width * 1e3:.3f} mm minimum "
                    f"[{self.process_name}, {self.material_family}]",
                )
            )
        if h < self.min_channel_height:
            issues.append(
                (
                    False,
                    f"Channel height {h * 1e3:.3f} mm{loc} < "
                    f"{self.min_channel_height * 1e3:.3f} mm minimum "
                    f"[{self.process_name}, {self.material_family}]",
                )
            )
        if w > 0.0 and h / w > self.max_internal_aspect_ratio:
            issues.append(
                (
                    False,
                    f"Channel aspect ratio h/w = {h / w:.2f}{loc} > "
                    f"{self.max_internal_aspect_ratio:.1f} maximum "
                    f"[{self.process_name}, {self.material_family}]",
                )
            )
        return issues

    def check_feature(self, size: float, where: str = "") -> tuple[bool, str]:
        """Return (ok, message) for a feature of given size [m]."""
        loc = f" ({where})" if where else ""
        if size < self.min_feature_size:
            return (
                False,
                f"Feature size {size * 1e3:.3f} mm{loc} < "
                f"{self.min_feature_size * 1e3:.3f} mm minimum "
                f"[{self.process_name}, {self.material_family}]",
            )
        return (True, "")

    def check_hole(self, diameter: float, where: str = "") -> tuple[bool, str]:
        """Return (ok, message) for a hole/orifice of given diameter [m]."""
        loc = f" ({where})" if where else ""
        if diameter < self.min_hole_diameter:
            return (
                False,
                f"Hole diameter {diameter * 1e3:.3f} mm{loc} < "
                f"{self.min_hole_diameter * 1e3:.3f} mm minimum "
                f"[{self.process_name}, {self.material_family}]",
            )
        return (True, "")


# ---------------------------------------------------------------------------
# Process presets
# ---------------------------------------------------------------------------

LPBF_GRCOP = AMRules(
    process_name="LPBF",
    material_family="GRCop-42",
    min_wall_thickness=0.5e-3,   # matches current _T_LINER_M / ChannelSizingProblem default
    min_feature_size=0.3e-3,     # matches current t_land_min default
    min_channel_width=0.6e-3,    # matches current min_channel_width default
    min_channel_height=0.6e-3,
    min_hole_diameter=0.5e-3,
    max_internal_aspect_ratio=8.0,  # matches current max_aspect_ratio default
    max_overhang_angle_no_support=45.0,
    max_unsupported_span=2e-3,
    surface_roughness_Ra=12e-6,
    source=(
        "NASA GRCop-42 LPBF process development: Gradl, P.R. and Protz, C.S. (2020). "
        "Acta Astronautica 174, 148-158. "
        "Wall/feature minimums validated against MSFC RAMPT test articles."
    ),
)

LPBF_INCONEL = AMRules(
    process_name="LPBF",
    material_family="Inconel 718",
    min_wall_thickness=0.5e-3,
    min_feature_size=0.3e-3,
    min_channel_width=0.5e-3,
    min_channel_height=0.5e-3,
    min_hole_diameter=0.5e-3,
    max_internal_aspect_ratio=8.0,
    max_overhang_angle_no_support=45.0,
    max_unsupported_span=2e-3,
    surface_roughness_Ra=10e-6,
    source=(
        "NASA MSFC RAMPT program envelope for LPBF IN718: Gradl et al. (2022). "
        "J. Mater. Eng. Perform. 31, 6013-6044. "
        "ASTM F3055-14(2022) minimum feature guidance."
    ),
)

BINDER_JET_INCONEL = AMRules(
    process_name="Binder Jet + Sinter",
    material_family="Inconel 718 (sintered)",
    min_wall_thickness=0.4e-3,
    min_feature_size=0.15e-3,
    min_channel_width=0.4e-3,
    min_channel_height=0.4e-3,
    min_hole_diameter=0.3e-3,
    max_internal_aspect_ratio=10.0,
    max_overhang_angle_no_support=90.0,   # no support needed
    max_unsupported_span=5e-3,
    surface_roughness_Ra=15e-6,            # rougher than LPBF
    source=(
        "ExOne / Desktop Metal Inconel 718 binder jet datasheets; "
        "Gradl et al. (2022) binder jet survey. "
        "NOTE: design must account for ~18-20 % sintering shrinkage in CAD."
    ),
)

SUBTRACTIVE_INCONEL = AMRules(
    process_name="CNC machining",
    material_family="Inconel 718 (wrought)",
    min_wall_thickness=1.0e-3,
    min_feature_size=0.5e-3,
    min_channel_width=2.0e-3,    # endmill minimum; internal channels typically require EDM
    min_channel_height=10.0e-3,  # deep internal channels require EDM or gun drill
    min_hole_diameter=0.5e-3,
    max_internal_aspect_ratio=4.0,
    max_overhang_angle_no_support=90.0,
    max_unsupported_span=50e-3,
    surface_roughness_Ra=1.6e-6,
    source=(
        "Standard CNC envelope: Smid, P. (2008). CNC Programming Handbook, 3rd ed., "
        "Industrial Press, Ch. 8. "
        "NOTE: internal regenerative cooling channels not feasible subtractively "
        "without brazing or EDM."
    ),
)

AM_RULES_PRESETS: dict[str, AMRules] = {
    "lpbf-grcop": LPBF_GRCOP,
    "lpbf-inconel": LPBF_INCONEL,
    "binder-jet-inconel": BINDER_JET_INCONEL,
    "subtractive-inconel": SUBTRACTIVE_INCONEL,
}
