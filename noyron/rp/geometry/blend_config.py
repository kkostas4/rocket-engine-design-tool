"""
blend_config.py
===============

Per-interface blend radii for smooth-union SDF assembly.

All values in metres.  Set a radius to 0.0 to get a sharp (hard) boolean
operation at that interface.

Critical constraint (Quilez smooth_min note):
    blend radius < 0.25 × (smallest adjacent feature size)
    or the feature will vanish into the blend region.

Source: Quilez, I. (2023). "Smooth Minimum." iquilezles.org/articles/smin
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BlendConfig:
    """
    Frozen blend-radius configuration for the engine assembly SDF.

    Each field controls the fillet radius (in metres) at one class of
    geometric interface.  Features smaller than 4× the blend radius may
    be distorted; set to 0 for sharp booleans at those interfaces.

    Parameters
    ----------
    boss_to_manifold : float
        Boss stub joins the manifold ring body.  Default 1.5 mm.
    manifold_to_chamber : float
        LOX dome / fuel manifold meets the injector face-plate plane.
        Default 2.0 mm.
    manifold_to_jacket : float
        Ring manifold shell meets the outer coolant jacket.  Used when
        unioning manifolds into the full-engine assembly.  Default 1.5 mm.
    chamber_to_nozzle : float
        Reserved for future chamber-to-nozzle contour transition.
        Default 1.0 mm.
    pintle_to_face_plate : float
        Pintle post base joins the injector face plate.  Conservative
        (0.3 mm) to preserve the 0.235 mm LOX annulus gap.  Default 0.3 mm.
    orifice_difference : float
        Blend radius when subtracting fuel orifice bores.  Must be 0 to
        preserve sharp orifice walls (0.82 mm diameter).  Default 0.0.
    channel_difference : float
        Blend radius when subtracting cooling channel voids.  Must be 0 to
        preserve channel corner geometry.  Default 0.0.
    default : float
        Generic top-level assembly blend (chamber+injector union, any
        interface not covered by a specific field).  Default 1.0 mm.
    """

    boss_to_manifold: float = 1.5e-3
    manifold_to_chamber: float = 2.0e-3
    manifold_to_jacket: float = 1.5e-3
    chamber_to_nozzle: float = 1.0e-3
    pintle_to_face_plate: float = 0.3e-3
    orifice_difference: float = 0.0      # SHARP — preserve 0.82 mm orifice bores
    channel_difference: float = 0.0      # SHARP — preserve channel corners
    default: float = 1.0e-3
