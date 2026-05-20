"""
Regenerative cooling subsystem: Bartz h, channel sizing, T_wall.

All public names from the pre-refactor API remain importable here.
New code should import from physics.heat_exchanger directly.
"""

from physics.regen.bartz import CombustionGas, mach_from_area_ratio
from physics.regen.coolant import (
    CoolingChannelGeometry,
    CoolantInlet,
    GRCOP42,
    INCONEL718,
    WallMaterial,
)
from physics.regen.sizing import ChannelSizingProblem, size_channels
from physics.regen.solver import solve_regen
