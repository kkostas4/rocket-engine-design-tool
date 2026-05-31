"""
heat_exchanger
==============

Domain-agnostic 1D heat exchanger kernel.  Rocket regenerative cooling is
one instance; the same solver can drive plate-fin recuperators, tube-in-tube
oil coolers, or any counterflow / coflow HX.

Mirrors the Noyron HX domain module (leap71.com/noyron): HX sits next to RP
on a shared physics kernel; RP (rocket propulsion) is an HX client, not the
other way around.

Public API
----------
Base classes (importable without rocket-specific imports):
    HotSideModel, ColdSideModel, HeatExchangerGeometry
    HotSideState, ColdSideResult, FluidProperties

Generic solver:
    solve_heat_exchanger

Correlations:
    dittus_boelter, sieder_tate, gnielinski, haaland_f
    DittusBoelterModel (ColdSideModel implementation)

Rocket-regen adapter:
    BartzHotSide, RegenChannelGeometry, solve_rocket_regen
"""

from noyron.kernel.heat_exchanger.base import (
    ColdSideResult,
    FluidProperties,
    HeatExchangerGeometry,
    HotSideModel,
    HotSideState,
    ColdSideModel,
)
from noyron.kernel.heat_exchanger.correlations import (
    DittusBoelterModel,
    dittus_boelter,
    gnielinski,
    haaland_f,
    sieder_tate,
)
from noyron.kernel.heat_exchanger.solver import solve_heat_exchanger

# Rocket-regen names are loaded lazily so that importing this package does NOT
# trigger physics.nozzle/physics.regen as a side effect.
_ROCKET_REGEN_NAMES = frozenset({"BartzHotSide", "RegenChannelGeometry", "solve_rocket_regen"})


def __getattr__(name: str):
    if name in _ROCKET_REGEN_NAMES:
        from physics.regen import rocket_regen as _rr
        return getattr(_rr, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Base classes
    "FluidProperties",
    "HotSideState",
    "ColdSideResult",
    "HotSideModel",
    "ColdSideModel",
    "HeatExchangerGeometry",
    # Generic solver
    "solve_heat_exchanger",
    # Correlations
    "dittus_boelter",
    "sieder_tate",
    "gnielinski",
    "haaland_f",
    "DittusBoelterModel",
    # Rocket adapter
    "BartzHotSide",
    "RegenChannelGeometry",
    "solve_rocket_regen",
]
