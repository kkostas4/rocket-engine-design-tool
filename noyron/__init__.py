"""
noyron — computational engine model package.

Top-level entry point::

    from noyron.engine import Engine, EngineRequirements, METHALOX

    req = EngineRequirements(
        thrust=4448.22,
        chamber_pressure=6.8948e6,
        mixture_ratio=3.60,
        propellant=METHALOX,
        expansion_ratio=10.605,
    )
    engine = Engine(req)
    design = engine.design()
    print(engine.summary(design))
"""

from noyron.engine import (
    Engine,
    EngineDesign,
    EngineRequirements,
    PropellantPair,
    HYDROLOX,
    KEROLOX,
    METHALOX,
)
from noyron.sweep import SweepConfig, run_sweep, cartesian_points

__all__ = [
    "Engine",
    "EngineDesign",
    "EngineRequirements",
    "PropellantPair",
    "METHALOX",
    "KEROLOX",
    "HYDROLOX",
    "SweepConfig",
    "run_sweep",
    "cartesian_points",
]
