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

Names are loaded lazily so that ``import noyron.kernel.component`` (or any
``from physics.* import`` that pulls in the kernel ABCs) does NOT eagerly
trigger a full load of ``noyron.engine``.  This prevents the circular
import:  physics.nozzle.rao -> noyron.kernel.component -> (noyron.__init__)
-> noyron.engine -> physics.nozzle.rao.
"""

_ENGINE_NAMES = frozenset({
    "Engine",
    "EngineDesign",
    "EngineRequirements",
    "PropellantPair",
    "HYDROLOX",
    "KEROLOX",
    "METHALOX",
})
_SWEEP_NAMES = frozenset({"SweepConfig", "run_sweep", "cartesian_points"})


def __getattr__(name: str):
    if name in _ENGINE_NAMES:
        from noyron import engine as _e
        return getattr(_e, name)
    if name in _SWEEP_NAMES:
        from noyron import sweep as _s
        return getattr(_s, name)
    raise AttributeError(f"module 'noyron' has no attribute {name!r}")


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
