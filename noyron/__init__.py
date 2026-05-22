"""
noyron — computational engine model package.

Top-level entry point::

    from noyron import Engine, EngineRequirements, METHALOX

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

Architecture
------------
noyron.kernel   Domain-agnostic geometry (SDF, mesh), heat exchanger kernel, utilities.
noyron.rp       Rocket Propulsion domain: thermochem, nozzle, injector, regen, structural.
noyron.hx       Heat Exchanger domain: plate-fin and other generic HX designs.

Names are loaded lazily so that ``import noyron.kernel.component`` does NOT
eagerly trigger a full load of ``noyron.rp.engine``.  This prevents the
circular import:  noyron.rp.nozzle.rao -> noyron.kernel.component ->
(noyron.__init__) -> noyron.rp.engine -> noyron.rp.nozzle.rao.
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
        from noyron.rp import engine as _e
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
