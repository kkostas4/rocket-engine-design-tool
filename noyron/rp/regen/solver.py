"""
solver.py
=========

Compatibility shim — re-exports solve_regen from the HeatExchanger kernel.

The regenerative cooling solver has been lifted into a domain-agnostic
HeatExchanger component in physics/heat_exchanger/.  Rocket regen is now
one instance (BartzHotSide + DittusBoelterModel + RegenChannelGeometry).

All existing ``from noyron.rp.regen.solver import solve_regen`` imports
continue to work.  Use physics.heat_exchanger.solve_rocket_regen for new
code; physics.heat_exchanger.solve_heat_exchanger for non-rocket HX.

Deprecation
-----------
This module is a backward-compatibility shim.  Import directly from
physics.heat_exchanger in new code:

    from physics.heat_exchanger import solve_rocket_regen
"""

from __future__ import annotations

from typing import Any

from noyron.rp.nozzle.rao import RaoNozzle
from noyron.rp.regen.bartz import CombustionGas
from noyron.rp.regen.coolant import CoolingChannelGeometry, CoolantInlet, WallMaterial


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def solve_regen(
    *,
    nozzle: RaoNozzle,
    gas: CombustionGas,
    channel_geom: CoolingChannelGeometry,
    coolant_inlet: CoolantInlet,
    wall_material: WallMaterial,
    direction: str = "counterflow",
    n_stations: int = 200,
    tol: float = 1.0,
) -> dict[str, Any]:
    """
    1-D coupled regenerative cooling solver.

    Backward-compatible shim that delegates to
    physics.heat_exchanger.rocket_regen.solve_rocket_regen.  All parameters
    and return keys are identical to the pre-refactor implementation.

    Parameters
    ----------
    nozzle : RaoNozzle
    gas : CombustionGas
    channel_geom : CoolingChannelGeometry
    coolant_inlet : CoolantInlet
    wall_material : WallMaterial
    direction : str     'counterflow' or 'coflow'.
    n_stations : int    Axial discretisation count.
    tol : float         T_wall_gas convergence tolerance [K].

    Returns
    -------
    dict
        x, r, T_wall_gas, T_wall_coolant, T_coolant, P_coolant,
        q_dot, h_g, h_c, dP_total — same as solve_rocket_regen.
    """
    from noyron.rp.regen.rocket_regen import solve_rocket_regen
    return solve_rocket_regen(
        nozzle=nozzle,
        gas=gas,
        channel_geom=channel_geom,
        coolant_inlet=coolant_inlet,
        wall_material=wall_material,
        direction=direction,
        n_stations=n_stations,
        tol=tol,
    )
