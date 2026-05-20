"""
rocket_regen.py
===============

Rocket regenerative cooling as a HeatExchanger client.

This module wraps the rocket-specific physics (Bartz gas-side h, nozzle
contour geometry, GRCop-42 wall conductivity) behind the abstract
HotSideModel / HeatExchangerGeometry interfaces defined in base.py, then
calls the domain-agnostic solve_heat_exchanger().

In Noyron terms this is "Noyron RP using the HX kernel": RP is a client of
HX, not the other way around.  The generic kernel has no knowledge of nozzles
or combustion gases.

Sources
-------
Bartz gas-side h:
    Bartz, D.R. (1957). "A Simple Equation for Rapid Estimation of Rocket
    Nozzle Convective Heat Transfer Coefficients." Jet Propulsion, Vol. 27,
    No. 1, pp. 49-51.  (See physics/regen/bartz.py for full derivation.)

Adiabatic wall temperature:
    Sutton, G.P. and Biblarz, O. (2017). "Rocket Propulsion Elements," 9th ed.
    Eq. (8-23): T_aw = Tc * (1 + r_f*(gamma-1)/2 * M^2) / (1 + (gamma-1)/2 * M^2)
    where r_f = Pr^(1/3) for turbulent flow.

Channel geometry:
    NASA SP-8087 (1977). "Liquid Rocket Engine Fluid-Cooled Combustion
    Chambers." Sec. 3.1.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from physics.heat_exchanger.base import (
    ColdSideModel,
    ColdSideResult,
    FluidProperties,
    HeatExchangerGeometry,
    HotSideModel,
    HotSideState,
)
from physics.heat_exchanger.correlations import DittusBoelterModel
from physics.heat_exchanger.solver import solve_heat_exchanger
from physics.regen.bartz import CombustionGas, adiabatic_wall_temperature, bartz_h
from physics.regen.coolant import CoolingChannelGeometry, CoolantInlet, WallMaterial
from physics.nozzle.rao import RaoNozzle


# ---------------------------------------------------------------------------
# HotSideModel: Bartz + adiabatic wall temperature
# ---------------------------------------------------------------------------


class BartzHotSide(HotSideModel):
    """
    Hot-side model using the Bartz (1957) correlation with sigma correction.

    Provides T_aw (adiabatic wall temperature as recovery temperature) and
    h_g = bartz_h (sigma-corrected) as required by the HotSideModel interface.

    Source: Bartz, D.R. (1957). Jet Propulsion, Vol. 27, No. 1, pp. 49-51.
    Validity: subsonic chamber + supersonic nozzle; gamma 1.1-1.3; Pr ~0.5-1.0.

    Parameters
    ----------
    nozzle : RaoNozzle
        Nozzle contour providing r(x) and area_ratio(x).
    gas : CombustionGas
        Combustion gas stagnation properties (Tc, Pc, gamma, MW, cp, mu, Pr, c*).
    """

    def __init__(self, nozzle: RaoNozzle, gas: CombustionGas) -> None:
        self._nozzle = nozzle
        self._gas = gas

    def evaluate(
        self,
        x: float,
        geometry: HeatExchangerGeometry,
        T_wall: float,
    ) -> HotSideState:
        """
        Return T_aw and Bartz h_g at axial station x given the wall temperature.

        The sigma correction in bartz_h makes h_g mildly T_wall-dependent, so
        this must be called inside the inner fixed-point loop.

        Parameters
        ----------
        x : float           Axial station [m].
        geometry : HeatExchangerGeometry   Not used; present for API symmetry.
        T_wall : float      Current gas-side wall temperature [K].

        Returns
        -------
        HotSideState
            T_recovery = T_aw, h_convective = h_g (Bartz sigma-corrected).
        """
        T_aw = adiabatic_wall_temperature(x, self._nozzle, self._gas)
        h_g = bartz_h(x=x, nozzle=self._nozzle, gas=self._gas, T_wall=T_wall)
        return HotSideState(T_recovery=T_aw, h_convective=h_g)


# ---------------------------------------------------------------------------
# HeatExchangerGeometry: nozzle + cooling channel geometry
# ---------------------------------------------------------------------------


class RegenChannelGeometry(HeatExchangerGeometry):
    """
    HeatExchangerGeometry for a rocket regen cooling system.

    Wraps CoolingChannelGeometry (channel cross-section) and RaoNozzle
    (hot-side area via nozzle radius) behind the abstract geometry interface.

    Parameters
    ----------
    channel_geom : CoolingChannelGeometry
        Cooling channel cross-section (width, height, wall_thickness, ...).
    nozzle : RaoNozzle
        Nozzle contour (provides r(x) for the hot-side arc area per channel).
    wall_material : WallMaterial
        Wall thermal conductivity k(T) and service temperature limit.
    """

    def __init__(
        self,
        channel_geom: CoolingChannelGeometry,
        nozzle: RaoNozzle,
        wall_material: WallMaterial,
    ) -> None:
        self._ch = channel_geom
        self._nozzle = nozzle
        self._mat = wall_material

    @property
    def n_channels(self) -> int:
        return self._ch.n_channels

    def wetted_perimeter_hot(self, x: float) -> float:
        """
        Gas-side arc length per channel per unit axial length [m].

        = 2π · r(x) / n_channels.

        Source: NASA SP-8087 (1977), Sec. 3.1.
        """
        return 2.0 * math.pi * self._nozzle.r(x) / self._ch.n_channels

    def wetted_perimeter_cold(self, x: float) -> float:
        """Coolant-side 3-sided wetted perimeter per channel [m]: w + 2H."""
        return self._ch.coolant_side_perimeter(x)

    def cross_section_cold(self, x: float) -> float:
        """Channel flow area per channel [m²]: w × H."""
        return self._ch.flow_area(x)

    def hydraulic_diameter_cold(self, x: float) -> float:
        """Hydraulic diameter [m]: 4A/(2(w+H)) for fully-enclosed rect channel."""
        return self._ch.hydraulic_diameter(x)

    def wall_thickness(self, x: float) -> float:
        """Gas-side liner thickness [m]."""
        return self._ch.wall_thickness(x)

    def fin_efficiency(
        self, x: float, h_cold: float, k_wall: float | None = None
    ) -> float:
        """
        Straight-fin efficiency for the rib acting as a fin.

        Source: Incropera et al. (2011) Table 3.4, Case A, Eq. (3.80).

        Parameters
        ----------
        x : float       Axial station [m].
        h_cold : float  Coolant-side h [W/(m²·K)].
        k_wall : float  Wall conductivity [W/(m·K)].  If None, uses k at 300 K.
        """
        if k_wall is None:
            k_wall = self._mat.k(300.0)
        return self._ch.fin_efficiency(x, h_cold, k_wall)

    def overall_surface_efficiency(self, x: float, eta_fin: float) -> float:
        """Overall surface efficiency. Source: Incropera et al. (2011) Eq. (3.86)."""
        return self._ch.overall_surface_efficiency(x, eta_fin)

    def wall_conductivity(self, x: float, T_wall: float) -> float:
        """Wall thermal conductivity k(T_wall) [W/(m·K)]."""
        return self._mat.k(T_wall)

    def temperature_limit(self) -> float:
        """Continuous service temperature limit [K]."""
        return self._mat.T_max_continuous

    def material_name(self) -> str:
        """Wall material name for warning messages."""
        return self._mat.name


# ---------------------------------------------------------------------------
# Convenience wrapper: solve_rocket_regen
# ---------------------------------------------------------------------------


def solve_rocket_regen(
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
    1-D regenerative cooling solver — rocket-regen adapter for solve_heat_exchanger.

    This function constructs BartzHotSide, DittusBoelterModel, and
    RegenChannelGeometry from the rocket-specific inputs, then delegates to
    the domain-agnostic solve_heat_exchanger().

    The return dict is augmented with an 'r' key (nozzle wall radius at each
    station) for backward compatibility with the structural subsystem.

    Parameters
    ----------
    nozzle : RaoNozzle
        Nozzle contour.
    gas : CombustionGas
        Combustion gas stagnation properties.
    channel_geom : CoolingChannelGeometry
        Cooling channel geometry (callables of x).
    coolant_inlet : CoolantInlet
        Coolant inlet state.
    wall_material : WallMaterial
        Gas-side wall material (k(T) and T_max_continuous).
    direction : str
        'counterflow' or 'coflow'.  Default 'counterflow'.
    n_stations : int
        Axial discretisation count.  Default 200.
    tol : float
        T_wall_gas convergence tolerance [K].  Default 1.0.

    Returns
    -------
    dict
        Same keys as solve_heat_exchanger plus:
          'r' : ndarray [m]  -- nozzle wall radius at each station.
    """
    hot_model = BartzHotSide(nozzle=nozzle, gas=gas)
    cold_model = DittusBoelterModel()
    geometry = RegenChannelGeometry(
        channel_geom=channel_geom,
        nozzle=nozzle,
        wall_material=wall_material,
    )

    result = solve_heat_exchanger(
        hot_model=hot_model,
        cold_model=cold_model,
        geometry=geometry,
        x_start=nozzle.x_inlet,
        x_end=nozzle.x_exit,
        coolant_inlet=coolant_inlet,
        direction=direction,
        n_stations=n_stations,
        tol=tol,
    )

    # Add 'r' for structural subsystem backward compatibility.
    x_arr = result["x"]
    result["r"] = np.array([nozzle.r(float(x)) for x in x_arr])
    return result
