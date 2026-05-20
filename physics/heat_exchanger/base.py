"""
base.py
=======

Abstract base classes for the domain-agnostic heat exchanger kernel.

This module is importable in a fresh Python process with no rocket-specific
dependencies.  It imports only from the standard library and dataclasses.

Design
------
The abstraction separates three concerns:

  HotSideModel    -- how the hot stream transfers heat to the wall.
                     Rocket: Bartz + adiabatic wall temperature.
                     Generic: any h(x, T_wall) callable.

  ColdSideModel   -- how the cold stream removes heat from the wall.
                     Implements the heat transfer correlation (Dittus-Boelter,
                     Sieder-Tate, Gnielinski, ...) as a callable object.

  HeatExchangerGeometry  -- cross-section, areas, fins, conductivity at each
                     axial station.
                     Rocket: nozzle radius, channel dimensions, GRCop-42 k(T).
                     Plate-fin: constant rectangular cross-section, fixed k.

The 1D solver (solver.py) takes one of each and marches along the axis,
threading the coolant state station-to-station with no knowledge of which
physical system it is modelling.

Sources
-------
Incropera, F.P. et al. (2011). "Fundamentals of Heat and Mass Transfer,"
    7th ed., Wiley.  Ch. 11 -- heat exchanger design methodology.
Noyron product family: leap71.com/noyron (HX kernel / RP kernel separation).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass
class FluidProperties:
    """Thermophysical properties of the coolant at one (T, P) state."""

    rho: float   # density [kg/m³]
    cp: float    # specific heat [J/(kg·K)]
    mu: float    # dynamic viscosity [Pa·s]
    k: float     # thermal conductivity [W/(m·K)]
    Pr: float    # Prandtl number [-]


@dataclass
class HotSideState:
    """
    Hot-side thermal state at one axial station and one wall temperature.

    Parameters
    ----------
    T_recovery : float
        Driving temperature for heat transfer [K].
        Rocket: adiabatic wall temperature T_aw (recovery factor applied).
        Plate-fin: hot-fluid bulk temperature.
    h_convective : float
        Gas-side (hot-side) convective coefficient [W/(m²·K)].
    q_flux : float
        Heat flux at the gas-side wall surface [W/m²].
        Filled by the solver after coupling; not set by HotSideModel.
    """

    T_recovery: float
    h_convective: float
    q_flux: float = 0.0


@dataclass
class ColdSideResult:
    """
    Cold-side thermal and flow result at one axial station.

    Parameters
    ----------
    h_convective : float
        Coolant-side convective coefficient [W/(m²·K)].
    Re : float
        Coolant Reynolds number [-].  Used for pressure drop (Haaland).
    V : float
        Mean coolant velocity per channel [m/s].  Used for Darcy-Weisbach.
    """

    h_convective: float
    Re: float
    V: float


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------


class HotSideModel(ABC):
    """
    Model for the hot-side convective heat transfer coefficient and
    recovery temperature.

    Rocket implementation: BartzHotSide in rocket_regen.py.
    Plate-fin implementation: a ConstantHotSide with fixed h and T.
    """

    @abstractmethod
    def evaluate(
        self,
        x: float,
        geometry: "HeatExchangerGeometry",
        T_wall: float,
    ) -> HotSideState:
        """
        Evaluate hot-side state at axial position x given the current
        gas-side wall temperature.

        Parameters
        ----------
        x : float
            Axial coordinate [m].
        geometry : HeatExchangerGeometry
            Local geometry (may be queried by hot-side model if needed).
        T_wall : float
            Current estimate of the gas-side wall temperature [K].

        Returns
        -------
        HotSideState
            T_recovery and h_convective at this station and wall temperature.
        """


class ColdSideModel(ABC):
    """
    Model for the cold-side (coolant) convective heat transfer coefficient.

    Implementations live in correlations.py:
      DittusBoelterModel  -- Nu = 0.023 Re^0.8 Pr^0.4
    """

    @abstractmethod
    def evaluate(
        self,
        x: float,
        properties: FluidProperties,
        m_dot_per_channel: float,
        geometry: "HeatExchangerGeometry",
    ) -> ColdSideResult:
        """
        Evaluate coolant-side heat transfer coefficient at axial station x.

        Parameters
        ----------
        x : float
            Axial coordinate [m].
        properties : FluidProperties
            Coolant thermophysical properties at the current bulk state.
        m_dot_per_channel : float
            Mass flow rate per channel [kg/s].
        geometry : HeatExchangerGeometry
            Local geometry (provides cross_section_cold, hydraulic_diameter_cold).

        Returns
        -------
        ColdSideResult
            h_convective, Re, V at this station.
        """


class HeatExchangerGeometry(ABC):
    """
    Geometry of the heat exchanger at each axial station.

    All per-area quantities are per-channel, per unit axial length, so that
    the solver can work in W/m (heat rate per unit axial length per channel).

    Rocket implementation: RegenChannelGeometry in rocket_regen.py.
    Plate-fin implementation: define a subclass with constant cross-section.
    """

    @property
    @abstractmethod
    def n_channels(self) -> int:
        """Number of parallel cold-side channels."""

    @abstractmethod
    def wetted_perimeter_hot(self, x: float) -> float:
        """
        Hot-side wetted perimeter per channel [m].

        This is the arc length (per unit axial length) over which the hot
        stream exchanges heat with the wall, divided by n_channels.

        Rocket: 2π·r(x) / n_channels  (nozzle circumference per channel).
        Plate-fin: channel width / n_channels.
        """

    @abstractmethod
    def wetted_perimeter_cold(self, x: float) -> float:
        """
        Cold-side wetted perimeter per channel [m].

        For a 3-sided rectangular regen channel: w + 2H.
        For a fully enclosed rectangular duct: 2(w+H).
        """

    @abstractmethod
    def cross_section_cold(self, x: float) -> float:
        """Cold-side flow cross-sectional area per channel [m²]."""

    @abstractmethod
    def hydraulic_diameter_cold(self, x: float) -> float:
        """
        Hydraulic diameter of the cold-side channel [m].

        D_h = 4 * A_c / P_wetted.
        """

    @abstractmethod
    def wall_thickness(self, x: float) -> float:
        """Gas-side wall thickness (hot wall to channel floor) [m]."""

    @abstractmethod
    def fin_efficiency(
        self, x: float, h_cold: float, k_wall: float | None = None
    ) -> float:
        """
        Straight-fin efficiency for the ribs/fins at axial station x.

        Source: Incropera et al. (2011) Table 3.4 Case A, Eq. (3.80).

        Parameters
        ----------
        x : float
            Axial coordinate [m].
        h_cold : float
            Coolant-side convective coefficient [W/(m²·K)].
        k_wall : float or None
            Wall thermal conductivity [W/(m·K)].  If None, the geometry
            object uses its own stored value (e.g. at a reference temperature).
            The solver always passes the current k_wall explicitly.

        Returns
        -------
        float
            Fin efficiency in [0, 1].
        """

    @abstractmethod
    def overall_surface_efficiency(self, x: float, eta_fin: float) -> float:
        """
        Overall surface efficiency accounting for fin contribution.

        Source: Incropera et al. (2011) Eq. (3.86).
        """

    @abstractmethod
    def wall_conductivity(self, x: float, T_wall: float) -> float:
        """
        Wall thermal conductivity at axial station x and wall temperature
        T_wall [W/(m·K)].

        For temperature-independent materials: ignore T_wall and return a constant.
        For GRCop-42: use the k(T) interpolant from WallMaterial.
        """

    @abstractmethod
    def temperature_limit(self) -> float:
        """Continuous service temperature limit for the wall material [K]."""

    @abstractmethod
    def material_name(self) -> str:
        """Human-readable name of the hot-wall material (for warning messages)."""
