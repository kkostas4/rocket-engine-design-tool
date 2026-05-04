"""
coolant.py
==========

Channel geometry and wall material dataclasses for the 1D regenerative
cooling solver (physics/regen/solver.py).

Sources
-------
Channel geometry model:
    NASA SP-8087 (1977). "Liquid Rocket Engine Fluid-Cooled Combustion
    Chambers." Sec. 3.1 -- channel cross-section definitions.

Fin efficiency:
    Incropera, F.P., DeWitt, D.P., Bergman, T.L., Lavine, A.S. (2011).
    "Fundamentals of Heat and Mass Transfer," 7th ed., Wiley.
    Table 3.4, Case A -- straight rectangular fin, adiabatic tip.
    Eq. (3.86) -- overall surface efficiency.

GRCop-42 thermal conductivity:
    Ellis, D.L., Michal, G., and Dreshfield, R.L. (2013). "GRCop-42:
    A High Conductivity, High Strength Copper Alloy for High Heat Flux
    Applications." NASA/TM-2013-217848. Table 3 (approximate values).
    NOTE: The exact tabulated values from the NASA TM should be verified
    against the published paper before production use.  Values here are
    consistent with the Cu-4wt%Cr-2wt%Nb alloy class.

Inconel 718 thermal conductivity:
    Special Metals Corporation (2007). "INCONEL Alloy 718."
    Publication No. SMC-045. Table of Properties.

GRCop-42 maximum service temperature (810 K continuous):
    Ellis, D.L. (2005). "GRCop-84: A High-Temperature Copper Alloy for
    High-Heat-Flux Applications." NASA/TM-2005-213566. Sec. 3.3.
    NOTE: This reference covers GRCop-84 (Cu-8Cr-4Nb); GRCop-42
    (Cu-4Cr-2Nb) has a similar but slightly different service limit.
    The 810 K figure is widely cited in the regen cooling literature for
    copper alloy chamber liners (see also NASA SP-8087, Ch. 4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.interpolate import interp1d  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# Cooling channel geometry
# ---------------------------------------------------------------------------


@dataclass
class CoolingChannelGeometry:
    """
    Rectangular cooling channel geometry, parametric in axial coordinate x.

    All callables take x [m] and return a value in SI units.  Use
    ``lambda x: const`` for constant-section channels.

    Parameters
    ----------
    n_channels : int
        Number of channels around the circumference.
    width : Callable[[float], float]
        Channel width (circumferential) [m] as a function of x.
    height : Callable[[float], float]
        Channel height (radial depth into the wall) [m] as a function of x.
    wall_thickness : Callable[[float], float]
        Gas-side wall thickness (from hot-gas surface to channel floor) [m].
    land_thickness : Callable[[float], float]
        Rib/land thickness (between adjacent channels) [m].
    """

    n_channels: int
    width: Callable[[float], float]
    height: Callable[[float], float]
    wall_thickness: Callable[[float], float]
    land_thickness: Callable[[float], float]

    def hydraulic_diameter(self, x: float) -> float:
        """D_h = 4 * A_c / P_wetted [m] for fully-enclosed rectangular channel."""
        w = self.width(x)
        H = self.height(x)
        return 4.0 * w * H / (2.0 * (w + H))

    def flow_area(self, x: float) -> float:
        """Channel cross-sectional area A_c = w * H [m2]."""
        return self.width(x) * self.height(x)

    def gas_side_arc(self, x: float, r_wall: float) -> float:
        """
        Gas-side arc length per channel per unit axial length [m/m].

        = 2 * pi * r_wall / n_channels
        Should equal w(x) + t_land(x) for a well-designed channel layout.
        """
        return 2.0 * math.pi * r_wall / self.n_channels

    def coolant_side_perimeter(self, x: float) -> float:
        """
        Coolant-side (3-sided) wetted perimeter per channel [m]:
        channel base + two land-side walls (ribs) = w + 2*H.
        """
        return self.width(x) + 2.0 * self.height(x)

    def fin_efficiency(self, x: float, h_c: float, k_wall: float) -> float:
        """
        Straight-fin efficiency for the rib/land acting as a fin.

        Treats each rib as a straight rectangular fin with an adiabatic tip
        (half-rib by symmetry: only one channel side, coolant on both faces).

        Source: Incropera et al. (2011), Table 3.4, Case A, Eq. (3.80).

            m = sqrt(2 * h_c / (k_wall * t_land))
            eta_fin = tanh(m * H) / (m * H)

        Parameters
        ----------
        x : float
            Axial station [m].
        h_c : float
            Coolant-side heat transfer coefficient [W/(m2.K)].
        k_wall : float
            Wall thermal conductivity [W/(m.K)].

        Returns
        -------
        float
            Fin efficiency eta_fin in [0, 1].
        """
        t_land = self.land_thickness(x)
        H = self.height(x)
        if t_land <= 0.0 or k_wall <= 0.0:
            return 1.0
        m = math.sqrt(2.0 * h_c / (k_wall * t_land))
        mH = m * H
        if mH < 1.0e-8:
            return 1.0
        return math.tanh(mH) / mH

    def overall_surface_efficiency(self, x: float, eta_fin: float) -> float:
        """
        Overall surface efficiency eta_0 accounting for fin (rib) contribution.

        Source: Incropera et al. (2011), Eq. (3.86):
            eta_0 = 1 - (A_fin / A_total) * (1 - eta_fin)

        where A_fin = 2*H (two rib surfaces) and A_total = w + 2*H (3-sided).

        Parameters
        ----------
        x : float
            Axial station [m].
        eta_fin : float
            Fin efficiency from ``fin_efficiency``.

        Returns
        -------
        float
            Overall surface efficiency eta_0 in (0, 1].
        """
        w = self.width(x)
        H = self.height(x)
        A_fin = 2.0 * H
        A_total = w + 2.0 * H
        return 1.0 - (A_fin / A_total) * (1.0 - eta_fin)


# ---------------------------------------------------------------------------
# Wall material
# ---------------------------------------------------------------------------


@dataclass
class WallMaterial:
    """
    Isotropic wall material with temperature-dependent thermal conductivity.

    Parameters
    ----------
    name : str
        Material name for reporting.
    k_func : Callable[[float], float]
        Thermal conductivity k(T) [W/(m.K)] as a function of temperature T [K].
    T_max_continuous : float
        Maximum continuous service temperature [K].  A warning is issued
        if T_wall_gas exceeds this limit.
    """

    name: str
    k_func: Callable[[float], float]
    T_max_continuous: float

    def k(self, T: float) -> float:
        """Thermal conductivity at temperature T [K] -> [W/(m.K)]."""
        return self.k_func(T)


def _make_interp(T_points: list[float], k_points: list[float]) -> Callable[[float], float]:
    """Build a clamped linear interpolant from table data."""
    T_arr = np.array(T_points, dtype=float)
    k_arr = np.array(k_points, dtype=float)
    f = interp1d(T_arr, k_arr, kind="linear", bounds_error=False, fill_value=(k_arr[0], k_arr[-1]))
    return lambda T: float(f(T))


# GRCop-42 (Cu-4wt%Cr-2wt%Nb) thermal conductivity
# Source: Ellis et al. (2013), NASA/TM-2013-217848, Table 3 (approximate).
# NOTE: Verify against the published paper before flight use.
_GRCOP42_T = [294.0, 473.0, 673.0, 873.0, 1073.0]  # K
_GRCOP42_K = [315.0, 295.0, 278.0, 263.0, 250.0]  # W/(m.K)

# Maximum continuous service temperature: ~810 K (creep/oxidation limit for
# copper alloy liners; see NASA SP-8087 Ch. 4 and Ellis 2005 TM-2005-213566).
GRCOP42 = WallMaterial(
    name="GRCop-42",
    k_func=_make_interp(_GRCOP42_T, _GRCOP42_K),
    T_max_continuous=810.0,
)

# Inconel 718 thermal conductivity
# Source: Special Metals Corporation (2007). "INCONEL Alloy 718."
# Publication No. SMC-045.  Table of Properties.
_IN718_T = [293.0, 477.0, 589.0, 700.0, 811.0, 922.0]  # K
_IN718_K = [11.4, 14.7, 16.3, 18.0, 19.4, 20.9]  # W/(m.K)

# Incipient oxidation / creep limit ~1200 K in oxidising environments
# (conservative; consult vendor data for specific heat treatment).
INCONEL718 = WallMaterial(
    name="Inconel 718",
    k_func=_make_interp(_IN718_T, _IN718_K),
    T_max_continuous=1200.0,
)


# ---------------------------------------------------------------------------
# Coolant inlet conditions
# ---------------------------------------------------------------------------


@dataclass
class CoolantInlet:
    """
    Coolant inlet state for the regenerative cooling circuit.

    Parameters
    ----------
    T : float
        Inlet temperature [K].
    P : float
        Inlet pressure [Pa].
    m_dot : float
        Total coolant mass flow rate (all channels combined) [kg/s].
    fluid : str
        CoolProp fluid name (e.g. 'Methane', 'Ethanol').
    """

    T: float
    P: float
    m_dot: float
    fluid: str = "Methane"
