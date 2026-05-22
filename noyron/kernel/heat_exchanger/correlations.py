"""
correlations.py
===============

Standalone heat transfer and friction correlations plus ColdSideModel
wrappers.  Each correlation is a plain function (no rocket-specific imports)
and also available as a ColdSideModel implementation.

Sources
-------
Dittus-Boelter:
    Dittus, F.W. and Boelter, L.M.K. (1930). Univ. of California Publ. Eng.
    Vol. 2, pp. 443-461.  Restated in McAdams, W.H. (1954). "Heat
    Transmission," 3rd ed., McGraw-Hill. Ch. 8.
    Nu = 0.023 * Re^0.8 * Pr^0.4 (heating the fluid).
    Validity: Re > 10,000; 0.7 < Pr < 160; L/D > 10.

Sieder-Tate:
    Sieder, E.N. and Tate, G.E. (1936). "Heat Transfer and Pressure Drop
    of Liquids in Tubes." Ind. Eng. Chem., Vol. 28, No. 12, pp. 1429-1435.
    Nu = 0.027 * Re^0.8 * Pr^(1/3) * (mu/mu_w)^0.14.
    Validity: 0.7 < Pr < 16,700; Re > 10,000.

Gnielinski:
    Gnielinski, V. (1976). "New Equations for Heat and Mass Transfer in
    Turbulent Pipe and Channel Flow." Int. Chem. Eng., Vol. 16, pp. 359-368.
    Nu = (f/8)(Re - 1000)Pr / [1 + 12.7*(f/8)^0.5*(Pr^(2/3) - 1)].
    Validity: 0.5 < Pr < 2000; 3000 < Re < 5e6.

Haaland friction factor:
    Haaland, S.E. (1983). "Simple and Explicit Formulas for the Friction
    Factor in Turbulent Pipe Flow." J. Fluids Eng., Vol. 105, pp. 89-90.
    Eq. (3): 1/sqrt(f) = -1.8*log10[(eps_r/D/3.7)^1.11 + 6.9/Re].
"""

from __future__ import annotations

import math
import warnings

from physics.heat_exchanger.base import (
    ColdSideModel,
    ColdSideResult,
    FluidProperties,
    HeatExchangerGeometry,
)


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------


def dittus_boelter(Re: float, Pr: float, D_h: float, k_c: float) -> float:
    """
    Coolant-side h from Dittus-Boelter (heating the fluid).

    Nu = 0.023 * Re^0.8 * Pr^0.4

    Source: McAdams (1954), restating Dittus & Boelter (1930).
    Equation: Nu = 0.023 Re^0.8 Pr^0.4 (heating).
    Validity: Re > 10,000; 0.7 < Pr < 160; L/D > 10.

    Parameters
    ----------
    Re : float      Reynolds number [-].
    Pr : float      Prandtl number [-].
    D_h : float     Hydraulic diameter [m].
    k_c : float     Fluid thermal conductivity [W/(m·K)].

    Returns
    -------
    float
        Convective heat transfer coefficient h [W/(m²·K)].
    """
    if Re < 1.0e4:
        warnings.warn(
            f"Re = {Re:.0f} < 10,000: Dittus-Boelter is outside validity range.",
            RuntimeWarning,
            stacklevel=2,
        )
    if not (0.7 <= Pr <= 160.0):
        warnings.warn(
            f"Pr = {Pr:.3f} outside Dittus-Boelter validity range [0.7, 160].",
            RuntimeWarning,
            stacklevel=2,
        )
    Nu = 0.023 * Re**0.8 * Pr**0.4
    return Nu * k_c / D_h


def sieder_tate(
    Re: float,
    Pr: float,
    D_h: float,
    k_c: float,
    mu: float,
    mu_w: float,
) -> float:
    """
    Coolant-side h from Sieder-Tate with viscosity correction.

    Nu = 0.027 * Re^0.8 * Pr^(1/3) * (mu/mu_w)^0.14

    Source: Sieder & Tate (1936). Ind. Eng. Chem. Vol. 28, pp. 1429-1435.
    Equation: Nu = 0.027 Re^0.8 Pr^(1/3) (mu/mu_w)^0.14.
    Validity: 0.7 < Pr < 16,700; Re > 10,000.

    Parameters
    ----------
    Re : float      Reynolds number [-].
    Pr : float      Prandtl number [-].
    D_h : float     Hydraulic diameter [m].
    k_c : float     Fluid thermal conductivity [W/(m·K)].
    mu : float      Bulk dynamic viscosity [Pa·s].
    mu_w : float    Wall dynamic viscosity [Pa·s].

    Returns
    -------
    float
        Convective heat transfer coefficient h [W/(m²·K)].
    """
    Nu = 0.027 * Re**0.8 * Pr ** (1.0 / 3.0) * (mu / mu_w) ** 0.14
    return Nu * k_c / D_h


def gnielinski(Re: float, Pr: float, D_h: float, k_c: float) -> float:
    """
    Coolant-side h from Gnielinski (improved accuracy vs. Dittus-Boelter).

    Uses Haaland to compute the friction factor f, then:
    Nu = (f/8)(Re-1000)*Pr / [1 + 12.7*(f/8)^0.5*(Pr^(2/3)-1)]

    Source: Gnielinski, V. (1976). Int. Chem. Eng. Vol. 16, pp. 359-368.
    Validity: 0.5 < Pr < 2000; 3000 < Re < 5e6.

    Parameters
    ----------
    Re : float      Reynolds number [-].
    Pr : float      Prandtl number [-].
    D_h : float     Hydraulic diameter [m].
    k_c : float     Fluid thermal conductivity [W/(m·K)].

    Returns
    -------
    float
        Convective heat transfer coefficient h [W/(m²·K)].
    """
    f = haaland_f(Re)
    Nu = (f / 8.0) * (Re - 1000.0) * Pr / (
        1.0 + 12.7 * math.sqrt(f / 8.0) * (Pr ** (2.0 / 3.0) - 1.0)
    )
    return Nu * k_c / D_h


def haaland_f(Re: float, roughness: float = 0.0, D_h: float = 1.0) -> float:
    """
    Darcy friction factor from Haaland (1983) Eq. (3).

    For smooth channels (roughness = 0):
        1/sqrt(f) = -1.8 * log10(6.9 / Re)

    Source: Haaland, S.E. (1983). J. Fluids Eng., Vol. 105, pp. 89-90.
    Validity: Re > ~3,000 (turbulent flow).

    Parameters
    ----------
    Re : float          Reynolds number [-].
    roughness : float   Absolute roughness [m].  0 for smooth channels.
    D_h : float         Hydraulic diameter [m].  Used for relative roughness.

    Returns
    -------
    float
        Darcy friction factor f [-].
    """
    if roughness <= 0.0:
        inv_sqrt_f = -1.8 * math.log10(6.9 / max(Re, 1.0))
    else:
        eps_r = roughness / D_h
        inv_sqrt_f = -1.8 * math.log10((eps_r / 3.7) ** 1.11 + 6.9 / max(Re, 1.0))
    if inv_sqrt_f == 0.0:
        return 0.02
    return 1.0 / inv_sqrt_f**2


# ---------------------------------------------------------------------------
# ColdSideModel implementations
# ---------------------------------------------------------------------------


class DittusBoelterModel(ColdSideModel):
    """
    Cold-side model using Dittus-Boelter (Nu = 0.023 Re^0.8 Pr^0.4).

    Source: McAdams (1954), restating Dittus & Boelter (1930).
    Validity: Re > 10,000; 0.7 < Pr < 160.
    """

    def evaluate(
        self,
        x: float,
        properties: FluidProperties,
        m_dot_per_channel: float,
        geometry: HeatExchangerGeometry,
    ) -> ColdSideResult:
        """
        Evaluate Dittus-Boelter h at the given station and flow state.

        Parameters
        ----------
        x : float
            Axial coordinate [m].
        properties : FluidProperties
            Coolant properties at the current bulk temperature and pressure.
        m_dot_per_channel : float
            Mass flow rate per channel [kg/s].
        geometry : HeatExchangerGeometry
            Local geometry providing cross_section_cold and hydraulic_diameter_cold.

        Returns
        -------
        ColdSideResult
            h_convective [W/(m²·K)], Re [-], V [m/s].
        """
        A_c = geometry.cross_section_cold(x)
        D_h = geometry.hydraulic_diameter_cold(x)
        V = m_dot_per_channel / (properties.rho * A_c)
        Re = properties.rho * V * D_h / properties.mu
        h = dittus_boelter(Re, properties.Pr, D_h, properties.k)
        return ColdSideResult(h_convective=h, Re=Re, V=V)
