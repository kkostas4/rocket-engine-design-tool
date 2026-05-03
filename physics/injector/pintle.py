"""
pintle.py
=========

Pintle injector sizing for a single-element LCH4/LOX rocket engine.

A pintle injector consists of a central pintle post through which one
propellant flows (here LOX, through an annular gap) and a co-axial sleeve
through which the second propellant (LCH4) enters via discrete orifices
arranged around the pintle periphery.

The Total Momentum Ratio (TMR) quantifies the relative momenta of the two
streams and is the primary mixing-quality parameter for pintle injectors.
A TMR of 1.0 indicates balanced momentum — a common starting point.

Source
------
    Dressler, G.A. and Bauer, J.M. (2000). "TRW Pintle Engine Heritage and
    Performance Characteristics." AIAA 2000-3871. Section 3 — injector
    geometry and TMR definition.

    Sutton, G.P. and Biblarz, O. (2017). *Rocket Propulsion Elements*,
    9th ed., Ch. 8 — orifice and annulus sizing.

    NASA SP-8089 (1976). *Liquid Rocket Engine Injectors.* Section 4 —
    element sizing methodology.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from physics.injector.orifice import (
    injection_velocity,
    orifice_area_required,
)


@dataclass
class PintleInjector:
    """
    Single-element pintle injector geometry and performance.

    The pintle OD and number of fuel orifices are design parameters chosen
    by the engineer. All derived geometry flows from the mass flow / pressure
    drop constraints via the incompressible orifice equation.

    Source: Dressler & Bauer (2000), AIAA 2000-3871.

    Parameters
    ----------
    m_dot_ox : float
        Oxidiser (LOX) mass flow rate [kg/s].
    m_dot_fuel : float
        Fuel (LCH4) mass flow rate [kg/s].
    rho_ox : float
        Oxidiser density at injection conditions [kg/m³].
    rho_fuel : float
        Fuel density at injection conditions [kg/m³].
    dP_inj : float
        Injector pressure drop [Pa]. Applied to both streams.
    Cd_ox : float
        Discharge coefficient for LOX annulus [-].
    Cd_fuel : float
        Discharge coefficient for fuel orifices [-].
    pintle_OD : float
        Outer diameter of the pintle post [m]. Design parameter.
    n_fuel_orifices : int
        Number of discrete fuel injection orifices [-]. Design parameter.
    """

    m_dot_ox: float       # kg/s
    m_dot_fuel: float     # kg/s
    rho_ox: float         # kg/m³
    rho_fuel: float       # kg/m³
    dP_inj: float         # Pa
    Cd_ox: float = 0.6
    Cd_fuel: float = 0.6
    pintle_OD: float = 0.030    # m
    n_fuel_orifices: int = 20

    # ------------------------------------------------------------------
    # Fuel orifice geometry
    # ------------------------------------------------------------------

    @property
    def fuel_orifice_area_total(self) -> float:
        """Total fuel orifice area across all n_fuel_orifices [m²].

        Source: Sutton & Biblarz (2017), Ch. 8, Eq. (8-1) rearranged.
        """
        return orifice_area_required(
            m_dot=self.m_dot_fuel,
            Cd=self.Cd_fuel,
            rho=self.rho_fuel,
            dP=self.dP_inj,
        )

    @property
    def fuel_orifice_area_single(self) -> float:
        """Area of a single fuel orifice [m²].

        Source: Sutton & Biblarz (2017), Ch. 8 — uniform orifice assumption.
        """
        return self.fuel_orifice_area_total / self.n_fuel_orifices

    @property
    def fuel_orifice_diameter(self) -> float:
        """Diameter of a single circular fuel orifice [m].

        Derived from single orifice area assuming circular cross-section:
        d = sqrt(4 * A / pi).

        Source: Sutton & Biblarz (2017), Ch. 8.
        """
        return math.sqrt(4.0 * self.fuel_orifice_area_single / math.pi)

    # ------------------------------------------------------------------
    # LOX annulus geometry
    # ------------------------------------------------------------------

    @property
    def lox_annulus_area(self) -> float:
        """Total LOX annulus flow area [m²].

        Source: Sutton & Biblarz (2017), Ch. 8, Eq. (8-1) rearranged.
        """
        return orifice_area_required(
            m_dot=self.m_dot_ox,
            Cd=self.Cd_ox,
            rho=self.rho_ox,
            dP=self.dP_inj,
        )

    @property
    def lox_annulus_OD(self) -> float:
        """Outer diameter of the LOX annulus [m].

        The annulus area is A = pi/4 * (OD^2 - pintle_OD^2), solved for OD:

        .. math::

            OD = \\sqrt{pintle\\_OD^2 + \\frac{4 A}{\\pi}}

        Source: Dressler & Bauer (2000), AIAA 2000-3871, annulus geometry.
        """
        return math.sqrt(
            self.pintle_OD**2 + 4.0 * self.lox_annulus_area / math.pi
        )

    @property
    def lox_annulus_gap(self) -> float:
        """Radial gap of the LOX annulus [m].

        gap = (OD - pintle_OD) / 2.

        Source: Dressler & Bauer (2000), AIAA 2000-3871.
        """
        return (self.lox_annulus_OD - self.pintle_OD) / 2.0

    # ------------------------------------------------------------------
    # Injection velocities
    # ------------------------------------------------------------------

    @property
    def V_ox(self) -> float:
        """Mean LOX injection velocity [m/s].

        Computed from continuity: V = m_dot / (rho * A).

        Source: Sutton & Biblarz (2017), Ch. 8.
        """
        return injection_velocity(
            m_dot=self.m_dot_ox,
            A=self.lox_annulus_area,
            rho=self.rho_ox,
        )

    @property
    def V_fuel(self) -> float:
        """Mean fuel injection velocity [m/s].

        Computed from continuity: V = m_dot / (rho * A).

        Source: Sutton & Biblarz (2017), Ch. 8.
        """
        return injection_velocity(
            m_dot=self.m_dot_fuel,
            A=self.fuel_orifice_area_total,
            rho=self.rho_fuel,
        )

    # ------------------------------------------------------------------
    # Mixing quality
    # ------------------------------------------------------------------

    @property
    def TMR(self) -> float:
        """Total Momentum Ratio: rho_ox * V_ox^2 / (rho_fuel * V_fuel^2) [-].

        TMR characterises the relative radial momentum of the two streams.
        TMR = 1.0 indicates matched momentum — a common design target for
        pintle injectors. With equal Cd and equal dP on both streams, TMR = 1.0
        exactly (algebraic identity).

        Source: Dressler, G.A. and Bauer, J.M. (2000). "TRW Pintle Engine
                Heritage and Performance Characteristics." AIAA 2000-3871.
                Section 3 — TMR definition.
        """
        return (self.rho_ox * self.V_ox**2) / (self.rho_fuel * self.V_fuel**2)

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a formatted multi-line summary in engineering units.

        Returns a string (does not print). Caller is responsible for output.
        """
        d_mm = self.fuel_orifice_diameter * 1e3
        lox_od_mm = self.lox_annulus_OD * 1e3
        lox_gap_mm = self.lox_annulus_gap * 1e3
        pintle_od_mm = self.pintle_OD * 1e3

        lines = [
            "=" * 55,
            "  Pintle Injector Summary",
            "=" * 55,
            "  Inputs",
            f"    m_dot_fuel          : {self.m_dot_fuel:.4f} kg/s",
            f"    m_dot_ox            : {self.m_dot_ox:.4f} kg/s",
            f"    rho_fuel            : {self.rho_fuel:.1f} kg/m³",
            f"    rho_ox              : {self.rho_ox:.1f} kg/m³",
            f"    dP_inj              : {self.dP_inj / 1e6:.4f} MPa  "
            f"({self.dP_inj / 6894.7572:.1f} psi)",
            f"    Cd_fuel / Cd_ox     : {self.Cd_fuel} / {self.Cd_ox}",
            f"    pintle OD           : {pintle_od_mm:.2f} mm",
            f"    n_fuel_orifices     : {self.n_fuel_orifices}",
            "",
            "  Fuel (LCH4) Orifices",
            f"    Total area          : {self.fuel_orifice_area_total:.4e} m²",
            f"    Single orifice area : {self.fuel_orifice_area_single:.4e} m²",
            f"    Orifice diameter    : {d_mm:.3f} mm",
            f"    Injection velocity  : {self.V_fuel:.2f} m/s",
            "",
            "  Oxidiser (LOX) Annulus",
            f"    Annulus area        : {self.lox_annulus_area:.4e} m²",
            f"    Annulus OD          : {lox_od_mm:.3f} mm",
            f"    Radial gap          : {lox_gap_mm:.3f} mm",
            f"    Injection velocity  : {self.V_ox:.2f} m/s",
            "",
            "  Mixing",
            f"    TMR                 : {self.TMR:.4f}",
            "=" * 55,
        ]
        return "\n".join(lines)
