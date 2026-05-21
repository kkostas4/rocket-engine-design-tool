"""
pintle.py
=========

Pintle injector sizing for a single-element LCH4/LOX rocket engine.

A pintle injector consists of a central pintle post through which one
propellant flows (here LOX, through an annular gap) and a co-axial sleeve
through which the second propellant (LCH4) enters via discrete orifices
arranged around the pintle periphery.

The Total Momentum Ratio (TMR) is the primary mixing-quality parameter.
It is defined here as the ratio of fuel-stream radial momentum to
oxidiser-stream radial momentum:

    TMR = (m_dot_fuel · V_fuel) / (m_dot_ox · V_ox)

Using the incompressible orifice relation (V = Cd · sqrt(2·dP/ρ)), this
reduces to:

    TMR = (1/MR) · (Cd_f / Cd_ox) · sqrt( (ρ_ox/ρ_f) · (dP_f/dP_ox) )

where MR = m_dot_ox / m_dot_fuel.

This formula does NOT simplify to 1.0 in general.  It collapses to 1.0 only
if MR = 1, Cd_f = Cd_ox, ρ_f = ρ_ox, and dP_f = dP_ox simultaneously — not
a physically realistic case for LCH4/LOX.

Historical note
---------------
An earlier version of this module used the definition
    TMR_old = ρ_ox · V_ox² / (ρ_f · V_f²)
which is a *dynamic-pressure* ratio, not a momentum ratio.  Substituting
V = Cd · sqrt(2·dP/ρ) into that formula gives
    TMR_old = (Cd_ox² · dP_ox) / (Cd_f² · dP_f)
which is entirely independent of densities and mass flows.  With equal Cd
and equal dP, TMR_old = 1.000 exactly — regardless of MR, ρ_ox, or ρ_f.
This produced the misleading "TMR = 1.000" result from the uniform-ΔP era.
The correct momentum-ratio formula above has been in use since the decoupled-
ΔP refactor (commit after 04a0cb6).

Sources
-------
    Dressler, G.A. and Bauer, J.M. (2000). "TRW Pintle Engine Heritage and
    Performance Characteristics." AIAA 2000-3871. Section 3 — TMR definition
    and recommended range 0.5 ≤ TMR ≤ 2.0.

    Sutton, G.P. and Biblarz, O. (2017). *Rocket Propulsion Elements*,
    9th ed., Ch. 8 — orifice and annulus sizing.

    NASA SP-8089 (1976). *Liquid Rocket Engine Injectors.* Section 4 —
    element sizing methodology.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from physics.injector.orifice import (
    injection_velocity,
    orifice_area_required,
)
from noyron.kernel.component import (
    Component,
    Renderable,
    Sizable,
    VerificationReport,
)


@dataclass
class PintleInjector(Component, Renderable, Sizable):
    """
    Single-element pintle injector geometry and performance.

    The pintle OD and number of fuel orifices are design parameters chosen
    by the engineer.  All derived geometry flows from the mass flow and
    pressure-drop constraints via the incompressible orifice equation.

    Oxidiser and fuel may now carry *different* injection pressure drops
    (dP_ox, dP_fuel), making TMR a genuine design variable.  For equal drops
    use the :meth:`from_uniform_dP` classmethod.

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
    dP_ox : float, optional
        Oxidiser injector pressure drop [Pa].  Required unless ``dP_inj``
        is supplied for backward compatibility.
    dP_fuel : float, optional
        Fuel injector pressure drop [Pa].  Required unless ``dP_inj``
        is supplied for backward compatibility.
    dP_inj : float, optional
        *Deprecated.*  Uniform pressure drop [Pa] applied to both streams.
        Provided for backward compatibility; prefer ``from_uniform_dP``.
    Cd_ox : float
        Discharge coefficient for LOX annulus (user input; Lichtarowicz does
        not apply to annular passages).  Default 0.6.
    Cd_fuel : float, optional
        Discharge coefficient for fuel orifices.  If ``None``, the
        Lichtarowicz–Duggins–Markland (1965) correlation is used and
        ``mu_fuel`` must be supplied.  Default 0.6 (constant).
    LD_fuel : float
        Length-to-diameter ratio of fuel orifices for Lichtarowicz.
        Default 4.0.
    mu_fuel : float, optional
        Dynamic viscosity of the fuel at injection conditions [Pa·s].
        Required when ``Cd_fuel`` is ``None``.
    pintle_OD : float
        Outer diameter of the pintle post [m].  Design parameter.
    n_fuel_orifices : int
        Number of discrete fuel injection orifices.  Design parameter.
    """

    m_dot_ox: float
    m_dot_fuel: float
    rho_ox: float
    rho_fuel: float
    dP_ox: Optional[float] = None
    dP_fuel: Optional[float] = None
    dP_inj: Optional[float] = None   # deprecated; use from_uniform_dP
    Cd_ox: float = 0.6
    Cd_fuel: Optional[float] = 0.6
    LD_fuel: float = 4.0
    mu_fuel: Optional[float] = None
    pintle_OD: float = 0.030
    n_fuel_orifices: int = 20

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        # Backward-compat: single dP_inj → both streams.
        if self.dP_inj is not None and self.dP_ox is None and self.dP_fuel is None:
            self.dP_ox = self.dP_inj
            self.dP_fuel = self.dP_inj
        if self.dP_ox is None or self.dP_fuel is None:
            raise ValueError(
                "PintleInjector requires dP_ox and dP_fuel.  "
                "For a uniform drop use PintleInjector.from_uniform_dP(dP_inj=…)."
            )
        if self.Cd_fuel is None and self.mu_fuel is None:
            raise ValueError(
                "mu_fuel is required when Cd_fuel is None "
                "(Lichtarowicz iteration needs viscosity)."
            )

    @classmethod
    def from_uniform_dP(
        cls,
        *,
        dP_inj: float,
        m_dot_ox: float,
        m_dot_fuel: float,
        rho_ox: float,
        rho_fuel: float,
        Cd_ox: float = 0.6,
        Cd_fuel: Optional[float] = 0.6,
        LD_fuel: float = 4.0,
        mu_fuel: Optional[float] = None,
        pintle_OD: float = 0.030,
        n_fuel_orifices: int = 20,
    ) -> "PintleInjector":
        """
        Construct with equal pressure drop on both propellant streams.

        This is the backward-compatible entry point that replaces the old
        single ``dP_inj`` field.  Passing ``dP_inj=X`` is exactly equivalent
        to ``PintleInjector(dP_ox=X, dP_fuel=X, …)``.

        Parameters
        ----------
        dP_inj : float
            Uniform injector pressure drop [Pa] applied to both streams.
        **kwargs
            All other ``PintleInjector`` fields.
        """
        return cls(
            m_dot_ox=m_dot_ox,
            m_dot_fuel=m_dot_fuel,
            rho_ox=rho_ox,
            rho_fuel=rho_fuel,
            dP_ox=dP_inj,
            dP_fuel=dP_inj,
            Cd_ox=Cd_ox,
            Cd_fuel=Cd_fuel,
            LD_fuel=LD_fuel,
            mu_fuel=mu_fuel,
            pintle_OD=pintle_OD,
            n_fuel_orifices=n_fuel_orifices,
        )

    # ------------------------------------------------------------------
    # Effective fuel Cd (constant or Lichtarowicz)
    # ------------------------------------------------------------------

    @property
    def cd_fuel_effective(self) -> float:
        """
        Effective fuel-orifice discharge coefficient [-].

        If ``Cd_fuel`` was supplied explicitly, returns it unchanged.
        Otherwise runs the Lichtarowicz–Duggins–Markland (1965) fixed-point
        iteration to find the Cd consistent with the Reynolds number of the
        sized orifice.

        Source: Lichtarowicz, A., Duggins, R.K. and Markland, E. (1965).
                "Discharge coefficients for incompressible non-cavitating
                flow through long orifices."
                Proc. IMechE, J. Mech. Eng. Sci., Vol. 7, pp. 210–219.
        """
        if self.Cd_fuel is not None:
            return self.Cd_fuel

        # Deferred import to keep the module dependency graph clean.
        from physics.injector.discharge import cd_iterative

        assert self.mu_fuel is not None  # validated in __post_init__
        m_dot_per = self.m_dot_fuel / self.n_fuel_orifices
        _cd, _re = cd_iterative(
            m_dot=m_dot_per,
            rho=self.rho_fuel,
            mu=self.mu_fuel,
            dP=self.dP_fuel,
            LD=self.LD_fuel,
        )
        return _cd

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
            Cd=self.cd_fuel_effective,
            rho=self.rho_fuel,
            dP=self.dP_fuel,
        )

    @property
    def fuel_orifice_area_single(self) -> float:
        """Area of one fuel orifice [m²].

        Source: Sutton & Biblarz (2017), Ch. 8.
        """
        return self.fuel_orifice_area_total / self.n_fuel_orifices

    @property
    def fuel_orifice_diameter(self) -> float:
        """Diameter of a single circular fuel orifice [m].

        d = sqrt(4 · A_single / π).

        Source: Sutton & Biblarz (2017), Ch. 8.
        """
        return math.sqrt(4.0 * self.fuel_orifice_area_single / math.pi)

    # ------------------------------------------------------------------
    # LOX annulus geometry
    # ------------------------------------------------------------------

    @property
    def lox_annulus_area(self) -> float:
        """LOX annulus flow area [m²].

        Source: Sutton & Biblarz (2017), Ch. 8, Eq. (8-1) rearranged.
        Uses dP_ox (may differ from dP_fuel).
        """
        return orifice_area_required(
            m_dot=self.m_dot_ox,
            Cd=self.Cd_ox,
            rho=self.rho_ox,
            dP=self.dP_ox,
        )

    @property
    def lox_annulus_OD(self) -> float:
        """Outer diameter of the LOX annulus [m].

        Solved from A = π/4·(OD² − pintle_OD²):

        .. math::

            OD = \\sqrt{pintle\\_OD^2 + \\frac{4 A}{\\pi}}

        Source: Dressler & Bauer (2000), AIAA 2000-3871.
        """
        return math.sqrt(
            self.pintle_OD**2 + 4.0 * self.lox_annulus_area / math.pi
        )

    @property
    def lox_annulus_gap(self) -> float:
        """Radial gap of the LOX annulus (= (OD − pintle_OD) / 2) [m].

        Source: Dressler & Bauer (2000), AIAA 2000-3871.
        """
        return (self.lox_annulus_OD - self.pintle_OD) / 2.0

    # ------------------------------------------------------------------
    # Injection velocities
    # ------------------------------------------------------------------

    @property
    def V_ox(self) -> float:
        """Mean LOX injection velocity from continuity [m/s].

        V = m_dot / (ρ · A).

        Source: Sutton & Biblarz (2017), Ch. 8.
        """
        return injection_velocity(
            m_dot=self.m_dot_ox,
            A=self.lox_annulus_area,
            rho=self.rho_ox,
        )

    @property
    def V_fuel(self) -> float:
        """Mean fuel injection velocity from continuity [m/s].

        V = m_dot / (ρ · A).

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
        """
        Total Momentum Ratio: (ṁ_fuel · V_fuel) / (ṁ_ox · V_ox) [-].

        Defined as the ratio of fuel radial momentum to oxidiser radial
        momentum per Dressler & Bauer (2000).  Using V = Cd · sqrt(2·dP/ρ)
        this expands to:

        .. math::

            TMR = \\frac{1}{MR}
                  \\cdot \\frac{C_{d,f}}{C_{d,ox}}
                  \\cdot \\sqrt{\\frac{\\rho_{ox}}{\\rho_f}
                               \\cdot \\frac{\\Delta P_f}{\\Delta P_{ox}}}

        where :math:`MR = \\dot{m}_{ox} / \\dot{m}_f`.

        Design guidance (Dressler & Bauer): 0.5 ≤ TMR ≤ 2.0.

        **Why the old result was TMR = 1.000**

        The earlier implementation used ρ_ox·V_ox² / (ρ_f·V_f²), a dynamic-
        pressure ratio.  Substituting V = Cd·sqrt(2·dP/ρ) gives
        Cd_ox²·dP_ox / (Cd_f²·dP_f), which is completely independent of
        densities and mass flows.  With equal Cd and equal dP the formula
        trivially returns 1.000 for *any* propellant combination.  The correct
        momentum-ratio formula carries the density and MR dependence that make
        TMR a meaningful design variable.

        Reference engine (equal Cd, equal dP, MR = 3.6, ρ_ox/ρ_f = 2.89):
            TMR = (1/3.6) · 1 · sqrt(2.89) ≈ 0.47

        Source: Dressler, G.A. and Bauer, J.M. (2000). "TRW Pintle Engine
                Heritage and Performance Characteristics." AIAA 2000-3871,
                Section 3.
        """
        return (self.m_dot_fuel * self.V_fuel) / (self.m_dot_ox * self.V_ox)

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a formatted multi-line summary in engineering units."""
        psi = 6_894.7572931783
        d_mm = self.fuel_orifice_diameter * 1e3
        lox_od_mm = self.lox_annulus_OD * 1e3
        lox_gap_mm = self.lox_annulus_gap * 1e3
        pintle_od_mm = self.pintle_OD * 1e3

        lines = [
            "=" * 60,
            "  Pintle Injector Summary",  # noqa: E501
            "=" * 60,
            "  Inputs",
            f"    m_dot_fuel          : {self.m_dot_fuel:.4f} kg/s",
            f"    m_dot_ox            : {self.m_dot_ox:.4f} kg/s",
            f"    MR (ox/fuel)        : {self.m_dot_ox/self.m_dot_fuel:.3f}",
            f"    rho_fuel            : {self.rho_fuel:.1f} kg/m³",
            f"    rho_ox              : {self.rho_ox:.1f} kg/m³",
            f"    dP_fuel             : {self.dP_fuel / 1e6:.4f} MPa"
            f"  ({self.dP_fuel / psi:.1f} psi)",
            f"    dP_ox               : {self.dP_ox / 1e6:.4f} MPa"
            f"  ({self.dP_ox / psi:.1f} psi)",
            f"    dP_fuel / dP_ox     : {self.dP_fuel / self.dP_ox:.3f}",
            f"    Cd_fuel (eff)       : {self.cd_fuel_effective:.4f}",
            f"    Cd_ox               : {self.Cd_ox:.4f}",
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
            "=" * 60,
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Component interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "PintleInjector"

    def verify(self) -> VerificationReport:
        """Self-check returning a VerificationReport.

        Checks: injector pressure drops are positive, mass flow rates
        are positive, n_fuel_orifices >= 4 (NASA SP-8089 §4 minimum for
        stable atomisation), and TMR is within recommended range [0.5, 2.0].

        Source: Dressler & Bauer (2000), AIAA 2000-3871, §3 — TMR range.
                NASA SP-8089 (1976), §4.2 — minimum orifice count guidance.
        """
        report = VerificationReport.ok(self.name)
        if (self.dP_ox is not None and self.dP_ox <= 0.0) or (
            self.dP_fuel is not None and self.dP_fuel <= 0.0
        ):
            report = report.add(
                "error",
                f"Injector pressure drops must be positive "
                f"(dP_ox={self.dP_ox}, dP_fuel={self.dP_fuel})",
                metric="dP_ox",
                value=float(self.dP_ox or 0.0),
                threshold=0.0,
            )
        if self.m_dot_ox <= 0.0 or self.m_dot_fuel <= 0.0:
            report = report.add(
                "error",
                f"Mass flow rates must be positive "
                f"(m_dot_ox={self.m_dot_ox:.4f}, m_dot_fuel={self.m_dot_fuel:.4f})",
            )
        if self.n_fuel_orifices < 4:
            report = report.add(
                "error",
                f"n_fuel_orifices = {self.n_fuel_orifices} < 4 minimum for "
                "stable atomisation (NASA SP-8089 §4.2)",
                metric="n_fuel_orifices",
                value=float(self.n_fuel_orifices),
                threshold=4.0,
            )
        tmr = self.TMR
        if tmr < 0.5:
            report = report.add(
                "warning",
                f"TMR = {tmr:.3f} below recommended lower bound 0.50 "
                "(Dressler & Bauer 2000, AIAA 2000-3871, §3)",
                metric="TMR", value=tmr, threshold=0.5,
            )
        elif tmr > 2.0:
            report = report.add(
                "warning",
                f"TMR = {tmr:.3f} above recommended upper bound 2.00 "
                "(Dressler & Bauer 2000, AIAA 2000-3871, §3)",
                metric="TMR", value=tmr, threshold=2.0,
            )
        return report

    # ------------------------------------------------------------------
    # Renderable interface
    # ------------------------------------------------------------------

    def to_sdf(self, **context: Any):
        """SDF of the injector face plate and pintle post.

        If ``nozzle`` is in context, the SDF is in global coordinates
        (z_face = nozzle.x_chamber_start); otherwise it uses local
        coordinates (z_face = 0.0).

        Parameters
        ----------
        **context : nozzle (RaoNozzle, optional), bilayer (BilayerWall, optional)
        """
        nozzle = context.get("nozzle")
        bilayer = context.get("bilayer")
        from geometry.components.injector import injector_sdf
        sdf, _ = injector_sdf(pintle=self, nozzle=nozzle, bilayer=bilayer)
        return sdf

    def bbox(self, **context: Any) -> tuple[np.ndarray, np.ndarray]:
        """Bounding box of the face plate + pintle post assembly.

        If ``nozzle`` is in context, global coordinates are used;
        otherwise local (z_face = 0.0).
        """
        nozzle = context.get("nozzle")
        face_thickness = 8e-3
        pintle_protrusion = 1.5 * self.pintle_OD
        if nozzle is not None:
            z_face = float(nozzle.x_chamber_start)
            r_ch = float(nozzle.r_chamber)
        else:
            z_face = 0.0
            r_ch = self.lox_annulus_OD  # rough local estimate
        z_tip = z_face + face_thickness + pintle_protrusion
        r_max = max(r_ch, self.lox_annulus_OD / 2.0)
        return (
            np.array([z_face, -r_max, -r_max]),
            np.array([z_tip, r_max, r_max]),
        )

    # ------------------------------------------------------------------
    # Sizable interface
    # ------------------------------------------------------------------

    @classmethod
    def size_from(cls, requirements: Any, **context: Any) -> "PintleInjector":
        """Construct a PintleInjector from EngineRequirements.

        Parameters
        ----------
        requirements : EngineRequirements
        context : m_dot_ox (float), m_dot_fuel (float)
        """
        m_dot_ox = context["m_dot_ox"]
        m_dot_fuel = context["m_dot_fuel"]
        prop = requirements.propellant
        dP_inj = 0.45 * requirements.chamber_pressure
        return cls.from_uniform_dP(
            dP_inj=dP_inj,
            m_dot_ox=m_dot_ox,
            m_dot_fuel=m_dot_fuel,
            rho_ox=prop.density_ox,
            rho_fuel=prop.density_fuel,
            Cd_ox=0.6,
            Cd_fuel=0.6,
            pintle_OD=0.030,
            n_fuel_orifices=20,
        )
