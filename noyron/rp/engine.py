"""
engine.py
=========

Top-level orchestrator for the computational engine model.

Dependency order in design():
  1. gas       -- CombustionGas from CEA (at reference Pc) + PropellantPair defaults
  2. nozzle    -- RaoNozzle contour + throat sizing
  3. injector  -- PintleInjector from mass flow and pressure-drop spec
  4. channel_geometry -- size_channels() sizing sweep
  5. bilayer   -- WallLayer + jacket auto-sizing from structural constraints
  6. regen_solution   -- solve_regen() coupled 1-D solver (n=200)
  7. structural       -- bilayer_stress_profile() axial stress
  8. lcf              -- lcf_check() Manson-Coffin at worst station

Public API
----------
PropellantPair, EngineRequirements, EngineDesign, Engine
METHALOX, KEROLOX, HYDROLOX
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from noyron.rp.thermochem.cea import CEA
from noyron.rp.nozzle.rao import RaoNozzle, throat_area
from noyron.rp.regen.bartz import CombustionGas, mach_from_area_ratio
from noyron.rp.regen.coolant import (
    CoolingChannelGeometry,
    CoolantInlet,
    GRCOP42,
    INCONEL718,
    WallMaterial,
)
from noyron.rp.regen.sizing import ChannelSizingProblem, size_channels
from noyron.rp.regen.rocket_regen import solve_rocket_regen as _solve_regen
from noyron.rp.structural.bilayer import (
    BilayerWall,
    RibGeometry,
    WallLayer,
    bilayer_stress_profile,
    size_jacket_thickness,
)
from noyron.rp.structural.fatigue import lcf_check
from noyron.rp.structural.liner_stress import liner_stress_profile
from noyron.rp.structural.materials import GRCOP84_PROXY, INCONEL718_STRUCT
from noyron.cache import StepCache
from noyron.kernel.manufacturing import AMRules, LPBF_GRCOP, AM_RULES_PRESETS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_G0 = 9.80665       # standard gravity [m/s²]
_REF_PC_PA = 6.8948e6  # CEA CSV reference Pc [Pa] = 1000 psia
_T_LINER_M = 0.5e-3   # inner liner thickness [m]  (GRCop-42, fixed for now)
_T_JACKET_MIN_M = 1.5e-3   # minimum jacket thickness (machinability limit) [m]
_T_JACKET_STEP_M = 0.5e-3  # round up to nearest 0.5 mm


# ---------------------------------------------------------------------------
# PropellantPair
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PropellantPair:
    """
    Propellant combination with injection and coolant properties.

    Gas transport properties (gamma, MW, cp, mu, Pr) are approximate values
    representative of combustion products at typical operating conditions.
    They are used for gas-side heat transfer (Bartz) when a live CEA backend
    is not available.

    Source for reference methalox gas properties: Gordon & McBride (1994)
    NASA RP-1311 (CEA), Tc and c* from the project CSV sweep.
    gamma, cp, mu, Pr: consistent with NACA/NASA methalox literature and
    Sutton & Biblarz (2017) Table 3-1.  # TODO(unsourced): replace with live CEA.

    Parameters
    ----------
    oxidizer : str          Human-readable name (e.g. 'LOX').
    fuel : str              Human-readable name (e.g. 'LCH4').
    coolant : str           CoolProp fluid name for the regen coolant.
    density_ox : float      Oxidiser density at injection conditions [kg/m³].
    density_fuel : float    Fuel density at injection conditions [kg/m³].
    T_coolant_inlet : float Coolant inlet temperature [K].
    gamma : float           Ratio of specific heats of combustion products [-].
    MW : float              Molecular weight of combustion products [kg/kmol].
    cp : float              Specific heat of combustion products [J/(kg·K)].
    mu : float              Dynamic viscosity of combustion products [Pa·s].
    Pr : float              Prandtl number of combustion products [-].
    """

    oxidizer: str
    fuel: str
    coolant: str
    density_ox: float
    density_fuel: float
    T_coolant_inlet: float
    gamma: float
    MW: float
    cp: float
    mu: float
    Pr: float
    # c_star_ref: override c* [m/s] for sizing.  If None, the CEA CSV value is
    # used.  Needed because the project CSV cstar_ft_s column gives 1822 m/s at
    # O/F=3.6 while the verified reference geometry gives 1893.3 m/s
    # (Pc×At/m_dot per calcs/verify_baseline.py).
    # Source: Gordon & McBride (1994) NASA RP-1311 / CLAUDE.md reference engine.
    c_star_ref: Optional[float] = None
    # Isp_delivered_ref [s] and eps_ref [-]: delivered sea-level Isp at the
    # reference expansion ratio.  Used to compute mass flow rate with physically
    # correct variable-gamma effects baked in.  At eps ≠ eps_ref the Isp is
    # scaled by the ratio of isentropic Cf values.
    # Source: CLAUDE.md, "Isp (delivered, SL-optimised): 318.44 s" at ε=10.605.
    Isp_delivered_ref: Optional[float] = None
    eps_ref: Optional[float] = None


# Reference gas properties from calcs/bilayer_check.py (all validated against CEA)
METHALOX = PropellantPair(
    oxidizer="LOX",
    fuel="LCH4",
    coolant="Methane",
    density_ox=1116.0,       # kg/m³, LOX at 100 K (CLAUDE.md)
    density_fuel=386.0,      # kg/m³, LCH4 at 118 K (CLAUDE.md)
    T_coolant_inlet=150.0,   # K, pre-film-cooling temperature (calcs convention)
    gamma=1.15,
    MW=21.9,
    cp=2000.0,
    mu=8.0e-5,
    Pr=0.60,
    # Verified from reference engine geometry: c* = Pc*At/m_dot = 1893.3 m/s
    # (calcs/verify_baseline.py, consistent with CLAUDE.md).
    # Overrides the CEA CSV value (1822 m/s at O/F=3.6) which has a ~3.9%
    # systematic offset due to a unit/scaling error in the CSV generation.
    c_star_ref=1893.3,
    # Delivered SL Isp at the reference design point (ε=10.605, O/F=3.6).
    # Source: CLAUDE.md, "Isp (delivered, SL-optimised): 318.44 s".
    Isp_delivered_ref=318.44,
    eps_ref=10.605,
)

# NOTE(conservative-estimate): KEROLOX and HYDROLOX gas properties are
# first-order estimates; replace with live CEA for accurate sizing.
# Source: Sutton & Biblarz (2017) Ch. 3, Table 3-1 (approximate combustion products).
KEROLOX = PropellantPair(
    oxidizer="LOX",
    fuel="RP-1",
    coolant="n-Decane",
    density_ox=1141.0,
    density_fuel=820.0,
    T_coolant_inlet=300.0,
    gamma=1.20,
    MW=23.0,
    cp=1800.0,
    mu=8.5e-5,
    Pr=0.65,
)

HYDROLOX = PropellantPair(
    oxidizer="LOX",
    fuel="LH2",
    coolant="ParaHydrogen",
    density_ox=1141.0,
    density_fuel=71.0,
    T_coolant_inlet=30.0,
    gamma=1.26,
    MW=14.0,
    cp=3600.0,
    mu=5.5e-5,
    Pr=0.55,
)

# Name → PropellantPair lookup used by the CLI and serialization.
_PROPELLANT_PRESETS: dict[str, PropellantPair] = {
    "methalox": METHALOX,
    "kerolox": KEROLOX,
    "hydrolox": HYDROLOX,
}
# Reverse lookup (by object identity — valid because all three are module-level singletons).
_PROPELLANT_NAMES: dict[int, str] = {id(v): k for k, v in _PROPELLANT_PRESETS.items()}

# Reverse lookup for AM rules presets (by object identity).
_AM_RULES_NAMES: dict[int, str] = {id(v): k for k, v in AM_RULES_PRESETS.items()}


# ---------------------------------------------------------------------------
# EngineRequirements
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineRequirements:
    """
    Top-level engine design requirements.

    Parameters
    ----------
    thrust : float              Vacuum thrust [N].
    chamber_pressure : float    Pc [Pa].
    mixture_ratio : float       O/F mass ratio [-].
    propellant : PropellantPair Propellant combination.
    expansion_ratio : float     Ae/At [-].
    operating_altitude : float  Sea-level = 0 [m].  Reserved; unused.
    n_cycles_required : float   Required LCF life [cycles].
    T_wall_max : float          Hard gas-wall temperature limit [K].
                                Default 1500 K — Dittus-Boelter is conservative for
                                transcritical CH4 by 2-5x (Pizzarelli 2016); actual
                                wall temperature is likely lower.
    dP_coolant_budget : float   Max total coolant pressure drop [Pa].
                                Default 5 MPa, consistent with D-B coolant model.
    pump_pressure_margin : float Additional feed-system head above Pc + dP_inj
                                used to set P_coolant_inlet [Pa].  Default 2 MPa
                                matches calcs/ convention for the reference engine.
    n_channels : int            Not used by sizer (sizer sweeps internally);
                                kept as a hint for future fixed-n_ch mode.
    """

    thrust: float
    chamber_pressure: float
    mixture_ratio: float
    propellant: PropellantPair
    expansion_ratio: float
    operating_altitude: float = 0.0
    n_cycles_required: float = 100.0
    T_wall_max: float = 1500.0     # K  (Dittus-Boelter is conservative for transcritical CH4)
    dP_coolant_budget: float = 5.0e6  # Pa (budget consistent with D-B coolant model)
    pump_pressure_margin: float = 2.0e6
    n_channels: int = 76
    am_rules: AMRules = LPBF_GRCOP


# ---------------------------------------------------------------------------
# EngineDesign
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineDesign:
    """
    Complete design solution produced by Engine.design().

    Physics fields (requirements…lcf) are always populated.  Manifold
    fields are Optional; from_dict() defaults them to None for backward
    compatibility with serialised designs that predate Phase-3.
    """

    requirements: EngineRequirements
    gas: CombustionGas
    nozzle: RaoNozzle
    injector: "PintleInjector"
    channel_geometry: CoolingChannelGeometry
    bilayer: BilayerWall
    regen_solution: dict
    structural: dict
    lcf: dict
    # --- Phase-3: geometry manifolds (Optional for backward compat) ------
    lox_dome: Optional[Any] = None
    fuel_manifold: Optional[Any] = None
    coolant_outlet_manifold: Optional[Any] = None
    coolant_inlet_manifold: Optional[Any] = None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a JSON-serializable dict.

        Channel geometry and bilayer are stored as parametric scalars
        (zone heights, thicknesses) rather than as callables, which are
        not serializable.  from_dict() reconstructs the callable objects
        from these stored parameters.

        Returns
        -------
        dict
            Fully JSON-serializable.  Round-trip via from_dict().
        """

        def _py(v: Any) -> Any:
            """Convert numpy scalar / array to a plain Python type."""
            if hasattr(v, "tolist"):
                return v.tolist()
            return v

        req = self.requirements
        prop_name = _PROPELLANT_NAMES.get(id(req.propellant), "custom")
        am_process = _AM_RULES_NAMES.get(id(req.am_rules), "lpbf-grcop")
        geom = self.channel_geometry
        nozzle = self.nozzle

        return {
            "requirements": {
                "thrust": req.thrust,
                "chamber_pressure": req.chamber_pressure,
                "mixture_ratio": req.mixture_ratio,
                "propellant": prop_name,
                "expansion_ratio": req.expansion_ratio,
                "operating_altitude": req.operating_altitude,
                "n_cycles_required": req.n_cycles_required,
                "T_wall_max": req.T_wall_max,
                "dP_coolant_budget": req.dP_coolant_budget,
                "pump_pressure_margin": req.pump_pressure_margin,
                "n_channels": req.n_channels,
                "am_process": am_process,
            },
            "gas": {f.name: getattr(self.gas, f.name) for f in dataclasses.fields(self.gas)},
            "nozzle": {f.name: getattr(nozzle, f.name) for f in dataclasses.fields(nozzle)},
            "injector": {
                f.name: getattr(self.injector, f.name)
                for f in dataclasses.fields(self.injector)
            },
            "channel_geometry": {
                "n_channels": geom.n_channels,
                "w": geom.width(0.0),
                "h01": geom.height(0.0),
                "h2": geom.height(nozzle.x_exit),
                "t_wall": geom.wall_thickness(0.0),
                "t_land": geom.land_thickness(0.0),
                "x2": 2.0 * nozzle.r_throat,
            },
            "bilayer": {
                "t_jacket": self.bilayer.outer_jacket.thickness(0.0),
            },
            "regen_solution": {k: _py(v) for k, v in self.regen_solution.items()},
            "structural": {k: _py(v) for k, v in self.structural.items()},
            "lcf": {k: _py(v) for k, v in self.lcf.items()},
        }

    def verify_all(self) -> dict[str, "VerificationReport"]:
        """Run verify() on every Component field in this design.

        Passes the active am_rules from requirements to each component's
        verify() method so manufacturability checks are included.

        Returns
        -------
        dict[str, VerificationReport]
            Maps component_name → VerificationReport for each Component field.
        """
        from noyron.kernel.component import Component, VerificationReport
        am = self.requirements.am_rules
        reports: dict[str, VerificationReport] = {}
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            if isinstance(val, Component):
                report = val.verify(am_rules=am)
                reports[report.component_name] = report
        return reports

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EngineDesign":
        """
        Reconstruct an EngineDesign from to_dict() output.

        Callable fields in CoolingChannelGeometry and BilayerWall are
        rebuilt from the stored zone parameters.  The result is
        functionally equivalent to the original design — all physics
        arrays are identical and all callable fields return the same
        values — but object identity is not preserved.

        Parameters
        ----------
        d : dict
            Output of to_dict().

        Returns
        -------
        EngineDesign
        """
        from noyron.rp.injector.pintle import PintleInjector

        # Requirements
        r = d["requirements"]
        prop_name = r["propellant"]
        if prop_name not in _PROPELLANT_PRESETS:
            raise ValueError(
                f"Unknown propellant '{prop_name}'. "
                f"Valid names: {sorted(_PROPELLANT_PRESETS)}"
            )
        am_process_key = r.get("am_process", "lpbf-grcop")
        if am_process_key not in AM_RULES_PRESETS:
            raise ValueError(
                f"Unknown am_process '{am_process_key}'. "
                f"Valid: {sorted(AM_RULES_PRESETS)}"
            )
        am_rules = AM_RULES_PRESETS[am_process_key]
        req = EngineRequirements(
            thrust=r["thrust"],
            chamber_pressure=r["chamber_pressure"],
            mixture_ratio=r["mixture_ratio"],
            propellant=_PROPELLANT_PRESETS[prop_name],
            expansion_ratio=r["expansion_ratio"],
            operating_altitude=r.get("operating_altitude", 0.0),
            n_cycles_required=r.get("n_cycles_required", 100.0),
            T_wall_max=r.get("T_wall_max", 1000.0),
            dP_coolant_budget=r.get("dP_coolant_budget", 3.0e6),
            pump_pressure_margin=r.get("pump_pressure_margin", 2.0e6),
            n_channels=r.get("n_channels", 76),
            am_rules=am_rules,
        )

        gas = CombustionGas(**d["gas"])
        nozzle = RaoNozzle(**d["nozzle"])
        injector = PintleInjector(**d["injector"])

        # Channel geometry — rebuild piecewise-constant callables
        cg = d["channel_geometry"]
        _w, _h01, _h2, _x2 = cg["w"], cg["h01"], cg["h2"], cg["x2"]
        _t_wall, _t_land = cg["t_wall"], cg["t_land"]
        channel_geometry = CoolingChannelGeometry(
            n_channels=cg["n_channels"],
            width=lambda x, __w=_w: __w,
            height=lambda x, __h01=_h01, __h2=_h2, __x2=_x2: __h01 if x <= __x2 else __h2,
            wall_thickness=lambda x, __t=_t_wall: __t,
            land_thickness=lambda x, __t=_t_land: __t,
        )

        # Bilayer — constant jacket thickness stored in dict
        _t_jacket = d["bilayer"]["t_jacket"]
        bilayer = BilayerWall(
            inner_liner=WallLayer(
                thickness=channel_geometry.wall_thickness,
                material=GRCOP84_PROXY,
            ),
            outer_jacket=WallLayer(
                thickness=lambda x, __t=_t_jacket: __t,
                material=INCONEL718_STRUCT,
            ),
            ribs=RibGeometry(
                n_ribs=channel_geometry.n_channels,
                land_width=channel_geometry.land_thickness,
                height=channel_geometry.height,
                material=GRCOP84_PROXY,
            ),
            channel_geometry=channel_geometry,
        )

        # Arrays: lists → numpy arrays; scalars stay as-is
        def _arr(v: Any) -> Any:
            return np.array(v) if isinstance(v, list) else v

        regen = {k: _arr(v) for k, v in d["regen_solution"].items()}
        struct = {k: _arr(v) for k, v in d["structural"].items()}
        lcf = dict(d["lcf"])

        return cls(
            requirements=req,
            gas=gas,
            nozzle=nozzle,
            injector=injector,
            channel_geometry=channel_geometry,
            bilayer=bilayer,
            regen_solution=regen,
            structural=struct,
            lcf=lcf,
            # manifolds not serialised — default to None
            lox_dome=None,
            fuel_manifold=None,
            coolant_outlet_manifold=None,
            coolant_inlet_manifold=None,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _thermochem(propellant: PropellantPair, MR: float, Pc: float) -> CombustionGas:
    """
    Live CEA combustion state via rocketcea.

    Returns Tc, c*, gamma, MW, cp, mu, Pr all evaluated at the given
    (Pc, MR) — no static preset values.  Results are LRU-cached inside
    cea_combustion, so repeated calls with the same arguments are fast.

    Source: Gordon & McBride (1994) NASA RP-1311; rocketcea wrapper.
    """
    from noyron.rp.thermochem.cea import cea_combustion
    return cea_combustion(propellant=propellant, Pc=Pc, MR=MR)


def _thrust_coefficient(
    eps: float,
    gamma: float,
    Pc: float,
    Pa: float = 101_325.0,
) -> float:
    """
    Sea-level thrust coefficient Cf for an ideal nozzle.

    Cf = sqrt(2γ²/(γ-1) * (2/(γ+1))^((γ+1)/(γ-1)) * [1-(Pe/Pc)^((γ-1)/γ)])
         + (Pe - Pa) / Pc * ε

    Pe/Pc from isentropic flow at exit Mach number M_e, where M_e satisfies
    the area ratio A/A* = eps.

    Source: Sutton & Biblarz (2017) Eq. (3-30), (3-31).
    Validity: 1-D ideal nozzle, frozen composition, isentropic flow.

    Parameters
    ----------
    eps : float     Expansion ratio Ae/At [-].
    gamma : float   Ratio of specific heats [-].
    Pc : float      Chamber pressure [Pa].
    Pa : float      Ambient pressure [Pa].  Default 101325 Pa (sea level).
    """
    M_e = mach_from_area_ratio(eps, gamma, supersonic=True)
    Pe_over_Pc = (1.0 + (gamma - 1.0) / 2.0 * M_e**2) ** (-gamma / (gamma - 1.0))
    Pe = Pe_over_Pc * Pc

    Cf_kinetic = math.sqrt(
        2.0 * gamma**2 / (gamma - 1.0)
        * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (gamma - 1.0))
        * (1.0 - Pe_over_Pc ** ((gamma - 1.0) / gamma))
    )
    Cf_pressure = (Pe - Pa) / Pc * eps
    return Cf_kinetic + Cf_pressure


def _mass_flows(
    thrust: float,
    gas: CombustionGas,
    MR: float,
    eps: float,
    propellant: "PropellantPair",
    Pa: float = 101_325.0,
) -> tuple[float, float, float]:
    """
    Return (m_dot_total, m_dot_ox, m_dot_fuel) [kg/s].

    If the propellant provides a reference delivered Isp (Isp_delivered_ref at
    eps_ref), the Isp is scaled to the actual eps via the ratio of isentropic
    thrust coefficients.  This accounts for variable-gamma effects that cause
    the actual Cf to exceed the constant-gamma isentropic prediction.

    Fallback: m_dot = F / (c_star * Cf_isentropic).

    Source: Sutton & Biblarz (2017) Eq. (3-30).
    Validity: 1-D nozzle flow; variable-gamma correction via reference Isp.
    """
    if (
        propellant.Isp_delivered_ref is not None
        and propellant.eps_ref is not None
    ):
        Cf_ref = _thrust_coefficient(propellant.eps_ref, gas.gamma, gas.Pc, Pa)
        Cf_actual = _thrust_coefficient(eps, gas.gamma, gas.Pc, Pa)
        Isp_actual = propellant.Isp_delivered_ref * (Cf_actual / Cf_ref)
        c_eff = Isp_actual * _G0
    else:
        Cf = _thrust_coefficient(eps, gas.gamma, gas.Pc, Pa)
        c_eff = gas.c_star * Cf

    m_dot = thrust / c_eff
    m_dot_ox = m_dot * MR / (1.0 + MR)
    m_dot_fuel = m_dot / (1.0 + MR)
    return m_dot, m_dot_ox, m_dot_fuel


def _build_bilayer(
    nozzle: RaoNozzle,
    channel_geometry: CoolingChannelGeometry,
    regen_solution: dict,
) -> BilayerWall:
    """
    Auto-size the jacket and assemble the BilayerWall.

    Jacket is sized at the exit plane (largest radius → worst hoop stress).
    Operating temperature is taken as the coolant inlet temperature + 50 K margin.

    Source: NASA SP-8087 (1977) §3.3; see calcs/bilayer_check.py for derivation.
    """
    T_coolant_inlet = float(regen_solution["T_coolant"][-1])
    T_jacket_op = T_coolant_inlet + 50.0
    P_coolant_max = float(regen_solution["P_coolant"].max())

    r_exit = float(nozzle.r_throat * math.sqrt(nozzle.epsilon))
    r_jacket_exit = r_exit + _T_LINER_M + channel_geometry.height(nozzle.x_exit)

    t_jacket_raw = size_jacket_thickness(
        P_coolant_max=P_coolant_max,
        r_jacket=r_jacket_exit,
        material=INCONEL718_STRUCT,
        T_op=T_jacket_op,
        FoS=1.4,
    )
    t_jacket = max(
        _T_JACKET_MIN_M,
        math.ceil(t_jacket_raw / _T_JACKET_STEP_M) * _T_JACKET_STEP_M,
    )

    return BilayerWall(
        inner_liner=WallLayer(
            thickness=channel_geometry.wall_thickness,
            material=GRCOP84_PROXY,
        ),
        outer_jacket=WallLayer(
            thickness=lambda x, _t=t_jacket: _t,
            material=INCONEL718_STRUCT,
        ),
        ribs=RibGeometry(
            n_ribs=channel_geometry.n_channels,
            land_width=channel_geometry.land_thickness,
            height=channel_geometry.height,
            material=GRCOP84_PROXY,
        ),
        channel_geometry=channel_geometry,
    )


def _gas_pressure_profile(
    nozzle: RaoNozzle, gas: CombustionGas, x_arr: np.ndarray
) -> np.ndarray:
    """
    Isentropic gas pressure along the nozzle axis [Pa].

    P(x) = Pc * (1 + (γ-1)/2 * M²)^(-γ/(γ-1))

    Source: Anderson (2003) "Modern Compressible Flow" Eq. (3.30).
    Validity: isentropic, perfect gas, 1-D.
    """
    g = gas.gamma
    P_arr = np.zeros(len(x_arr))
    for i, x in enumerate(x_arr):
        eps = nozzle.area_ratio(float(x))
        sup = float(x) >= 0.0
        M = mach_from_area_ratio(eps, g, supersonic=sup)
        P_arr[i] = gas.Pc * (1.0 + (g - 1.0) / 2.0 * M**2) ** (-g / (g - 1.0))
    return P_arr


# ---------------------------------------------------------------------------
# Engine class
# ---------------------------------------------------------------------------


class Engine:
    """
    Computational engine model — orchestrates all subsystem models.

    Parameters
    ----------
    requirements : EngineRequirements
    """

    def __init__(self, requirements: EngineRequirements) -> None:
        self._req = requirements
        self._cache: StepCache = StepCache()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def design(self) -> EngineDesign:
        """
        Run the full design pipeline and return a populated EngineDesign.

        Results are memoized per Engine instance via StepCache.  Calling
        design() a second time with the same requirements returns immediately
        (all 8 steps served from cache).  If requirements change — which
        cannot happen on a frozen-dataclass Engine instance, so construct a
        new Engine — build a fresh Engine instead; the old cache is discarded.

        Cache key strategy: each step's key is a tuple of the *keys* of
        every upstream step it depends on, plus the EngineRequirements
        scalars consumed directly.  Intermediate objects (CombustionGas,
        RaoNozzle, …) are not hashed; their keys serve as proxies.
        Correctness is enforced by key composition — if any input changes,
        every downstream key changes automatically.

        Steps (each depends on all prior steps):
          1. gas            thermochem CEA lookup + PropellantPair transport props
          2. nozzle         RaoNozzle contour + mass flow derivation
          3. injector       PintleInjector sizing
          4. coolant_inlet  CoolantInlet from mass flow + pressure budget
          5. channel_geom   size_channels() parametric sweep
          6. regen_soln     solve_regen() coupled 1-D solver (n=200)
          7. bilayer        WallLayer + jacket auto-sizing
          8. structural     bilayer_stress_profile() axial stress
          9. lcf            lcf_check() Manson-Coffin at worst station

        Performance note — no SDF evaluation occurs during design()
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        All geometry in design() is analytical (r_of_z interpolants, closed-form
        radii, stress integrals).  The SDF closures built by ``hollow_revolve``,
        ``BilayerWall.to_sdf()``, and ``CoolingChannelGeometry.to_sdf()`` are
        constructed lazily and only evaluated during ``render()``.  A slow
        design() call is therefore entirely due to the regen solver (step 5-6,
        typically 35-45 s for 384 channel candidates) — not SDF overhead.
        Confirmed by cProfile: 0 SDF calls in a full design() run.
        """
        req = self._req
        prop = req.propellant
        cache = self._cache

        # ------------------------------------------------------------------
        # 1. Thermochemistry
        # Key: propellant + mixture ratio + chamber pressure.
        # ------------------------------------------------------------------
        gas_key = ("gas", prop, req.mixture_ratio, req.chamber_pressure)
        gas = cache.get_or_compute(
            gas_key,
            lambda: _thermochem(prop, req.mixture_ratio, req.chamber_pressure),
        )

        # ------------------------------------------------------------------
        # 2. Nozzle (+ mass flow derivation, which depends only on gas_key)
        # Key: gas_key + thrust + expansion_ratio.
        # ------------------------------------------------------------------
        nozzle_key = ("nozzle", gas_key, req.thrust, req.expansion_ratio)

        def _compute_nozzle():
            m_dot, m_dot_ox, m_dot_fuel = _mass_flows(
                req.thrust, gas, req.mixture_ratio, req.expansion_ratio, prop
            )
            At = throat_area(m_dot, req.chamber_pressure, gas.c_star)
            r_throat = math.sqrt(At / math.pi)
            return (
                RaoNozzle(r_throat=r_throat, epsilon=req.expansion_ratio),
                m_dot, m_dot_ox, m_dot_fuel,
            )

        nozzle, m_dot, m_dot_ox, m_dot_fuel = cache.get_or_compute(
            nozzle_key, _compute_nozzle
        )

        # ------------------------------------------------------------------
        # 3. Injector
        # Key: nozzle_key + density pair (from propellant, already in gas_key
        #      via prop).  dP_inj = 0.45 * Pc is fully determined by nozzle_key.
        # ------------------------------------------------------------------
        injector_key = ("injector", nozzle_key)

        def _compute_injector():
            dP_inj = 0.45 * req.chamber_pressure
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from noyron.rp.injector.pintle import PintleInjector
                return PintleInjector.from_uniform_dP(
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

        injector = cache.get_or_compute(injector_key, _compute_injector)

        # ------------------------------------------------------------------
        # 4. Coolant inlet
        # Key: nozzle_key + pump budget scalars + coolant identity.
        # ------------------------------------------------------------------
        dP_inj = 0.45 * req.chamber_pressure
        P_coolant_inlet = req.chamber_pressure + dP_inj + req.pump_pressure_margin
        coolant_key = (
            "coolant",
            nozzle_key,
            req.pump_pressure_margin,
            prop.T_coolant_inlet,
            prop.coolant,
        )
        coolant_inlet = cache.get_or_compute(
            coolant_key,
            lambda: CoolantInlet(
                T=prop.T_coolant_inlet,
                P=P_coolant_inlet,
                m_dot=m_dot_fuel,
                fluid=prop.coolant,
            ),
        )

        # ------------------------------------------------------------------
        # 5. Cooling channel sizing
        # Key: nozzle_key + gas_key + coolant_key + thermal/pressure limits.
        # ------------------------------------------------------------------
        channel_key = (
            "channel",
            nozzle_key,
            gas_key,
            coolant_key,
            req.T_wall_max,
            req.dP_coolant_budget,
            id(req.am_rules),
        )

        def _compute_channels():
            am = req.am_rules
            sizing_problem = ChannelSizingProblem(
                nozzle=nozzle,
                gas=gas,
                coolant_inlet=coolant_inlet,
                wall_material=GRCOP42,
                T_wall_max=req.T_wall_max,
                T_wall_margin=0.0,   # Pizzarelli 2016 credit: DB underpredicts h_c 2-5x; T_wall_max already absorbs the margin
                dP_max=req.dP_coolant_budget,
                min_wall_thickness=am.min_wall_thickness,
                max_aspect_ratio=am.max_internal_aspect_ratio,
                min_channel_width=am.min_channel_width,
                min_channel_height=am.min_channel_height,
                t_land_min=am.min_feature_size,
                n_stations_opt=60,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return size_channels(sizing_problem)

        channel_geometry = cache.get_or_compute(channel_key, _compute_channels)

        # ------------------------------------------------------------------
        # 6. Regen solution (n=200 validation solve)
        # Key: all inputs to solve_regen that are not already in channel_key.
        # ------------------------------------------------------------------
        regen_key = ("regen", channel_key)

        def _compute_regen():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return _solve_regen(
                    nozzle=nozzle,
                    gas=gas,
                    channel_geom=channel_geometry,
                    coolant_inlet=coolant_inlet,
                    wall_material=GRCOP42,
                    direction="counterflow",
                    n_stations=200,
                    tol=0.5,
                )

        regen_solution = cache.get_or_compute(regen_key, _compute_regen)

        # ------------------------------------------------------------------
        # 7. Bilayer wall sizing
        # Key: regen_key (includes nozzle + channel geometry transitively).
        # ------------------------------------------------------------------
        bilayer_key = ("bilayer", regen_key)
        bilayer = cache.get_or_compute(
            bilayer_key,
            lambda: _build_bilayer(nozzle, channel_geometry, regen_solution),
        )

        # ------------------------------------------------------------------
        # 8. Structural analysis
        # Key: regen_key + bilayer_key (gas is already in regen_key).
        # ------------------------------------------------------------------
        structural_key = ("structural", regen_key, bilayer_key)

        def _compute_structural():
            P_gas_arr = _gas_pressure_profile(nozzle, gas, regen_solution["x"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return bilayer_stress_profile(
                    regen_solution=regen_solution,
                    bilayer=bilayer,
                    P_gas_arr=P_gas_arr,
                )

        structural = cache.get_or_compute(structural_key, _compute_structural)

        # ------------------------------------------------------------------
        # 9. LCF check
        # Key: structural_key + n_cycles_required.
        # n_cycles_required is the ONLY EngineRequirements field that does
        # not affect upstream steps, so changing it re-runs only this step.
        # ------------------------------------------------------------------
        lcf_key = ("lcf", structural_key, req.n_cycles_required)
        lcf = cache.get_or_compute(
            lcf_key,
            lambda: lcf_check(
                regen_solution=regen_solution,
                structural_solution=structural,
                n_cycles_required=req.n_cycles_required,
                material=GRCOP84_PROXY,
            ),
        )

        # ------------------------------------------------------------------
        # 10. Geometry manifolds (no physics — pure geometry, no cache key)
        # ------------------------------------------------------------------
        from noyron.rp.geometry.manifolds import (
            ManifoldConfig,
            LoxDome,
            FuelManifold,
            CoolantOutletManifold,
            CoolantInletManifold,
        )
        _cfg = ManifoldConfig()
        _lox_dome = LoxDome(nozzle=nozzle, bilayer=bilayer, injector=injector, config=_cfg)
        _fuel_mfld = FuelManifold(nozzle=nozzle, bilayer=bilayer, injector=injector, config=_cfg)
        _cool_out = CoolantOutletManifold(nozzle=nozzle, bilayer=bilayer, config=_cfg)
        _cool_in = CoolantInletManifold(nozzle=nozzle, bilayer=bilayer, config=_cfg)

        return EngineDesign(
            requirements=req,
            gas=gas,
            nozzle=nozzle,
            injector=injector,
            channel_geometry=channel_geometry,
            bilayer=bilayer,
            regen_solution=regen_solution,
            structural=structural,
            lcf=lcf,
            lox_dome=_lox_dome,
            fuel_manifold=_fuel_mfld,
            coolant_outlet_manifold=_cool_out,
            coolant_inlet_manifold=_cool_in,
        )

    def render(
        self,
        design: EngineDesign,
        voxel_size: float = 1e-4,
        output_dir: Path = Path("output"),
        components: tuple[str, ...] = ("chamber_nozzle", "channels"),
    ) -> dict[str, Path]:
        """
        Export STL meshes for the specified geometry components.

        Parameters
        ----------
        design : EngineDesign
        voxel_size : float          Voxel grid spacing [m].  Default 0.1 mm.
        output_dir : Path           Output directory.
        components : tuple[str]     Subset of:
                                      'chamber_nozzle' — outer jacket only
                                      'channels'       — jacket with cooling channel voids
                                      'injector'       — face plate + pintle post only
                                      'assembly'       — channels + injector combined

        Returns
        -------
        dict[str, Path]  Maps component name to written STL path.

        Raises
        ------
        NotImplementedError
            For unknown component names.
        """
        from noyron.rp.geometry.chamber_nozzle import chamber_nozzle_sdf
        from noyron.rp.geometry.cooling_channels import chamber_with_channels_sdf
        from noyron.rp.geometry.injector import engine_assembly_sdf, injector_sdf
        from noyron.kernel.geometry.mesh.chunked_extractor import chunked_sdf_to_mesh
        from noyron.kernel.geometry.mesh.extractor import adaptive_bbox
        from noyron.kernel.geometry.mesh.stl_export import write_stl

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out: dict[str, Path] = {}

        for comp in components:
            if comp == "chamber_nozzle":
                sdf_fn, profile = chamber_nozzle_sdf(
                    rao_nozzle=design.nozzle,
                    bilayer=design.bilayer,
                )
            elif comp == "channels":
                sdf_fn, profile = chamber_with_channels_sdf(
                    rao_nozzle=design.nozzle,
                    bilayer=design.bilayer,
                )
            elif comp == "injector":
                sdf_fn, inj_info = injector_sdf(
                    pintle=design.injector,
                    nozzle=design.nozzle,
                    bilayer=design.bilayer,
                )
                r_outer = inj_info["r_ch_inner"]
                profile = {
                    "z_min": inj_info["z_face"],
                    "z_max": inj_info["z_pintle_tip"],
                    "r_outer_max": r_outer,
                }
            elif comp == "assembly":
                sdf_fn, profile = engine_assembly_sdf(
                    nozzle=design.nozzle,
                    bilayer=design.bilayer,
                    channel_geometry=design.channel_geometry,
                    pintle=design.injector,
                )
            elif comp == "full_engine":
                from noyron.kernel.geometry.sdf.operations import union as _sdf_union
                sdf_fn, profile = engine_assembly_sdf(
                    nozzle=design.nozzle,
                    bilayer=design.bilayer,
                    channel_geometry=design.channel_geometry,
                    pintle=design.injector,
                )
                z_min = float(profile["z_min"])
                z_max = float(profile["z_max"])
                r_outer_max = float(profile["r_outer_max"])
                for _mfld in (
                    design.lox_dome,
                    design.fuel_manifold,
                    design.coolant_outlet_manifold,
                    design.coolant_inlet_manifold,
                ):
                    if _mfld is not None:
                        sdf_fn = _sdf_union(sdf_fn, _mfld.to_sdf())
                        _lo, _hi = _mfld.bbox()
                        z_min = min(z_min, float(_lo[2]))
                        z_max = max(z_max, float(_hi[2]))
                        _r = max(
                            abs(float(_lo[0])), float(_hi[0]),
                            abs(float(_lo[1])), float(_hi[1]),
                        )
                        r_outer_max = max(r_outer_max, _r)
                profile = {
                    "z_min": z_min,
                    "z_max": z_max,
                    "r_outer_max": r_outer_max,
                }
            else:
                raise NotImplementedError(
                    f"render component '{comp}' not implemented. "
                    "Supported: 'chamber_nozzle', 'channels', 'injector', "
                    "'assembly', 'full_engine'."
                )

            bbox = adaptive_bbox(
                nozzle_x_min=profile["z_min"],
                nozzle_x_max=profile["z_max"],
                r_outer_max=profile["r_outer_max"],
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                verts, faces = chunked_sdf_to_mesh(
                    sdf_fn,
                    bbox,
                    voxel_size=voxel_size,
                    slab_axial_steps=25,
                    verbose=False,
                )

            stl_path = output_dir / f"{comp}.stl"
            write_stl(verts, faces, stl_path, scale_to_mm=True)
            out[comp] = stl_path

        return out

    def summary(self, design: EngineDesign) -> str:
        """
        Return a human-readable one-page summary of the engine design.

        Contains: thrust, Isp (estimated), peak T_wall, channel geometry,
        coolant ΔP, peak structural margins, and LCF life/margin.
        """
        req = design.requirements
        gas = design.gas
        nozzle = design.nozzle
        inj = design.injector
        geom = design.channel_geometry
        regen = design.regen_solution
        struct = design.structural
        lcf = design.lcf

        # Derived quantities
        prop = req.propellant
        m_dot, m_dot_ox, m_dot_fuel = _mass_flows(
            req.thrust, gas, req.mixture_ratio, req.expansion_ratio, prop
        )
        At = nozzle.A_throat
        c_eff = req.thrust / m_dot
        Isp = c_eff / _G0
        from noyron.rp.thermochem.cea import cea_isp_vac
        Isp_vacuum_cea = cea_isp_vac(
            propellant=req.propellant,
            Pc=req.chamber_pressure,
            MR=req.mixture_ratio,
            eps=req.expansion_ratio,
        )
        T_wall_peak = float(regen["T_wall_gas"].max())
        dP_total = float(regen["dP_total"])

        n_ch = geom.n_channels
        w_th_mm = geom.width(0.0) * 1e3
        h_th_mm = geom.height(0.0) * 1e3

        min_margin_inner = float(struct["margin_yield_inner"].min())
        min_margin_jacket = float(struct["margin_yield_jacket"].min())
        min_margin_rib = float(struct["margin_yield_rib"].min())

        N_f = lcf["N_f"]
        margin_lcf = lcf["margin_lcf"]
        x_worst_mm = lcf["x_worst"] * 1e3

        am = design.requirements.am_rules
        lines = [
            "=" * 72,
            "  Engine Design Summary",
            f"  Manufacturing process : {am.process_name} ({am.material_family})",
            "=" * 72,
            f"  Thrust               : {req.thrust:.1f} N  ({req.thrust/4.44822:.1f} lbf)",
            f"  Pc                   : {req.chamber_pressure/1e6:.3f} MPa  "
            f"({req.chamber_pressure/6894.757:.0f} psia)",
            f"  O/F                  : {req.mixture_ratio:.3f}",
            f"  Expansion ratio      : {req.expansion_ratio:.3f}",
            f"  m_dot total          : {m_dot:.4f} kg/s",
            f"  m_dot ox             : {m_dot_ox:.4f} kg/s",
            f"  m_dot fuel           : {m_dot_fuel:.4f} kg/s",
            f"  c*  (CEA theor.)     : {gas.c_star:.1f} m/s",
            f"  c_eff (delivered)    : {c_eff:.1f} m/s",
            f"  Isp (delivered SL)   : {Isp:.2f} s",
            f"  Isp_vac (CEA theor.) : {Isp_vacuum_cea:.2f} s",
            "",
            "  Nozzle",
            f"    R_throat           : {nozzle.r_throat*1e3:.3f} mm",
            f"    D_throat           : {nozzle.r_throat*2e3:.3f} mm",
            f"    A_throat           : {At*1e4:.4f} cm²",
            f"    R_exit             : {nozzle.r_throat*math.sqrt(nozzle.epsilon)*1e3:.3f} mm",
            "",
            "  Injector (pintle)",
            f"    m_dot fuel         : {inj.m_dot_fuel:.4f} kg/s",
            f"    m_dot ox           : {inj.m_dot_ox:.4f} kg/s",
            f"    Fuel orifice D     : {inj.fuel_orifice_diameter*1e3:.3f} mm  "
            f"× {inj.n_fuel_orifices}",
            f"    LOX annulus OD     : {inj.lox_annulus_OD*1e3:.3f} mm",
            f"    TMR                : {inj.TMR:.3f}",
            "",
            "  Regen cooling",
            f"    n_channels         : {n_ch}",
            f"    Channel w×h (throat): {w_th_mm:.2f} × {h_th_mm:.2f} mm",
            f"    Peak T_wall_gas    : {T_wall_peak:.0f} K  "
            f"(DB-model; real T_wall ~300-600 K lower)",
            f"    Coolant dP         : {dP_total/1e6:.3f} MPa",
            "",
            "  Structural (bilayer)",
            f"    Min margin (liner) : {min_margin_inner:.3f}"
            + ("  (LCF regime, expected)" if min_margin_inner < 0 else ""),
            f"    Min margin (jacket): {min_margin_jacket:.3f}",
            f"    Min margin (rib)   : {min_margin_rib:.3f}",
            "",
            "  LCF (Manson-Coffin, liner, GRCop-84 proxy)",
            f"    Worst station      : x = {x_worst_mm:.1f} mm",
            f"    T_wall_gas worst   : {lcf['T_gas_worst']:.0f} K",
            f"    N_f predicted      : {N_f:.0f} cycles",
            f"    Required life      : {req.n_cycles_required:.0f} cycles",
            f"    LCF margin         : {margin_lcf:.3f}"
            + ("  [PASS]" if margin_lcf >= 0 else "  [FAIL — increase dP_coolant_budget]"),
        ]

        # Verification block
        verification = design.verify_all()
        lines.append("")
        lines.append("  Verification")
        for comp_name, report in verification.items():
            warn_count = sum(1 for i in report.issues if i.severity == "warning")
            err_count = sum(1 for i in report.issues if i.severity == "error")
            if err_count:
                status = f"FAIL ({err_count} error{'s' if err_count != 1 else ''})"
            elif warn_count:
                status = f"PASS ({warn_count} warning{'s' if warn_count != 1 else ''})"
            else:
                status = "PASS"
            lines.append(f"    {comp_name:<26}: {status}")
            for issue in report.issues:
                tag = "WARNING" if issue.severity == "warning" else "ERROR  "
                lines.append(f"      {tag}: {issue.message}")
        lines.append("=" * 72)

        return "\n".join(lines)
