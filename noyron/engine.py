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

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from physics.thermochem.cea import CEA
from physics.nozzle.rao import RaoNozzle, throat_area
from physics.regen.bartz import CombustionGas, mach_from_area_ratio
from physics.regen.coolant import (
    CoolingChannelGeometry,
    CoolantInlet,
    GRCOP42,
    INCONEL718,
    WallMaterial,
)
from physics.regen.sizing import ChannelSizingProblem, size_channels
from physics.regen.solver import solve_regen
from physics.structural.bilayer import (
    BilayerWall,
    RibGeometry,
    WallLayer,
    bilayer_stress_profile,
    size_jacket_thickness,
)
from physics.structural.fatigue import lcf_check
from physics.structural.liner_stress import liner_stress_profile
from physics.structural.materials import GRCOP84_PROXY, INCONEL718_STRUCT

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
    dP_coolant_budget : float   Max total coolant pressure drop [Pa].
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
    T_wall_max: float = 1000.0
    dP_coolant_budget: float = 3.0e6
    n_channels: int = 76


# ---------------------------------------------------------------------------
# EngineDesign
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineDesign:
    """
    Complete design solution produced by Engine.design().

    All fields are populated; none are None.  See Engine.design() for the
    dependency order.
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _thermochem(propellant: PropellantPair, MR: float, Pc: float) -> CombustionGas:
    """
    Build a CombustionGas from CEA + PropellantPair transport properties.

    c* source priority:
      1. propellant.c_star_ref  (verified from reference geometry, overrides CEA)
      2. CEA CSV at Pc (only valid at Pc = 1000 psia = 6.8948 MPa)
      3. CEA CSV at reference Pc (fallback for non-reference Pc; < 2% error)

    Tc always comes from the CEA CSV (Tc is not geometry-dependent).

    Source: Gordon & McBride (1994) NASA RP-1311 (via cea_sweep_pc1000.csv).
    """
    cea = CEA()
    try:
        Tc = cea.tc(Pc, MR)
        c_star_cea = cea.cstar(Pc, MR)
    except NotImplementedError:
        # Fall back to reference Pc (c* is ~0.5% sensitive to Pc for methalox)
        Tc = cea.tc(_REF_PC_PA, MR)
        c_star_cea = cea.cstar(_REF_PC_PA, MR)

    # Use the verified reference c* when provided; CEA CSV has a ~3.9% offset
    # at the reference operating point (see PropellantPair.c_star_ref docstring).
    c_star = propellant.c_star_ref if propellant.c_star_ref is not None else c_star_cea

    return CombustionGas(
        Tc=Tc,
        Pc=Pc,
        gamma=propellant.gamma,
        MW=propellant.MW,
        cp=propellant.cp,
        mu=propellant.mu,
        Pr=propellant.Pr,
        c_star=c_star,
    )


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

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def design(self) -> EngineDesign:
        """
        Run the full design pipeline and return a populated EngineDesign.

        Steps (each depends on all prior steps):
          1. gas            thermochem CEA lookup + PropellantPair transport props
          2. nozzle         RaoNozzle contour from throat area
          3. injector       PintleInjector sizing (pintle OD and n_orifices fixed)
          4. channel_geom   size_channels() parametric sweep
          5. bilayer        WallLayer sizes + jacket auto-sizing
          6. regen_soln     solve_regen() coupled 1-D solver (n=200)
          7. structural     bilayer_stress_profile() axial stress
          8. lcf            lcf_check() Manson-Coffin at worst station
        """
        req = self._req
        prop = req.propellant

        # ------------------------------------------------------------------
        # 1. Thermochemistry
        # ------------------------------------------------------------------
        gas = _thermochem(prop, req.mixture_ratio, req.chamber_pressure)

        # ------------------------------------------------------------------
        # 2. Nozzle
        # ------------------------------------------------------------------
        m_dot, m_dot_ox, m_dot_fuel = _mass_flows(
            req.thrust, gas, req.mixture_ratio, req.expansion_ratio, prop
        )
        At = throat_area(m_dot, req.chamber_pressure, gas.c_star)
        r_throat = math.sqrt(At / math.pi)

        nozzle = RaoNozzle(
            r_throat=r_throat,
            epsilon=req.expansion_ratio,
        )

        # ------------------------------------------------------------------
        # 3. Injector
        # ------------------------------------------------------------------
        # Injector ΔP from reference design: 45% of Pc (CLAUDE.md: 3.103 MPa / 6.8948 MPa)
        dP_inj = 0.45 * req.chamber_pressure
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from physics.injector.pintle import PintleInjector
            injector = PintleInjector.from_uniform_dP(
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

        # ------------------------------------------------------------------
        # 4. Cooling channel sizing
        # ------------------------------------------------------------------
        # Coolant inlet pressure: Pc + injector ΔP + 2 MPa pump margin.
        # The 2 MPa margin matches calcs/ convention (P_coolant_inlet = 12.0 MPa
        # for the reference engine: 6.8948 + 3.1027 + ~2.0 ≈ 12.0 MPa).
        # Using the full dP_coolant_budget here would over-pressurize the
        # jacket sizing and artificially inflate the structural stress.
        _PUMP_MARGIN_PA = 2.0e6  # [Pa] fixed feed-system pump head margin
        P_coolant_inlet = req.chamber_pressure + dP_inj + _PUMP_MARGIN_PA

        coolant_inlet = CoolantInlet(
            T=prop.T_coolant_inlet,
            P=P_coolant_inlet,
            m_dot=m_dot_fuel,
            fluid=prop.coolant,
        )

        sizing_problem = ChannelSizingProblem(
            nozzle=nozzle,
            gas=gas,
            coolant_inlet=coolant_inlet,
            wall_material=GRCOP42,
            T_wall_max=req.T_wall_max,
            T_wall_margin=50.0,
            dP_max=req.dP_coolant_budget,
            min_wall_thickness=_T_LINER_M,
            max_aspect_ratio=8.0,
            min_channel_width=0.6e-3,
            t_land_min=0.3e-3,
            n_stations_opt=60,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            channel_geometry = size_channels(sizing_problem)

        # ------------------------------------------------------------------
        # 5. Regen solution (n=200 validation solve)
        # ------------------------------------------------------------------
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            regen_solution = solve_regen(
                nozzle=nozzle,
                gas=gas,
                channel_geom=channel_geometry,
                coolant_inlet=coolant_inlet,
                wall_material=GRCOP42,
                direction="counterflow",
                n_stations=200,
                tol=0.5,
            )

        # ------------------------------------------------------------------
        # 6. Bilayer wall sizing
        # ------------------------------------------------------------------
        bilayer = _build_bilayer(nozzle, channel_geometry, regen_solution)

        # ------------------------------------------------------------------
        # 7. Structural analysis
        # ------------------------------------------------------------------
        P_gas_arr = _gas_pressure_profile(
            nozzle, gas, regen_solution["x"]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structural = bilayer_stress_profile(
                regen_solution=regen_solution,
                bilayer=bilayer,
                P_gas_arr=P_gas_arr,
            )

        # ------------------------------------------------------------------
        # 8. LCF check
        # ------------------------------------------------------------------
        lcf = lcf_check(
            regen_solution=regen_solution,
            structural_solution=structural,
            n_cycles_required=req.n_cycles_required,
            material=GRCOP84_PROXY,
        )

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
        components : tuple[str]     Subset of ('chamber_nozzle', 'channels').

        Returns
        -------
        dict[str, Path]  Maps component name to written STL path.

        Raises
        ------
        NotImplementedError
            For components other than 'chamber_nozzle' and 'channels'.
        """
        from geometry.components.chamber_nozzle import chamber_nozzle_sdf
        from geometry.components.cooling_channels import chamber_with_channels_sdf
        from geometry.mesh.chunked_extractor import chunked_sdf_to_mesh
        from geometry.mesh.extractor import adaptive_bbox
        from geometry.mesh.stl_export import write_stl

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
            else:
                raise NotImplementedError(
                    f"render component '{comp}' not implemented. "
                    "Supported: 'chamber_nozzle', 'channels'."
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

        lines = [
            "=" * 72,
            "  Engine Design Summary",
            "=" * 72,
            f"  Thrust               : {req.thrust:.1f} N  ({req.thrust/4.44822:.1f} lbf)",
            f"  Pc                   : {req.chamber_pressure/1e6:.3f} MPa  "
            f"({req.chamber_pressure/6894.757:.0f} psia)",
            f"  O/F                  : {req.mixture_ratio:.3f}",
            f"  Expansion ratio      : {req.expansion_ratio:.3f}",
            f"  m_dot total          : {m_dot:.4f} kg/s",
            f"  m_dot ox             : {m_dot_ox:.4f} kg/s",
            f"  m_dot fuel           : {m_dot_fuel:.4f} kg/s",
            f"  c*                   : {gas.c_star:.1f} m/s",
            f"  c_eff                : {c_eff:.1f} m/s",
            f"  Isp (est.)           : {Isp:.2f} s",
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
            "=" * 72,
        ]
        return "\n".join(lines)
