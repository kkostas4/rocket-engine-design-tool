"""
bilayer.py
==========

Analytical stress model for a bilayer regenerative-cooling chamber wall:
inner liner (high-k copper alloy) + outer jacket (high-strength superalloy)
joined by integral ribs (milled-channel construction).

Wall construction (milled-channel, the RS-25 / standard flight architecture)
------------------------------------------------------------------------------

    |  gas  |  t_liner  |  channel height  |  t_jacket  |  atmosphere  |
                         ^^^^^^^^^^^^^^^^
                         ribs at pitch = w_ch + t_land

The inner liner is a thin copper-alloy shell.  Coolant channels are milled
into its outer surface.  The outer jacket (Inconel 718 or similar) is
electroformed, brazed, or press-fit over the liner.

Stress decomposition (NASA SP-8087 §3.3)
-----------------------------------------
1. Inner-liner hoop (differential pressure across liner only):
       sigma_h_inner = (P_coolant - P_gas) * r_inner / t_liner        [tensile]
   The liner sees differential pressure — gas pushes out, coolant pushes in.
   Net is tensile (P_coolant > P_gas in a regen system).
   Source: NASA SP-8087 (1977) §3.3, Eq. 3-3.

2. Outer-jacket hoop (full coolant pressure; atmosphere outside):
       sigma_h_jacket = P_coolant * r_jacket / t_jacket                [tensile]
   The jacket is a pressure vessel containing the coolant.
   Source: Timoshenko & Goodier (1951) §131; NASA SP-8087 §3.3.

3. Rib shear (differential pressure loading on rib span):
       tau_rib = (P_coolant - P_gas) * w_channel / (2 * t_land)
   The rib acts as a cantilever beam loaded by the net pressure differential
   over the channel span w_channel.  The factor of 2 comes from both sides
   of the rib sharing the load.
   Source: Roark's Formulas for Stress and Strain (Young & Budynas, 8th ed.),
   Table 8.1 (cantilever with uniform lateral load).
   NASA SP-8087 §3.3 gives an equivalent form.

4. Thermal stress in inner liner: same linear-gradient method as liner_stress.py,
   evaluated across t_liner only.  The thermal gradient is steeper in a thin
   liner → higher sigma_th for same dT.

5. Thermal stress in outer jacket: jacket runs near coolant temperature.
   Gradient from T_coolant_wall to T_jacket_outer (assumed = T_coolant, since
   jacket thermal resistance is small at low q).  Computed and reported but
   typically << liner thermal stress.

Note on rib construction
------------------------
In milled-channel construction the rib is integral to the inner liner
(Cu alloy).  The jacket is applied afterward.  The rib material is therefore
GRCOP-84 (same as liner) and its yield criterion uses the liner material.

Note on differential thermal expansion
---------------------------------------
The liner runs hotter than the jacket → liner wants to grow more in the hoop
direction.  This introduces a secondary hoop tension in the jacket and
compression in the liner beyond what the pressure loading alone predicts.
This effect requires coupled FEA to quantify accurately and is OUT OF SCOPE
for this first-pass analytical model.  It is flagged here for future work.

Sources
-------
NASA SP-8087 (1977). "Liquid Rocket Engine Fluid-Cooled Combustion Chambers."
    §3.3 Wall Construction, §4.2 Mechanical Stress Analysis,
    §4.3 Thermal Stress Analysis.

Cook, R.L. and Coffey, G.A. (1973). "Space Shuttle Orbiter Main Engine
    Cooling." AIAA Paper 73-1364. (Bilayer liner + jacket architecture.)

Esposito, J.J. and Zappa, R.F. (1975). "Thrust Chamber Life Prediction."
    NASA CR-2503. (RS-25 NARloy-Z liner + Inconel 718 jacket; stress model.)

Quentmeyer, R.J. (1977). "Experimental Fatigue Life Investigation of
    Cylindrical Thrust Chambers." NASA TM-X-73665. §2 (rib shear, bilayer).

Timoshenko, S. and Goodier, J.N. (1951). "Theory of Elasticity." §131.

Roark, R.J. and Young, W.C. (2002). "Roark's Formulas for Stress and Strain,"
    8th ed.  Table 8.1 (cantilever uniform-load shear).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from physics.structural.liner_stress import (
    through_wall_thermal_stress,
    von_mises,
)
from physics.structural.margin import FOS_YIELD, FOS_ULTIMATE
from noyron.kernel.component import Component, Renderable, VerificationReport

if TYPE_CHECKING:
    from physics.regen.coolant import CoolingChannelGeometry
    from physics.structural.materials import StructuralMaterial


# ---------------------------------------------------------------------------
# Geometry data models
# ---------------------------------------------------------------------------


@dataclass
class WallLayer:
    """
    One layer (liner or jacket) of the bilayer wall.

    Parameters
    ----------
    thickness : Callable[[float], float]
        Layer thickness t(x) [m] as a function of axial coordinate x [m].
    material : StructuralMaterial
        Structural material of this layer.
    """

    thickness: Callable[[float], float]
    material: "StructuralMaterial"


@dataclass
class RibGeometry:
    """
    Integral rib (land) geometry for milled-channel construction.

    The rib is the wall of copper alloy between adjacent coolant channels.
    In milled-channel construction the rib material is the same as the inner
    liner (the rib is machined from the liner blank).

    Parameters
    ----------
    n_ribs : int
        Number of ribs = number of channels.
    land_width : Callable[[float], float]
        Rib (land) width t_land(x) [m].  Same as CoolingChannelGeometry.land_thickness.
    height : Callable[[float], float]
        Rib height = channel height h_ch(x) [m].
    material : StructuralMaterial
        Rib material (same as liner for milled-channel).
    """

    n_ribs: int
    land_width: Callable[[float], float]
    height: Callable[[float], float]
    material: "StructuralMaterial"


@dataclass
class BilayerWall(Component, Renderable):
    """
    Complete bilayer wall description.

    Parameters
    ----------
    inner_liner : WallLayer
        Gas-side liner (copper alloy, thin).
    outer_jacket : WallLayer
        Structural outer jacket (Inconel 718 or similar).
    ribs : RibGeometry
        Integral ribs connecting liner to jacket.
    channel_geometry : CoolingChannelGeometry
        Cooling channel geometry (provides width and n_channels).
    """

    inner_liner: WallLayer
    outer_jacket: WallLayer
    ribs: RibGeometry
    channel_geometry: "CoolingChannelGeometry"

    # ------------------------------------------------------------------
    # Component interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "BilayerWall"

    def verify(self) -> VerificationReport:
        """Self-check returning a VerificationReport.

        Checks: liner and jacket thickness > 0 at x=0, rib count > 0.

        Source: NASA SP-8087 (1977) §3.3 — bilayer wall geometry constraints.
        """
        report = VerificationReport.ok(self.name)
        t_lin = self.inner_liner.thickness(0.0)
        t_jkt = self.outer_jacket.thickness(0.0)
        if t_lin <= 0.0:
            report = report.add(
                "error",
                f"inner_liner thickness at x=0 = {t_lin:.4f} m must be positive",
                metric="t_liner", value=t_lin, threshold=0.0,
            )
        if t_jkt <= 0.0:
            report = report.add(
                "error",
                f"outer_jacket thickness at x=0 = {t_jkt:.4f} m must be positive",
                metric="t_jacket", value=t_jkt, threshold=0.0,
            )
        if self.ribs.n_ribs <= 0:
            report = report.add(
                "error",
                f"n_ribs = {self.ribs.n_ribs} must be positive",
                metric="n_ribs", value=float(self.ribs.n_ribs), threshold=1.0,
            )
        return report

    # ------------------------------------------------------------------
    # Renderable interface
    # ------------------------------------------------------------------

    def to_sdf(self, **context: Any):
        """SDF of the outer jacket (without channel voids).

        Requires ``nozzle`` (RaoNozzle) in context.
        """
        nozzle = context["nozzle"]
        from geometry.components.chamber_nozzle import chamber_nozzle_sdf
        sdf, _ = chamber_nozzle_sdf(rao_nozzle=nozzle, bilayer=self)
        return sdf

    def bbox(self, **context: Any) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box of the outer jacket.

        Requires ``nozzle`` (RaoNozzle) in context.
        """
        nozzle = context["nozzle"]
        r_exit = nozzle.r_throat * math.sqrt(nozzle.epsilon)
        r_chamber = nozzle.r_chamber
        r_max = max(
            r_chamber + self.inner_liner.thickness(0.0)
            + self.channel_geometry.height(0.0) + self.outer_jacket.thickness(0.0),
            r_exit + self.inner_liner.thickness(nozzle.x_exit)
            + self.channel_geometry.height(nozzle.x_exit)
            + self.outer_jacket.thickness(nozzle.x_exit),
        )
        return (
            np.array([nozzle.x_chamber_start, -r_max, -r_max]),
            np.array([nozzle.x_exit, r_max, r_max]),
        )


# ---------------------------------------------------------------------------
# Outer jacket auto-sizer
# ---------------------------------------------------------------------------


def size_jacket_thickness(
    *,
    P_coolant_max: float,
    r_jacket: float,
    material: "StructuralMaterial",
    T_op: float,
    FoS: float = FOS_YIELD,
) -> float:
    """
    Minimum jacket thickness to satisfy the yield-limited thin-shell criterion.

    From thin-shell hoop stress:
        sigma_h = P * r / t  <=  sigma_y(T_op) / FoS
    Solving for t:
        t_min = P * r * FoS / sigma_y(T_op)

    Source: Timoshenko & Goodier (1951) §131; NASA SP-8087 §3.3.

    Parameters
    ----------
    P_coolant_max : float
        Maximum coolant pressure the jacket must contain [Pa].
        Use inlet pressure (highest) for conservatism.
    r_jacket : float
        Inner radius of the jacket [m].
        r_jacket = r_inner + t_liner + h_channel.
    material : StructuralMaterial
        Jacket material.
    T_op : float
        Operating temperature of the jacket [K].
        Jacket runs near coolant temperature; use the coolant inlet T + margin.
    FoS : float
        Factor of safety on yield.  Default 1.4 (NASA-STD-5001).

    Returns
    -------
    float
        Minimum jacket thickness t_jacket [m].
    """
    sy = material.sigma_y(T_op)
    if sy <= 0.0:
        raise ValueError(f"sigma_y({T_op} K) = {sy:.1f} Pa: non-positive yield strength")
    return P_coolant_max * r_jacket * FoS / sy


# ---------------------------------------------------------------------------
# Inner liner hoop (differential)
# ---------------------------------------------------------------------------


def _liner_hoop(P_coolant: float, P_gas: float, r_inner: float, t_liner: float) -> float:
    """
    Hoop stress in the inner liner from differential pressure.

    sigma_h_inner = (P_coolant - P_gas) * r_inner / t_liner

    Positive = tensile (P_coolant > P_gas in a regen system).

    Source: NASA SP-8087 (1977) §3.3, Eq. 3-3.
    """
    if t_liner <= 0.0:
        raise ValueError(f"t_liner must be > 0, got {t_liner}")
    if r_inner <= 0.0:
        raise ValueError(f"r_inner must be > 0, got {r_inner}")
    dP = P_coolant - P_gas
    t_over_r = t_liner / r_inner
    if t_over_r > 0.1:
        warnings.warn(
            f"Liner t/r = {t_over_r:.3f} > 0.1: thin-wall approximation may be inaccurate.",
            RuntimeWarning,
            stacklevel=3,
        )
    return dP * r_inner / t_liner


# ---------------------------------------------------------------------------
# Outer jacket hoop (full coolant pressure)
# ---------------------------------------------------------------------------


def _jacket_hoop(P_coolant: float, r_jacket: float, t_jacket: float) -> float:
    """
    Hoop stress in the outer jacket from coolant pressure (atmosphere outside).

    sigma_h_jacket = P_coolant * r_jacket / t_jacket

    Source: Timoshenko & Goodier (1951) §131; NASA SP-8087 §3.3.
    """
    if t_jacket <= 0.0:
        raise ValueError(f"t_jacket must be > 0, got {t_jacket}")
    if r_jacket <= 0.0:
        raise ValueError(f"r_jacket must be > 0, got {r_jacket}")
    t_over_r = t_jacket / r_jacket
    if t_over_r > 0.1:
        warnings.warn(
            f"Jacket t/r = {t_over_r:.3f} > 0.1: thin-wall approximation may be inaccurate.",
            RuntimeWarning,
            stacklevel=3,
        )
    return P_coolant * r_jacket / t_jacket


# ---------------------------------------------------------------------------
# Rib shear stress
# ---------------------------------------------------------------------------


def _rib_shear(
    P_coolant: float, P_gas: float, w_channel: float, t_land: float
) -> float:
    """
    Peak shear stress at the rib root from differential pressure.

    The rib is treated as a cantilever beam of width t_land, uniformly loaded
    by the net pressure differential dP = P_coolant - P_gas over the channel
    span w_channel.  The factor-of-2 accounts for both sides of the rib
    sharing the load (the rib is bounded by two channels).

        tau_rib = dP * w_channel / (2 * t_land)

    Source: Roark's Formulas for Stress and Strain (Young & Budynas, 8th ed.),
    Table 8.1 (cantilever, uniform lateral load, peak shear at root).
    NASA SP-8087 (1977) §3.3 gives an equivalent expression.

    Parameters
    ----------
    P_coolant : float   Coolant pressure at this station [Pa].
    P_gas : float       Gas pressure at this station [Pa].
    w_channel : float   Channel width (rib span) [m].
    t_land : float      Rib/land width [m].

    Returns
    -------
    float
        Peak shear stress [Pa], always >= 0.
    """
    if t_land <= 0.0:
        raise ValueError(f"t_land must be > 0, got {t_land}")
    dP = max(0.0, P_coolant - P_gas)  # only compressive rib loading is considered
    return dP * w_channel / (2.0 * t_land)


# ---------------------------------------------------------------------------
# Full axial bilayer profile
# ---------------------------------------------------------------------------


def bilayer_stress_profile(
    *,
    regen_solution: dict,
    bilayer: BilayerWall,
    P_gas_arr: "np.ndarray | None" = None,
    P_atm: float = 101_325.0,
    fos_yield: float = FOS_YIELD,
    fos_ultimate: float = FOS_ULTIMATE,
) -> dict:
    """
    Axial stress profile for the bilayer (liner + jacket + ribs) wall.

    At each axial station:

    INNER LINER
    -----------
    1. T_liner_mean = (T_wall_gas + T_wall_coolant) / 2
       Material properties evaluated at T_liner_mean.
    2. sigma_h_inner = (P_coolant - P_gas) * r_inner / t_liner   [tensile]
    3. (sigma_th_inner_gas, sigma_th_inner_cool) = through_wall_thermal_stress(
           T_gas_side=T_wall_gas, T_cool_side=T_wall_coolant,
           E, alpha evaluated at T_liner_mean)
    4. Gas-side combined state (biaxial):
           sigma_1 = sigma_h_inner + sigma_th_inner_gas   [hoop]
           sigma_2 = sigma_th_inner_gas                   [axial, thermal only]
           sigma_vm_inner = von_mises(sigma_1, sigma_2)
    5. margin_yield_inner  = sigma_y(T_wall_gas) / (fos_yield * sigma_vm_inner) - 1
       margin_ultimate_inner = sigma_u(T_wall_gas) / (fos_ult * sigma_vm_inner) - 1

    OUTER JACKET
    ------------
    6. T_jacket ~ T_wall_coolant (jacket runs near coolant temperature)
       Material properties evaluated at T_wall_coolant.
    7. r_jacket = r_inner + t_liner + h_channel
    8. sigma_h_jacket = P_coolant * r_jacket / t_jacket             [tensile]
    9. Jacket thermal gradient: T_hot_jacket=T_wall_coolant,
       T_cold_jacket=T_wall_coolant (zero gradient, conservative approximation).
       -> sigma_th_jacket_inner = 0 (jacket runs isothermal to first order)
    10. sigma_vm_jacket = von_mises(sigma_h_jacket, 0)  = sigma_h_jacket
    11. margin_yield_jacket  = sigma_y(T_jkt) / (fos_yield * sigma_vm_jacket) - 1
        margin_ultimate_jacket = sigma_u(T_jkt) / (fos_ult * sigma_vm_jacket) - 1

    RIB
    ---
    12. tau_rib = (P_coolant - P_gas) * w_channel / (2 * t_land)
    13. Equivalent von Mises for pure shear: sigma_vm_rib = sqrt(3) * tau_rib
    14. Rib yield margin uses shear yield = sigma_y / sqrt(3)  (von Mises):
        margin_yield_rib = (sigma_y(T_liner) / sqrt(3)) / (fos_yield * tau_rib) - 1

    Sources
    -------
    NASA SP-8087 (1977) §3.3, §4.2, §4.3.
    Cook & Coffey (1973) AIAA 73-1364.
    Esposito & Zappa (1975) NASA CR-2503.

    Parameters
    ----------
    regen_solution : dict
        Output of solve_regen().  Required keys: 'x', 'r', 'T_wall_gas',
        'T_wall_coolant', 'P_coolant'.
    bilayer : BilayerWall
        Bilayer geometry and material description.
    P_gas_arr : ndarray or None
        Local gas pressure [Pa] at each station.  If None, uses
        regen_solution['P_coolant'] * 0 + min(P_coolant) as a conservative
        approximation (P_gas ~ Pc, which is close to chamber pressure).
        Providing P_gas_arr computed isentropically from chamber pressure is
        strongly preferred.
    P_atm : float
        Ambient pressure outside the jacket [Pa].  Default 101325 Pa (sea level).
        Reduces effective jacket hoop slightly.  Set to 0 for vacuum operation.
    fos_yield : float
        Factor of safety on yield (default 1.4).
    fos_ultimate : float
        Factor of safety on ultimate (default 2.0).

    Returns
    -------
    dict with arrays indexed by axial station:
        x                        [m]   axial coordinate
        r_inner                  [m]   gas-wall inner radius
        r_jacket                 [m]   outer radius of coolant channels (= jacket inner)
        t_liner                  [m]   liner thickness
        t_jacket                 [m]   jacket thickness
        T_gas                    [K]   gas-side wall temperature
        T_cool                   [K]   coolant-side wall temperature
        sigma_h_inner            [Pa]  liner hoop (differential P)
        sigma_th_inner_gas       [Pa]  liner thermal stress, gas face (<=0)
        sigma_th_inner_cool      [Pa]  liner thermal stress, cool face (>=0)
        sigma_vm_inner           [Pa]  liner von Mises at gas face
        margin_yield_inner       [-]   liner yield margin
        margin_ultimate_inner    [-]   liner ultimate margin
        sigma_h_jacket           [Pa]  jacket hoop (full P_coolant)
        sigma_vm_jacket          [Pa]  jacket von Mises (= sigma_h_jacket, zero thermal)
        margin_yield_jacket      [-]   jacket yield margin
        margin_ultimate_jacket   [-]   jacket ultimate margin
        tau_rib                  [Pa]  rib peak shear stress
        sigma_vm_rib             [Pa]  sqrt(3)*tau_rib (von Mises equivalent)
        margin_yield_rib         [-]   rib shear yield margin
    """
    x_arr = regen_solution["x"]
    r_arr = regen_solution["r"]
    T_wg = regen_solution["T_wall_gas"]
    T_wc = regen_solution["T_wall_coolant"]
    P_c = regen_solution["P_coolant"]
    n = len(x_arr)

    liner_mat = bilayer.inner_liner.material
    jacket_mat = bilayer.outer_jacket.material
    rib_mat = bilayer.ribs.material
    geom = bilayer.channel_geometry

    # Output arrays
    r_jacket_arr = np.zeros(n)
    t_liner_arr = np.zeros(n)
    t_jacket_arr = np.zeros(n)

    sh_inner = np.zeros(n)
    sth_inner_gas = np.zeros(n)
    sth_inner_cool = np.zeros(n)
    svm_inner = np.zeros(n)
    my_inner = np.zeros(n)
    mu_inner = np.zeros(n)

    sh_jacket = np.zeros(n)
    svm_jacket = np.zeros(n)
    my_jacket = np.zeros(n)
    mu_jacket = np.zeros(n)

    tau_rib_arr = np.zeros(n)
    svm_rib_arr = np.zeros(n)
    my_rib_arr = np.zeros(n)

    _sqrt3 = math.sqrt(3.0)

    for i in range(n):
        x = float(x_arr[i])
        r_in = float(r_arr[i])
        T_g = float(T_wg[i])
        T_c = float(T_wc[i])
        P_cool = float(P_c[i])
        P_gas = float(P_gas_arr[i]) if P_gas_arr is not None else P_cool * 0.0

        t_lin = bilayer.inner_liner.thickness(x)
        t_jkt = bilayer.outer_jacket.thickness(x)
        h_ch = geom.height(x)
        w_ch = geom.width(x)
        t_land = geom.land_thickness(x)

        r_jkt = r_in + t_lin + h_ch   # inner radius of jacket

        t_liner_arr[i] = t_lin
        t_jacket_arr[i] = t_jkt
        r_jacket_arr[i] = r_jkt

        # ---- Inner liner ----
        T_lin_mean = 0.5 * (T_g + T_c)
        E_lin = liner_mat.E(T_lin_mean)
        al_lin = liner_mat.alpha(T_lin_mean)
        nu_lin = liner_mat.nu

        sig_h_in = _liner_hoop(P_cool, P_gas, r_in, t_lin)
        sig_th_g, sig_th_c = through_wall_thermal_stress(
            T_gas_side=T_g, T_cool_side=T_c, E=E_lin, alpha=al_lin, nu=nu_lin
        )

        vm_in = von_mises(sig_h_in + sig_th_g, sig_th_g)

        sy_lin = liner_mat.sigma_y(T_g)
        su_lin = liner_mat.sigma_u(T_g)
        my_in = sy_lin / (fos_yield * vm_in) - 1.0 if vm_in > 0.0 else math.inf
        mu_in = su_lin / (fos_ultimate * vm_in) - 1.0 if vm_in > 0.0 else math.inf

        sh_inner[i] = sig_h_in
        sth_inner_gas[i] = sig_th_g
        sth_inner_cool[i] = sig_th_c
        svm_inner[i] = vm_in
        my_inner[i] = my_in
        mu_inner[i] = mu_in

        # ---- Outer jacket ----
        # Jacket operates near coolant temperature; thermal gradient negligible.
        T_jkt = T_c
        P_jkt_net = P_cool - P_atm   # net outward pressure: coolant in, atmosphere out
        sig_h_jkt = _jacket_hoop(P_jkt_net, r_jkt, t_jkt)
        vm_jkt = abs(sig_h_jkt)   # von Mises for uniaxial = |sigma|

        sy_jkt = jacket_mat.sigma_y(T_jkt)
        su_jkt = jacket_mat.sigma_u(T_jkt)
        my_jkt = sy_jkt / (fos_yield * vm_jkt) - 1.0 if vm_jkt > 0.0 else math.inf
        mu_jkt = su_jkt / (fos_ultimate * vm_jkt) - 1.0 if vm_jkt > 0.0 else math.inf

        sh_jacket[i] = sig_h_jkt
        svm_jacket[i] = vm_jkt
        my_jacket[i] = my_jkt
        mu_jacket[i] = mu_jkt

        # ---- Rib shear ----
        tau = _rib_shear(P_cool, P_gas, w_ch, t_land)
        vm_rib = _sqrt3 * tau   # von Mises equivalent for pure shear

        sy_rib = rib_mat.sigma_y(T_g)   # rib is at gas-side temperature (conservative)
        tau_yield_rib = sy_rib / _sqrt3   # von Mises shear yield
        my_rib = (tau_yield_rib / (fos_yield * tau) - 1.0
                  if tau > 0.0 else math.inf)

        tau_rib_arr[i] = tau
        svm_rib_arr[i] = vm_rib
        my_rib_arr[i] = my_rib

    return {
        "x": x_arr,
        "r_inner": r_arr,
        "r_jacket": r_jacket_arr,
        "t_liner": t_liner_arr,
        "t_jacket": t_jacket_arr,
        "T_gas": T_wg,
        "T_cool": T_wc,
        "sigma_h_inner": sh_inner,
        "sigma_th_inner_gas": sth_inner_gas,
        "sigma_th_inner_cool": sth_inner_cool,
        "sigma_vm_inner": svm_inner,
        "margin_yield_inner": my_inner,
        "margin_ultimate_inner": mu_inner,
        "sigma_h_jacket": sh_jacket,
        "sigma_vm_jacket": svm_jacket,
        "margin_yield_jacket": my_jacket,
        "margin_ultimate_jacket": mu_jacket,
        "tau_rib": tau_rib_arr,
        "sigma_vm_rib": svm_rib_arr,
        "margin_yield_rib": my_rib_arr,
    }
