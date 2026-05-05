"""
liner_stress.py
===============

Analytical stress calculations for the inner hot-wall liner of a
regeneratively-cooled rocket engine chamber.

Stress components
-----------------
1. Mechanical hoop stress (thin-shell, from coolant pressure).
2. Through-wall thermal stress (linear gradient, biaxial plane-stress).
3. Combined von Mises stress at gas-side and coolant-side surfaces.

Scope and limitations
---------------------
- Thin-wall approximation: valid when t_wall / r_inner < 0.1.
  For this engine: t_wall = 0.5 mm, r_throat = 11.16 mm → t/r ≈ 0.045. OK.
- Thermal stress uses a linear temperature gradient through the wall.
  The actual gradient through a wall with coolant channels is nonlinear;
  the linear model is an overestimate of the peak thermal stress and is
  therefore conservative.
- Biaxial thermal stress: the wall is assumed fully constrained in both
  hoop and axial directions (plane-strain approximation), which is standard
  for thin-shell rocket liners (NASA SP-8087 Sec. 4.3).
- Rib/land stress between channels is OUT OF SCOPE (flagged for future work).
- Buckling, creep, manifold stress concentrations: OUT OF SCOPE.
- Axial mechanical stress from chamber end loads: OUT OF SCOPE (conservative:
  setting axial mechanical stress to zero means only thermal contribution is
  included axially, which may underestimate combined axial stress).

Pressure convention — liner vs jacket
--------------------------------------
The inner liner is the thin copper wall between the hot gas (inside) and the
coolant channels (outside).

  P_gas    -- hot gas, acts on the gas-side surface (outward on the liner)
  P_coolant -- coolant, acts on the coolant-channel side (inward on the liner)

For a thin cylindrical shell with inner pressure p_i and outer pressure p_o:
    sigma_hoop = (p_i - p_o) * r / t

In this engine P_coolant > P_gas everywhere, so the net hoop stress in the
liner is COMPRESSIVE:
    sigma_hoop = (P_gas - P_coolant) * r / t  (<= 0)

The outer structural jacket (not modelled here) carries the full coolant
pressure in tension.  The liner sees only the differential pressure.

liner_stress_profile() accepts an optional P_gas_arr parameter.  If provided,
it uses (P_gas - P_coolant) for the net hoop stress.  If omitted, it defaults
to using P_coolant alone with a positive (tensile) sign — which is conservative
for the von Mises magnitude (|P_coolant * r / t| is similar to
|(P_gas - P_coolant) * r / t| when P_coolant >> P_gas) but gives the WRONG
sign (compressive, not tensile) and may flag the wrong failure mode.

Source: NASA SP-8087 (1977) Sec. 4.2 discusses the differential pressure across
the liner; the outer jacket hoop load is treated separately.

Sources
-------
Thin-shell hoop stress:
    Timoshenko, S. and Goodier, J.N. (1951). "Theory of Elasticity,"
    2nd ed., McGraw-Hill. §131 (reduces to sigma = p*r/t for t/r -> 0).
    NASA SP-8087 (1977) Sec. 4.2.

Through-wall thermal stress (linear gradient, biaxial):
    Timoshenko, S. and Goodier, J.N. (1951). §143 "Thermal Stresses in a
    Plate." Eq. (b): sigma = -E*alpha/(1-nu) * [T(z) - T_mean].
    Boley, B.A. and Weiner, J.H. (1960). "Theory of Thermal Stresses,"
    Wiley. Ch. 3.
    NASA SP-8087 (1977) Sec. 4.3 -- applies the linear-gradient result to
    regeneratively-cooled chamber liners.

Von Mises criterion:
    von Mises, R. (1913). "Mechanik der festen Koerper im plastisch-deformablen
    Zustand." Nachrichten von der Gesellschaft der Wissenschaften zu
    Goettingen, pp. 582-592.
    Timoshenko, S. and Goodier, J.N. (1951). §151.

Safety margin convention:
    M = (sigma_allow / (FoS * sigma_actual)) - 1
    M >= 0  passes;  M < 0  fails.
    FoS_yield = 1.4 (general aerospace practice, NASA-STD-5001).
    FoS_ultimate = 2.0 (general aerospace practice, NASA-STD-5001).
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from physics.regen.coolant import CoolingChannelGeometry
    from physics.structural.materials import StructuralMaterial


# ---------------------------------------------------------------------------
# Fundamental stress functions
# ---------------------------------------------------------------------------


def hoop_stress(*, P_internal: float, r_inner: float, t_wall: float) -> float:
    """
    Thin-shell hoop stress [Pa].

    sigma_h = P_internal * r_inner / t_wall

    Source: Timoshenko & Goodier (1951), §131; NASA SP-8087 (1977) Sec. 4.2.
    Valid when t_wall / r_inner < 0.1 (thin-wall limit of the Lame solution).

    Parameters
    ----------
    P_internal : float
        Net internal pressure [Pa].  For the inner liner, use P_coolant
        (conservative; actual net = P_coolant - P_gas reduces the stress).
    r_inner : float
        Inner radius at the station of interest [m].
    t_wall : float
        Gas-side wall thickness [m].

    Returns
    -------
    float
        Hoop stress [Pa], tension positive.
    """
    if t_wall <= 0.0:
        raise ValueError(f"t_wall must be > 0, got {t_wall}")
    if r_inner <= 0.0:
        raise ValueError(f"r_inner must be > 0, got {r_inner}")
    t_over_r = t_wall / r_inner
    if t_over_r > 0.1:
        warnings.warn(
            f"t/r = {t_over_r:.3f} > 0.1: thin-wall approximation may be inaccurate.",
            RuntimeWarning,
            stacklevel=2,
        )
    return P_internal * r_inner / t_wall


def through_wall_thermal_stress(
    *,
    T_gas_side: float,
    T_cool_side: float,
    E: float,
    alpha: float,
    nu: float,
) -> tuple[float, float]:
    """
    Thermal stress at the gas-side and coolant-side surfaces for a wall with
    a linear temperature gradient.

    For a plate (or thin-shell segment) with T(z) varying linearly from
    T_gas_side (z = 0) to T_cool_side (z = t_wall), the thermal stress at
    position z under biaxial plane-stress constraint is:

        sigma_th(z) = -E * alpha / (1 - nu) * [T(z) - T_mean]

    where T_mean = (T_gas_side + T_cool_side) / 2.

    At the gas-side surface (z = 0, hot face):
        sigma_th_gas  = -E * alpha * (T_gas_side - T_mean) / (1 - nu)
                      = -E * alpha * dT / [2 * (1 - nu)]    [compressive]

    At the coolant-side surface (z = t_wall, cool face):
        sigma_th_cool = +E * alpha * dT / [2 * (1 - nu)]    [tensile]

    where dT = T_gas_side - T_cool_side >= 0.

    Sign convention: tension positive, compression negative.

    Physical interpretation:
        Hot face wants to expand relative to the cooler bulk -> constrained
        -> compressive thermal stress.
        Cool face wants to contract relative to the hotter bulk -> constrained
        -> tensile thermal stress.

    Source: Timoshenko & Goodier (1951), §143, Eq. (b).
            NASA SP-8087 (1977) Sec. 4.3.
            Boley & Weiner (1960), Ch. 3.

    Parameters
    ----------
    T_gas_side : float   Temperature at the gas-side surface [K].
    T_cool_side : float  Temperature at the coolant-side surface [K].
    E : float            Young's modulus at T_mean [Pa].
    alpha : float        Thermal expansion coefficient at T_mean [1/K].
    nu : float           Poisson's ratio [-].

    Returns
    -------
    tuple[float, float]
        (sigma_th_gas [Pa], sigma_th_cool [Pa])
        sigma_th_gas  <= 0  (compressive for T_gas > T_cool)
        sigma_th_cool >= 0  (tensile for T_gas > T_cool)
    """
    dT = T_gas_side - T_cool_side
    amplitude = E * alpha * dT / (2.0 * (1.0 - nu))
    sigma_th_gas = -amplitude
    sigma_th_cool = +amplitude
    return sigma_th_gas, sigma_th_cool


def von_mises(sigma_1: float, sigma_2: float, sigma_3: float = 0.0) -> float:
    """
    Von Mises equivalent (effective) stress [Pa].

    sigma_vm = sqrt(0.5 * [(sigma_1 - sigma_2)^2
                          + (sigma_2 - sigma_3)^2
                          + (sigma_3 - sigma_1)^2])

    Source: von Mises (1913); Timoshenko & Goodier (1951) §151.

    Parameters
    ----------
    sigma_1, sigma_2, sigma_3 : float
        Principal stresses [Pa].  sigma_3 defaults to 0 (plane stress).

    Returns
    -------
    float
        Von Mises stress [Pa], always non-negative.
    """
    return math.sqrt(
        0.5 * (
            (sigma_1 - sigma_2) ** 2
            + (sigma_2 - sigma_3) ** 2
            + (sigma_3 - sigma_1) ** 2
        )
    )


# ---------------------------------------------------------------------------
# Full axial profile
# ---------------------------------------------------------------------------

# Default factors of safety (NASA-STD-5001 general aerospace practice)
_FOS_YIELD = 1.4
_FOS_ULTIMATE = 2.0


def liner_stress_profile(
    *,
    regen_solution: dict,
    geometry: "CoolingChannelGeometry",
    material: "StructuralMaterial",
    P_gas_arr: "np.ndarray | None" = None,
    fos_yield: float = _FOS_YIELD,
    fos_ultimate: float = _FOS_ULTIMATE,
) -> dict:
    """
    Compute the axial stress profile for the inner liner using regen solver output.

    At each axial station:
      1. T_mean = (T_wall_gas + T_wall_coolant) / 2  -- bulk wall temperature
         for material property evaluation.
      2. Net hoop pressure:
           If P_gas_arr is provided:
               P_net = P_gas_arr[i] - P_coolant[i]   (negative when P_coolant > P_gas)
           else:
               P_net = P_coolant[i]                   (conservative tensile approximation)
         sigma_hoop = P_net * r / t_wall
         (negative = compressive in the liner, which is the physical case when
          the outer jacket carries the structural load)
      3. (sigma_th_gas, sigma_th_cool) = through_wall_thermal_stress(...)
         Properties evaluated at T_mean.
      4. Stress state at the GAS-SIDE surface (biaxial):
            sigma_hoop_total = sigma_hoop + sigma_th_gas   [hoop direction]
            sigma_axial      = sigma_th_gas                [axial direction, thermal only]
            sigma_3          = 0                           [through-thickness, thin wall]
         -> sigma_vm_gas = von_mises(sigma_hoop_total, sigma_axial)
      5. Stress state at the COOLANT-SIDE surface:
            sigma_hoop_total = sigma_hoop + sigma_th_cool
            sigma_axial      = sigma_th_cool
         -> sigma_vm_cool = von_mises(sigma_hoop_total, sigma_axial)
      6. Margins (evaluated at gas-side, worst for material):
            margin_yield    = sigma_y(T_gas) / (fos_yield    * sigma_vm_gas) - 1
            margin_ultimate = sigma_u(T_gas) / (fos_ultimate * sigma_vm_gas) - 1

    Negative margins indicate the design does not meet the factor-of-safety
    requirement.  For regen liners, thermal stress typically exceeds sigma_y,
    meaning the liner operates in the plastic range -- the correct failure
    criterion is then Manson-Coffin LCF (see fatigue.py), not static yield.
    Negative margin_yield should therefore be interpreted as "LCF governs,
    not static yield."

    Sources
    -------
    Stress combination: NASA SP-8087 (1977) Sec. 4.3.
    Margin convention: NASA-STD-5001B (2014) §4.2.

    Parameters
    ----------
    regen_solution : dict
        Output dictionary from physics.regen.solver.solve_regen().
        Required keys: 'x', 'r', 'T_wall_gas', 'T_wall_coolant', 'P_coolant'.
    geometry : CoolingChannelGeometry
        Cooling channel geometry (provides wall_thickness(x)).
    material : StructuralMaterial
        Structural material model (E, alpha, nu, sigma_y, sigma_u).
    P_gas_arr : ndarray or None
        Local gas pressure [Pa] at each axial station.  When provided, the net
        liner hoop stress uses (P_gas - P_coolant), correctly reflecting that
        the outer jacket carries the structural load.  When None, uses
        P_coolant alone (same magnitude, but wrong sign -- conservative for
        the VM magnitude but mis-identifies the failure mode at large radii).
    fos_yield : float
        Factor of safety applied to yield strength. Default 1.4.
    fos_ultimate : float
        Factor of safety applied to ultimate strength. Default 2.0.

    Returns
    -------
    dict with keys:
        x                  ndarray [m]   axial stations (inlet to exit)
        T_gas              ndarray [K]   gas-side wall temperature
        T_cool             ndarray [K]   coolant-side wall temperature
        t_wall             ndarray [m]   hot-wall thickness at each station
        sigma_hoop         ndarray [Pa]  net mechanical hoop stress in liner
        sigma_thermal_gas  ndarray [Pa]  thermal stress at gas surface (<=0)
        sigma_thermal_cool ndarray [Pa]  thermal stress at cool surface (>=0)
        sigma_vm_gas       ndarray [Pa]  von Mises at gas-side surface
        sigma_vm_cool      ndarray [Pa]  von Mises at cool-side surface
        margin_yield       ndarray [-]   yield safety margin at gas side
        margin_ultimate    ndarray [-]   ultimate safety margin at gas side
    """
    x_arr = regen_solution["x"]
    r_arr = regen_solution["r"]
    T_wg = regen_solution["T_wall_gas"]
    T_wc = regen_solution["T_wall_coolant"]
    P_c = regen_solution["P_coolant"]
    n = len(x_arr)

    sigma_hoop_arr = np.zeros(n)
    sigma_th_gas_arr = np.zeros(n)
    sigma_th_cool_arr = np.zeros(n)
    sigma_vm_gas_arr = np.zeros(n)
    sigma_vm_cool_arr = np.zeros(n)
    margin_yield_arr = np.zeros(n)
    margin_ultimate_arr = np.zeros(n)
    t_wall_arr = np.zeros(n)

    for i in range(n):
        x = float(x_arr[i])
        r = float(r_arr[i])
        T_g = float(T_wg[i])
        T_c = float(T_wc[i])
        P_cool = float(P_c[i])
        # Net pressure on the liner: positive = tensile hoop, negative = compressive
        if P_gas_arr is not None:
            P = float(P_gas_arr[i]) - P_cool
        else:
            P = P_cool

        t_w = geometry.wall_thickness(x)
        t_wall_arr[i] = t_w

        # Mean wall temperature for material properties
        T_mean = 0.5 * (T_g + T_c)

        E = material.E(T_mean)
        al = material.alpha(T_mean)
        nu = material.nu

        # Mechanical hoop stress
        sig_h = hoop_stress(P_internal=P, r_inner=r, t_wall=t_w)

        # Thermal stresses
        sig_th_g, sig_th_c = through_wall_thermal_stress(
            T_gas_side=T_g,
            T_cool_side=T_c,
            E=E,
            alpha=al,
            nu=nu,
        )

        # Combined stress state -- gas side
        sig1_g = sig_h + sig_th_g   # hoop: mechanical + thermal
        sig2_g = sig_th_g            # axial: thermal only
        vm_g = von_mises(sig1_g, sig2_g)

        # Combined stress state -- cool side
        sig1_c = sig_h + sig_th_c
        sig2_c = sig_th_c
        vm_c = von_mises(sig1_c, sig2_c)

        # Allowables at the gas-side temperature (worst degradation)
        sy = material.sigma_y(T_g)
        su = material.sigma_u(T_g)

        m_yield = sy / (fos_yield * vm_g) - 1.0 if vm_g > 0.0 else float("inf")
        m_ult = su / (fos_ultimate * vm_g) - 1.0 if vm_g > 0.0 else float("inf")

        sigma_hoop_arr[i] = sig_h
        sigma_th_gas_arr[i] = sig_th_g
        sigma_th_cool_arr[i] = sig_th_c
        sigma_vm_gas_arr[i] = vm_g
        sigma_vm_cool_arr[i] = vm_c
        margin_yield_arr[i] = m_yield
        margin_ultimate_arr[i] = m_ult

    return {
        "x": x_arr,
        "T_gas": T_wg,
        "T_cool": T_wc,
        "t_wall": t_wall_arr,
        "sigma_hoop": sigma_hoop_arr,
        "sigma_thermal_gas": sigma_th_gas_arr,
        "sigma_thermal_cool": sigma_th_cool_arr,
        "sigma_vm_gas": sigma_vm_gas_arr,
        "sigma_vm_cool": sigma_vm_cool_arr,
        "margin_yield": margin_yield_arr,
        "margin_ultimate": margin_ultimate_arr,
    }
