"""
materials.py
============

Temperature-dependent structural material properties for the regeneratively-
cooled chamber liner.

Provides:
  - StructuralMaterial dataclass (E, alpha, nu, sigma_y, sigma_u, Manson-Coffin)
  - GRCOP84_PROXY  -- GRCop-84 data used as a proxy for GRCop-42; flagged
  - INCONEL718_STRUCT -- Inconel 718 structural properties

FLAG — GRCop-42 substitution
-----------------------------
GRCop-42 (Cu-4wt%Cr-2wt%Nb) temperature-dependent MECHANICAL properties are
not widely published as of 2024.  GRCop-84 (Cu-8wt%Cr-4wt%Nb) data from
Ellis (2005) is used as a proxy.  GRCop-42 is expected to have slightly lower
strength (~10-15 %) due to its lower Nb content, so using GRCop-84 data
is NON-CONSERVATIVE for strength.  Apply an additional knockdown of 15 % to
sigma_y and sigma_u if a formal design review is required.

Sources
-------
GRCop-84 mechanical properties (proxy for GRCop-42):
    Ellis, D.L. (2005). "GRCop-84: A High-Temperature Copper Alloy for
    High-Heat-Flux Applications." NASA/TM-2005-213566. Table 2 (wrought,
    hot-pressed and extruded).

GRCop-84 thermal expansion coefficient:
    Ellis, D.L. (2005). NASA/TM-2005-213566. Table 3.
    NOTE: alpha treated as constant 17.0e-6 /K; variation over 295-922 K
    is < 5 % per the published data.

Inconel 718 mechanical properties:
    Special Metals Corporation (2007). "INCONEL Alloy 718."
    Publication No. SMC-045. Tables: Tensile Properties and Physical Properties.

Manson-Coffin fatigue parameters for GRCop-84 / copper-alloy liners:
    Quentmeyer, R.J. (1977). "Experimental Fatigue Life Investigation of
    Cylindrical Thrust Chambers." NASA TM-X-73665. Sec. 3.
    NOTE: The specific sigma_f', b, epsilon_f', c values here are
    conservative estimates consistent with the life ranges reported by
    Quentmeyer.  They have NOT been individually curve-fit to test data.
    Flag: # TODO(unsourced-fatigue-params) -- obtain curve-fit parameters
    from coupon LCF tests at operating temperature before flight.

Manson-Coffin fatigue parameters for Inconel 718:
    Meran, C., Kovan, V. (2007). "Fatigue Behavior of Inconel 718 Superalloy."
    Scientific and Technical Journal of Information Technologies,
    Mechanics and Optics -- approximate values consistent with the alloy class.
    FLAG: # NOTE(conservative-estimate) -- use alloy-specific LCF data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.interpolate import interp1d  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class StructuralMaterial:
    """
    Isotropic structural material with temperature-dependent elastic and
    strength properties plus Manson-Coffin LCF parameters.

    Parameters
    ----------
    name : str
    E_func : Callable[[float], float]
        Young's modulus E(T) [Pa].
    alpha_func : Callable[[float], float]
        Coefficient of thermal expansion alpha(T) [1/K].
    nu : float
        Poisson's ratio (treated as constant).
    sigma_y_func : Callable[[float], float]
        0.2 % offset yield strength sigma_y(T) [Pa].
    sigma_u_func : Callable[[float], float]
        Ultimate tensile strength sigma_u(T) [Pa].
    sigma_f_prime : float
        Manson-Coffin fatigue strength coefficient [Pa].
    b_fatigue : float
        Manson-Coffin fatigue strength exponent (negative, typically -0.05 to -0.12).
    epsilon_f_prime : float
        Manson-Coffin fatigue ductility coefficient [-].
    c_fatigue : float
        Manson-Coffin fatigue ductility exponent (negative, typically -0.5 to -0.7).
    """

    name: str
    E_func: Callable[[float], float]
    alpha_func: Callable[[float], float]
    nu: float
    sigma_y_func: Callable[[float], float]
    sigma_u_func: Callable[[float], float]
    sigma_f_prime: float
    b_fatigue: float
    epsilon_f_prime: float
    c_fatigue: float

    def E(self, T: float) -> float:
        """Young's modulus at temperature T [K] -> [Pa]."""
        return self.E_func(T)

    def alpha(self, T: float) -> float:
        """Thermal expansion coefficient at temperature T [K] -> [1/K]."""
        return self.alpha_func(T)

    def sigma_y(self, T: float) -> float:
        """0.2 % offset yield strength at temperature T [K] -> [Pa]."""
        return self.sigma_y_func(T)

    def sigma_u(self, T: float) -> float:
        """Ultimate tensile strength at temperature T [K] -> [Pa]."""
        return self.sigma_u_func(T)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _interp(T_pts: list[float], val_pts: list[float]) -> Callable[[float], float]:
    """Clamped linear interpolant: clamps at boundary values outside the range."""
    T_arr = np.array(T_pts, dtype=float)
    v_arr = np.array(val_pts, dtype=float)
    f = interp1d(T_arr, v_arr, kind="linear", bounds_error=False,
                 fill_value=(float(v_arr[0]), float(v_arr[-1])))
    return lambda T: float(f(float(T)))


# ---------------------------------------------------------------------------
# GRCop-84 (proxy for GRCop-42)
# ---------------------------------------------------------------------------
#
# FLAG: Using GRCop-84 (Cu-8Cr-4Nb) as a proxy for GRCop-42 (Cu-4Cr-2Nb).
# Source: Ellis, D.L. (2005). NASA/TM-2005-213566. Table 2.
# The tabulated values below are read from Table 2 of the TM.
# Temperatures converted from °F to K.

_GR84_T_K = [295.0, 589.0, 700.0, 811.0, 922.0]  # K

# Young's modulus [Pa] -- Ellis (2005) Table 2
_GR84_E_PA = [117.0e9, 109.0e9, 106.0e9, 99.0e9, 91.0e9]

# Coefficient of thermal expansion [1/K] -- Ellis (2005) Table 3
# Treated as constant; variation < 5 % over operating range.
_GR84_ALPHA_CONST = 17.0e-6  # /K

# 0.2 % offset yield strength [Pa] -- Ellis (2005) Table 2, wrought form
_GR84_SY_PA = [266.0e6, 216.0e6, 194.0e6, 168.0e6, 139.0e6]

# Ultimate tensile strength [Pa] -- Ellis (2005) Table 2, wrought form
_GR84_SU_PA = [388.0e6, 310.0e6, 261.0e6, 220.0e6, 179.0e6]

# Poisson's ratio for copper alloys -- copper alloy typical value
# Source: Metals Handbook (ASM International); not in Ellis 2005.
# NOTE(conservative-estimate): 0.34 is standard for Cu alloys.
_GR84_NU = 0.34

# Manson-Coffin parameters for Cu-Cr-Nb alloy liners
# Source: Quentmeyer (1977) NASA TM-X-73665 -- conservative estimates
# consistent with reported life ranges of 500-3000 cycles for copper alloy
# chambers at high strain amplitudes (Sec. 3 and Fig. 6).
# TODO(unsourced-fatigue-params): replace with curve-fit to coupon LCF data.
_GR84_SIGMA_F_PRIME = 700.0e6   # Pa  (fatigue strength coefficient)
_GR84_B_FATIGUE = -0.12          # fatigue strength exponent
_GR84_EPS_F_PRIME = 0.30         # fatigue ductility coefficient
_GR84_C_FATIGUE = -0.56          # fatigue ductility exponent

GRCOP84_PROXY = StructuralMaterial(
    name="GRCop-84 (proxy for GRCop-42; Ellis 2005 NASA/TM-2005-213566)",
    E_func=_interp(_GR84_T_K, _GR84_E_PA),
    alpha_func=lambda T: _GR84_ALPHA_CONST,
    nu=_GR84_NU,
    sigma_y_func=_interp(_GR84_T_K, _GR84_SY_PA),
    sigma_u_func=_interp(_GR84_T_K, _GR84_SU_PA),
    sigma_f_prime=_GR84_SIGMA_F_PRIME,
    b_fatigue=_GR84_B_FATIGUE,
    epsilon_f_prime=_GR84_EPS_F_PRIME,
    c_fatigue=_GR84_C_FATIGUE,
)

# ---------------------------------------------------------------------------
# Inconel 718
# ---------------------------------------------------------------------------
#
# Source: Special Metals Corporation (2007). "INCONEL Alloy 718."
# Publication No. SMC-045. Tensile and physical properties tables.

_IN718_T_K = [294.0, 422.0, 533.0, 644.0, 755.0, 866.0, 977.0]  # K

# Young's modulus [Pa] -- SMC-045
_IN718_E_PA = [200.0e9, 196.0e9, 193.0e9, 189.0e9, 182.0e9, 170.0e9, 162.0e9]

# CTE [1/K] -- SMC-045 (mean CTE from RT to given T)
_IN718_ALPHA_T = [294.0, 644.0, 866.0, 977.0]
_IN718_ALPHA_V = [13.0e-6, 14.0e-6, 14.8e-6, 15.2e-6]

# 0.2 % offset yield strength [Pa] -- SMC-045 (Agemat condition)
_IN718_SY_PA = [1034.0e6, 1013.0e6, 985.0e6, 957.0e6, 929.0e6, 874.0e6, 792.0e6]

# Ultimate tensile strength [Pa] -- SMC-045
_IN718_SU_PA = [1241.0e6, 1227.0e6, 1193.0e6, 1165.0e6, 1110.0e6, 1041.0e6, 979.0e6]

# Poisson's ratio -- SMC-045
_IN718_NU = 0.29

# Manson-Coffin parameters for Inconel 718
# NOTE(conservative-estimate): estimates from superalloy fatigue literature.
# TODO(unsourced-fatigue-params): replace with alloy-heat-treatment-specific data.
_IN718_SIGMA_F_PRIME = 1900.0e6  # Pa
_IN718_B_FATIGUE = -0.12
_IN718_EPS_F_PRIME = 0.20
_IN718_C_FATIGUE = -0.65

INCONEL718_STRUCT = StructuralMaterial(
    name="Inconel 718 (Special Metals SMC-045)",
    E_func=_interp(_IN718_T_K, _IN718_E_PA),
    alpha_func=_interp(_IN718_ALPHA_T, _IN718_ALPHA_V),
    nu=_IN718_NU,
    sigma_y_func=_interp(_IN718_T_K, _IN718_SY_PA),
    sigma_u_func=_interp(_IN718_T_K, _IN718_SU_PA),
    sigma_f_prime=_IN718_SIGMA_F_PRIME,
    b_fatigue=_IN718_B_FATIGUE,
    epsilon_f_prime=_IN718_EPS_F_PRIME,
    c_fatigue=_IN718_C_FATIGUE,
)
