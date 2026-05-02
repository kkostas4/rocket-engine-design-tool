"""
units.py
========

Single source of truth for unit conversions. Internal physics is strictly SI;
this module converts at I/O boundaries only.

Never define conversion factors anywhere else in the codebase.
"""

from __future__ import annotations

# --- Length ---------------------------------------------------------------
INCH_TO_M = 0.0254
FT_TO_M = 0.3048
MM_TO_M = 1e-3

# --- Pressure -------------------------------------------------------------
PSI_TO_PA = 6_894.7572931783
BAR_TO_PA = 100_000.0
ATM_TO_PA = 101_325.0
MPA_TO_PA = 1e6

# --- Force ----------------------------------------------------------------
LBF_TO_N = 4.4482216152605

# --- Mass / mass flow -----------------------------------------------------
LBM_TO_KG = 0.45359237
LBM_PER_FT3_TO_KG_PER_M3 = 16.018463373960140

# --- Temperature ----------------------------------------------------------
def f_to_k(t_f: float) -> float:
    """Convert Fahrenheit to Kelvin."""
    return (t_f - 32.0) * 5.0 / 9.0 + 273.15


def r_to_k(t_r: float) -> float:
    """Convert Rankine to Kelvin."""
    return t_r * 5.0 / 9.0


def c_to_k(t_c: float) -> float:
    """Convert Celsius to Kelvin."""
    return t_c + 273.15


def k_to_f(t_k: float) -> float:
    return (t_k - 273.15) * 9.0 / 5.0 + 32.0


def k_to_r(t_k: float) -> float:
    return t_k * 9.0 / 5.0


def k_to_c(t_k: float) -> float:
    return t_k - 273.15


# --- Velocity / Isp -------------------------------------------------------
G0 = 9.80665  # m/s^2 — standard gravity, definition of Isp


def isp_to_c_eff(isp_s: float) -> float:
    """Convert specific impulse [s] to effective exhaust velocity [m/s]."""
    return isp_s * G0


def c_eff_to_isp(c_eff_m_s: float) -> float:
    """Convert effective exhaust velocity [m/s] to Isp [s]."""
    return c_eff_m_s / G0
