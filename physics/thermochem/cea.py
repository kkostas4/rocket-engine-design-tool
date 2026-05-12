"""
cea.py
======

CSV-backed interpolator for NASA CEA equilibrium combustion results.

This module provides a lightweight ``CEA`` class that interpolates over a
pre-computed sweep table rather than calling rocketcea at runtime.  A live
CEA backend (via rocketcea) will be added once a Fortran build environment is
available; until then this module is the authoritative source for combustion
state at the reference operating point.

Reference data
--------------
``physics/thermochem/reference_data/cea_sweep_pc1000.csv``

    Generated with NASA CEA via rocketcea at:
        Pc  = 1,000 psia (6.8948 MPa)
        ε   = 68.027  (vacuum-optimised expansion)
        O/F = 1.0 … 4.0, step 0.2
        Propellants: LCH4 (methane) / LOX

    Columns: of_ratio, isp_vac_s, cstar_ft_s, tc_R, tc_K

Limitations
-----------
* Only valid at Pc = 1,000 psia and ε = 68.027 (the CSV reference conditions).
  Calling with any other values raises ``NotImplementedError``.
* ``isp_sl`` is not in the reference data and always raises ``NotImplementedError``.
  The delivered sea-level Isp at ε = 10.605 is 318.44 s (see CLAUDE.md).
* ``gamma`` and ``mw`` are not in the reference data and always raise
  ``NotImplementedError``.

Source
------
NASA CEA program (Chemical Equilibrium with Applications), originally by
Gordon & McBride:
    Gordon, S. and McBride, B.J. (1994). "Computer Program for Calculation of
    Complex Chemical Equilibrium Compositions and Applications."
    NASA RP-1311.

rocketcea Python wrapper (version unknown at CSV generation time):
    https://rocketcea.readthedocs.io/
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import interp1d  # type: ignore[import-untyped]

from physics.util.units import PSI_TO_PA, FT_TO_M

if TYPE_CHECKING:
    from noyron.engine import PropellantPair

# ---------------------------------------------------------------------------
# Reference conditions for the pre-computed sweep
# ---------------------------------------------------------------------------
_REF_PC_PSIA: float = 1_000.0
_REF_PC_PA: float = _REF_PC_PSIA * PSI_TO_PA
_REF_EPS: float = 68.027
_TOL_FRAC: float = 0.001  # 0.1% — window within which Pc / eps are "equal"

_CSV_PATH = Path(__file__).parent / "reference_data" / "cea_sweep_pc1000.csv"


def _load_csv() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the reference CSV and return (of, isp_vac_s, cstar_m_s, tc_K)."""
    of_list: list[float] = []
    isp_list: list[float] = []
    cstar_list: list[float] = []
    tc_list: list[float] = []

    with _CSV_PATH.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            of_list.append(float(row["of_ratio"]))
            isp_list.append(float(row["isp_vac_s"]))
            cstar_list.append(float(row["cstar_ft_s"]) * FT_TO_M)
            tc_list.append(float(row["tc_K"]))

    return (
        np.asarray(of_list),
        np.asarray(isp_list),
        np.asarray(cstar_list),
        np.asarray(tc_list),
    )


class CEA:
    """
    Interpolator for NASA CEA equilibrium combustion results.

    Wraps the pre-computed CSV sweep at Pc = 1,000 psia, ε = 68.027.
    All inputs are SI (Pa for pressure); all outputs are SI.

    Parameters
    ----------
    (none — loads the reference CSV on construction)

    Notes
    -----
    Raises ``NotImplementedError`` for any (Pc, eps) that differs from the
    reference conditions by more than 0.1 %.  A rocketcea backend for
    arbitrary conditions is planned once a Fortran build is available.
    """

    def __init__(self) -> None:
        of, isp_vac, cstar, tc = _load_csv()
        kw = dict(kind="linear", bounds_error=True, assume_sorted=True)
        self._isp_vac_interp: interp1d = interp1d(of, isp_vac, **kw)
        self._cstar_interp: interp1d = interp1d(of, cstar, **kw)
        self._tc_interp: interp1d = interp1d(of, tc, **kw)
        self._of_range: tuple[float, float] = (float(of[0]), float(of[-1]))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_ref_conditions(self, Pc_pa: float, eps: float) -> None:
        """Raise if (Pc, eps) are outside the reference CSV conditions."""
        if abs(Pc_pa - _REF_PC_PA) / _REF_PC_PA > _TOL_FRAC:
            raise NotImplementedError(
                f"CEA CSV wrapper only supports Pc = {_REF_PC_PSIA} psia "
                f"({_REF_PC_PA:.0f} Pa); got {Pc_pa:.0f} Pa.  "
                "A rocketcea backend for arbitrary Pc is not yet implemented."
            )
        if abs(eps - _REF_EPS) / _REF_EPS > _TOL_FRAC:
            raise NotImplementedError(
                f"CEA CSV wrapper only supports ε = {_REF_EPS}; got {eps}.  "
                "A rocketcea backend for arbitrary ε is not yet implemented."
            )

    def _require_ref_pc(self, Pc_pa: float) -> None:
        if abs(Pc_pa - _REF_PC_PA) / _REF_PC_PA > _TOL_FRAC:
            raise NotImplementedError(
                f"CEA CSV wrapper only supports Pc = {_REF_PC_PSIA} psia; got "
                f"{Pc_pa / PSI_TO_PA:.1f} psia."
            )

    def _check_mr_range(self, MR: float) -> None:
        lo, hi = self._of_range
        if not (lo <= MR <= hi):
            raise ValueError(
                f"O/F ratio {MR:.3f} is outside the reference sweep range "
                f"[{lo}, {hi}]."
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def isp_vac(self, Pc_pa: float, MR: float, eps: float) -> float:
        """
        Vacuum specific impulse [s] at (Pc, O/F, ε).

        Source: interpolated from NASA CEA sweep via cea_sweep_pc1000.csv.
        Validity: Pc = 1,000 psia, ε = 68.027, O/F ∈ [1.0, 4.0].

        Parameters
        ----------
        Pc_pa : float
            Chamber pressure [Pa].
        MR : float
            Oxidiser-to-fuel mass ratio (O/F).
        eps : float
            Nozzle expansion area ratio Ae/At.

        Returns
        -------
        float
            Isp_vac [s].
        """
        self._require_ref_conditions(Pc_pa, eps)
        self._check_mr_range(MR)
        return float(self._isp_vac_interp(MR))

    def cstar(self, Pc_pa: float, MR: float) -> float:
        """
        Characteristic velocity c* [m/s] at (Pc, O/F).

        Source: interpolated from NASA CEA sweep via cea_sweep_pc1000.csv
                (column cstar_ft_s converted to m/s using FT_TO_M = 0.3048).
        Validity: Pc = 1,000 psia, O/F ∈ [1.0, 4.0].

        Parameters
        ----------
        Pc_pa : float
            Chamber pressure [Pa].
        MR : float
            Oxidiser-to-fuel mass ratio (O/F).

        Returns
        -------
        float
            c* [m/s].
        """
        self._require_ref_pc(Pc_pa)
        self._check_mr_range(MR)
        return float(self._cstar_interp(MR))

    def tc(self, Pc_pa: float, MR: float) -> float:
        """
        Adiabatic flame (chamber) temperature Tc [K] at (Pc, O/F).

        Source: interpolated from NASA CEA sweep via cea_sweep_pc1000.csv.
        Validity: Pc = 1,000 psia, O/F ∈ [1.0, 4.0].

        Parameters
        ----------
        Pc_pa : float
            Chamber pressure [Pa].
        MR : float
            Oxidiser-to-fuel mass ratio (O/F).

        Returns
        -------
        float
            Tc [K].
        """
        self._require_ref_pc(Pc_pa)
        self._check_mr_range(MR)
        return float(self._tc_interp(MR))

    def isp_sl(self, Pc_pa: float, MR: float, eps: float) -> float:
        """Not implemented — sea-level Isp requires a different CEA run."""
        raise NotImplementedError(
            "isp_sl is not available in the CSV sweep.  "
            "The delivered SL Isp at ε = 10.605 is 318.44 s (see CLAUDE.md).  "
            "A full CEA backend is needed for arbitrary conditions."
        )

    def gamma(self, Pc_pa: float, MR: float) -> float:
        """Not implemented — gamma not stored in the reference CSV."""
        raise NotImplementedError("gamma not in reference CSV; needs rocketcea backend.")

    def mw(self, Pc_pa: float, MR: float) -> float:
        """Not implemented — molecular weight not stored in the reference CSV."""
        raise NotImplementedError("mw not in reference CSV; needs rocketcea backend.")


# ---------------------------------------------------------------------------
# Live rocketcea backend
# ---------------------------------------------------------------------------

_R_UNIV: float = 8314.46  # universal gas constant [J/(kmol·K)]

# Maps PropellantPair.fuel string → rocketcea fuelName.
# rocketcea uses the CEA propellant card names; LCH4 is not a valid card.
_ROCKETCEA_FUEL: dict[str, str] = {
    "LCH4": "CH4",
    "CH4":  "CH4",
    "RP-1": "RP-1",
    "RP1":  "RP-1",
    "LH2":  "LH2",
    "H2":   "LH2",
}


@lru_cache(maxsize=4096)
def cea_isp_vac(
    *,
    propellant: "PropellantPair",
    Pc: float,
    MR: float,
    eps: float,
) -> float:
    """
    Equilibrium vacuum specific impulse [s] at nozzle exit via rocketcea.

    Uses rocketcea get_IvacCstrTc which applies the full equilibrium-flow
    expansion, accounting for species recombination in the nozzle.  This is
    more accurate than c_star × Cf(frozen-gamma) / g0, which overestimates
    Cf at fuel-rich conditions because frozen chamber gamma is elevated there.

    Source: Gordon & McBride (1994) NASA RP-1311; rocketcea wrapper.
    Validity: equilibrium expansion; MR range 0.5–10; any Pc and eps.

    Parameters
    ----------
    propellant : PropellantPair  Propellant combination.
    Pc : float                   Chamber pressure [Pa].
    MR : float                   O/F mass ratio [-].
    eps : float                  Nozzle expansion ratio Ae/At [-].

    Returns
    -------
    float
        Isp_vac [s] (equilibrium-flow, vacuum back-pressure).
    """
    try:
        from rocketcea.cea_obj import CEA_Obj
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "rocketcea is not installed.  pip install rocketcea"
        ) from exc

    fuel_cea = _ROCKETCEA_FUEL.get(propellant.fuel)
    if fuel_cea is None:
        raise ValueError(
            f"No rocketcea card name for PropellantPair.fuel='{propellant.fuel}'."
        )

    Pc_psia = Pc / PSI_TO_PA
    obj = CEA_Obj(oxName=propellant.oxidizer, fuelName=fuel_cea)
    isp_vac, _cstar_ft, _tc_r = obj.get_IvacCstrTc(Pc=Pc_psia, MR=MR, eps=eps)
    return float(isp_vac)


@lru_cache(maxsize=4096)
def cea_combustion(
    *,
    propellant: "PropellantPair",
    Pc: float,
    MR: float,
    expansion_ratio: float = 1.0,
    frozen: bool = False,
) -> "CombustionGas":
    """
    Live CEA equilibrium combustion state via rocketcea.

    Returns a CombustionGas with Tc, c*, gamma, MW, cp, mu, Pr all
    evaluated at the given (Pc, MR), not taken from a static preset.
    Results are LRU-cached (maxsize=4096); repeated calls with the same
    arguments are essentially free after the first call.

    Gas-phase transport properties:
      - gamma  : frozen-composition ratio of specific heats at chamber exit
                 (get_Chamber_MolWt_gamma).
      - cp     : derived from frozen gamma — cp = gamma·R/(gamma-1)/MW
                 (Anderson 2003 "Modern Compressible Flow" Eq. 1.12).
                 Internally consistent with gamma for nozzle calculations.
      - mu     : dynamic viscosity from get_Chamber_Transport [millipoise → Pa·s].
      - Pr     : Prandtl number from get_Chamber_Transport (dimensionless).

    Source: Gordon, S. and McBride, B.J. (1994). "Computer Program for
            Calculation of Complex Chemical Equilibrium Compositions and
            Applications." NASA RP-1311.
    Wrapper: rocketcea (https://rocketcea.readthedocs.io/).
    Validity: equilibrium chamber composition; frozen-flow nozzle assumption
              for gamma and cp; MR range 0.5–10 for most propellant pairs.

    Parameters
    ----------
    propellant : PropellantPair
        Propellant combination.  propellant.oxidizer and propellant.fuel
        must map to valid rocketcea card names (see _ROCKETCEA_FUEL).
    Pc : float
        Chamber pressure [Pa].
    MR : float
        Oxidiser-to-fuel mass ratio (O/F) [-].
    expansion_ratio : float
        Nozzle expansion ratio Ae/At [-].  Reserved for future exit-plane
        properties; chamber properties are independent of this value.
    frozen : bool
        If True, request frozen-flow transport from rocketcea.

    Returns
    -------
    CombustionGas
        Populated CombustionGas dataclass (all values in SI).

    Raises
    ------
    ModuleNotFoundError
        If rocketcea is not installed.
    ValueError
        If propellant.fuel is not in _ROCKETCEA_FUEL.
    """
    try:
        from rocketcea.cea_obj import CEA_Obj
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "rocketcea is not installed.  Install it with:\n"
            "    pip install rocketcea\n"
            "rocketcea is required for live CEA thermochemistry."
        ) from exc

    from physics.regen.bartz import CombustionGas

    fuel_cea = _ROCKETCEA_FUEL.get(propellant.fuel)
    if fuel_cea is None:
        raise ValueError(
            f"No rocketcea card name for PropellantPair.fuel='{propellant.fuel}'. "
            f"Known mappings: {sorted(_ROCKETCEA_FUEL)}.  "
            f"Add an entry to _ROCKETCEA_FUEL in physics/thermochem/cea.py."
        )

    Pc_psia = Pc / PSI_TO_PA
    obj = CEA_Obj(oxName=propellant.oxidizer, fuelName=fuel_cea)

    # Chamber temperature [°R → K]
    Tc = float(obj.get_Temperatures(Pc=Pc_psia, MR=MR)[0]) / 1.8

    # Characteristic velocity [ft/s → m/s]
    c_star = float(obj.get_Cstar(Pc=Pc_psia, MR=MR)) * FT_TO_M

    # Frozen-composition molecular weight [g/mol = kg/kmol] and gamma [-]
    MW_raw, gamma_raw = obj.get_Chamber_MolWt_gamma(Pc=Pc_psia, MR=MR)
    MW = float(MW_raw)
    gamma = float(gamma_raw)

    # Frozen Cp from ideal gas (self-consistent with gamma)
    cp = gamma * _R_UNIV / (gamma - 1.0) / MW

    # Transport: (Cp_eng, mu_mP, lambda_eng, Pr)
    transport = obj.get_Chamber_Transport(Pc=Pc_psia, MR=MR, frozen=int(frozen))
    mu = float(transport[1]) * 1.0e-4   # millipoise → Pa·s
    Pr = float(transport[3])            # dimensionless

    return CombustionGas(
        Tc=Tc, Pc=Pc, gamma=gamma, MW=MW, cp=cp, mu=mu, Pr=Pr, c_star=c_star,
    )
