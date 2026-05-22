"""
plate_fin.py
============

Stub placeholder for a plate-fin heat exchanger design.

A plate-fin HX is a compact, multi-pass counterflow device used in
cryogenic propellant conditioning and turbopump preburner circuits.
This module provides the dataclass skeleton; the physics implementation
(channel pressure drop, fin effectiveness, thermal resistance network)
will be added in a future iteration.

Sources
-------
Kays, W.M. and London, A.L. (1984). "Compact Heat Exchangers," 3rd ed.,
    McGraw-Hill. Ch. 2 — plate-fin geometry definitions.

Shah, R.K. and Sekulic, D.P. (2003). "Fundamentals of Heat Exchanger
    Design," Wiley. Ch. 1 — classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from noyron.kernel.heat_exchanger.base import (
    ColdSideModel,
    HeatExchangerGeometry,
    HotSideModel,
)


@dataclass
class PlateFinHeatExchanger:
    """
    Constant-section plate-fin heat exchanger (stub).

    Parameters
    ----------
    length : float
        Flow-path length [m].
    n_channels_hot : int
        Number of hot-side channels.
    n_channels_cold : int
        Number of cold-side channels.
    fin_pitch : float
        Fin pitch (channel width + fin thickness) [m].
    fin_height : float
        Fin height (channel depth) [m].
    fin_thickness : float
        Fin wall thickness [m].
    plate_thickness : float
        Separating plate thickness [m].
    hot_side_model : HotSideModel
        Hot-fluid heat transfer model.
    cold_side_model : ColdSideModel
        Cold-fluid heat transfer model.

    Notes
    -----
    TODO: implement solve() using noyron.kernel.heat_exchanger.solver.solve_heat_exchanger.
    """

    length: float
    n_channels_hot: int
    n_channels_cold: int
    fin_pitch: float
    fin_height: float
    fin_thickness: float
    plate_thickness: float
    hot_side_model: HotSideModel
    cold_side_model: ColdSideModel

    def hydraulic_diameter_cold(self) -> float:
        """D_h = 4 * A_c / P_wetted for a rectangular cold channel [m]."""
        w = self.fin_pitch - self.fin_thickness
        h = self.fin_height
        return 4.0 * w * h / (2.0 * (w + h))
