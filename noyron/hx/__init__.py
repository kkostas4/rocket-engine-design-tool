"""
noyron.hx — Heat Exchanger domain layer.

Mirrors Noyron HX (leap71.com/noyron): generic heat exchanger designs built
on the noyron.kernel.heat_exchanger substrate.

Public API
----------
PlateFinHeatExchanger — constant-section plate-fin HX stub (to be expanded).
"""
from noyron.hx.plate_fin import PlateFinHeatExchanger

__all__ = ["PlateFinHeatExchanger"]
