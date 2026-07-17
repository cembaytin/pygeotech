"""Consolidation: Terzaghi 1-D theory and settlement estimation.

Public API
----------
Terzaghi solution:
    :class:`Consolidation1D`, :func:`average_degree_of_consolidation`,
    :func:`time_factor_from_degree`, :func:`excess_pressure_ratio`,
    :func:`degree_of_consolidation_at_depth`.
Settlement:
    :func:`primary_consolidation_settlement`,
    :func:`secondary_compression_settlement`,
    :func:`overconsolidation_ratio`.
Plotting (lazy; needs matplotlib):
    :func:`plot_isochrones`, :func:`plot_degree_vs_time_factor`,
    :func:`plot_time_settlement`.
"""

from typing import List

from pygeotech.consolidation.settlement import (
    overconsolidation_ratio,
    primary_consolidation_settlement,
    secondary_compression_settlement,
)
from pygeotech.consolidation.terzaghi import (
    Consolidation1D,
    average_degree_of_consolidation,
    degree_of_consolidation_at_depth,
    excess_pressure_ratio,
    time_factor_from_degree,
)

__all__: List[str] = [
    "Consolidation1D",
    "average_degree_of_consolidation",
    "time_factor_from_degree",
    "excess_pressure_ratio",
    "degree_of_consolidation_at_depth",
    "primary_consolidation_settlement",
    "secondary_compression_settlement",
    "overconsolidation_ratio",
    "plot_isochrones",
    "plot_degree_vs_time_factor",
    "plot_time_settlement",
]

_LAZY = {"plot_isochrones", "plot_degree_vs_time_factor",
         "plot_time_settlement"}


def __getattr__(name: str):
    if name in _LAZY:
        from pygeotech.consolidation import plotting

        return getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
