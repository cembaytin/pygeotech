"""Ground improvement: vertical drains and stone columns.

Public API
----------
Vertical drains (radial consolidation):
    :func:`drain_influence_diameter`, :func:`hansbo_factor`,
    :func:`radial_time_factor`, :func:`radial_degree_of_consolidation`,
    :func:`combined_degree_of_consolidation`.
Stone columns:
    :func:`area_replacement_ratio`,
    :func:`settlement_improvement_equilibrium`,
    :func:`priebe_improvement_factor`.
Plotting (lazy; needs matplotlib):
    :func:`plot_pvd_consolidation`.
"""

from typing import List

from pygeotech.ground_improvement.stone_columns import (
    area_replacement_ratio,
    priebe_improvement_factor,
    settlement_improvement_equilibrium,
)
from pygeotech.ground_improvement.vertical_drains import (
    combined_degree_of_consolidation,
    drain_influence_diameter,
    hansbo_factor,
    radial_degree_of_consolidation,
    radial_time_factor,
)

__all__: List[str] = [
    "drain_influence_diameter",
    "hansbo_factor",
    "radial_time_factor",
    "radial_degree_of_consolidation",
    "combined_degree_of_consolidation",
    "area_replacement_ratio",
    "settlement_improvement_equilibrium",
    "priebe_improvement_factor",
    "plot_pvd_consolidation",
]

_LAZY = {"plot_pvd_consolidation"}


def __getattr__(name: str):
    if name in _LAZY:
        from pygeotech.ground_improvement import plotting

        return getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
