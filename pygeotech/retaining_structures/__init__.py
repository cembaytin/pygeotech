"""Retaining structures: earth pressures and wall stability.

Public API
----------
Earth-pressure coefficients:
    :func:`rankine_active_coefficient`, :func:`rankine_passive_coefficient`,
    :func:`coulomb_active_coefficient`, :func:`coulomb_passive_coefficient`.
Thrust and stability:
    :func:`active_thrust`, :class:`GravityWall`, :class:`WallStability`.
Plotting (lazy; needs matplotlib):
    :func:`plot_active_pressure_diagram`.
"""

from typing import List

from pygeotech.retaining_structures.earth_pressure import (
    GravityWall,
    WallStability,
    active_thrust,
    coulomb_active_coefficient,
    coulomb_passive_coefficient,
    rankine_active_coefficient,
    rankine_passive_coefficient,
)

__all__: List[str] = [
    "rankine_active_coefficient",
    "rankine_passive_coefficient",
    "coulomb_active_coefficient",
    "coulomb_passive_coefficient",
    "active_thrust",
    "GravityWall",
    "WallStability",
    "plot_active_pressure_diagram",
]

_LAZY = {"plot_active_pressure_diagram"}


def __getattr__(name: str):
    if name in _LAZY:
        from pygeotech.retaining_structures import plotting

        return getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
