"""Foundations: shallow-foundation bearing capacity.

Public API
----------
:func:`bearing_capacity_factors`
    Nc, Nq, Ngamma for Terzaghi / Meyerhof / Hansen / Vesic.
:class:`ShallowFoundation`, :class:`BearingCapacityResult`
    Strip / rectangular footing capacity with shape, depth and
    water-table corrections.
Deep foundations:
    :func:`pile_capacity_alpha`, :func:`pile_capacity_beta`,
    :func:`alpha_api`, :func:`downdrag_force`,
    :func:`group_efficiency_converse_labarre`, :class:`PileCapacityResult`.
Laterally loaded piles (p-y finite-difference solver):
    :func:`solve_laterally_loaded_pile`, :func:`matlock_clay_py`,
    :func:`linear_subgrade_py`, :class:`PyResult`.
Plotting (lazy; needs matplotlib):
    :func:`plot_bearing_capacity_factors`.
"""

from typing import List

from pygeotech.foundations.bearing_capacity import (
    BearingCapacityResult,
    ShallowFoundation,
    bearing_capacity_factors,
)
from pygeotech.foundations.laterally_loaded_pile import (
    PyResult,
    api_sand_modulus,
    api_sand_py,
    linear_subgrade_py,
    matlock_clay_py,
    solve_laterally_loaded_pile,
)
from pygeotech.foundations.piles import (
    PileCapacityResult,
    alpha_api,
    downdrag_force,
    group_efficiency_converse_labarre,
    pile_capacity_alpha,
    pile_capacity_beta,
)

__all__: List[str] = [
    "bearing_capacity_factors",
    "ShallowFoundation",
    "BearingCapacityResult",
    "pile_capacity_alpha",
    "pile_capacity_beta",
    "alpha_api",
    "downdrag_force",
    "group_efficiency_converse_labarre",
    "PileCapacityResult",
    "solve_laterally_loaded_pile",
    "matlock_clay_py",
    "api_sand_py",
    "api_sand_modulus",
    "linear_subgrade_py",
    "PyResult",
    "plot_bearing_capacity_factors",
]

_LAZY = {"plot_bearing_capacity_factors"}


def __getattr__(name: str):
    if name in _LAZY:
        from pygeotech.foundations import plotting

        return getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
