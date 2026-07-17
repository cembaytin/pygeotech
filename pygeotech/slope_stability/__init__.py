"""Slope stability: infinite slope and circular method of slices.

Public API
----------
:func:`infinite_slope_factor`
    Infinite-slope factor of safety (dry / seepage).
:class:`SlipCircle`, :func:`simple_slope_surface`
    Trial slip circle and a simple-slope ground-surface generator.
:func:`slope_factor_of_safety`
    Method-of-slices FoS (Fellenius / Bishop / Janbu).
:func:`critical_circle`
    Grid search for the minimum-FoS circle.
:func:`yield_acceleration`, :func:`newmark_displacement`
    Seismic (Newmark sliding-block) permanent displacement.
:class:`LayeredSoil`, :class:`SlopeLayer`, :class:`PolylineSurface`,
:func:`advanced_factor_of_safety`
    Layered soils, non-circular slip surfaces and Spencer's method.
Plotting (lazy; needs matplotlib):
    :func:`plot_slope_circle`, :func:`plot_layered_slope`.
"""

from typing import List

from pygeotech.slope_stability.advanced import (
    LayeredSoil,
    PolylineSurface,
    SlopeLayer,
    advanced_factor_of_safety,
)
from pygeotech.slope_stability.infinite import infinite_slope_factor
from pygeotech.slope_stability.seismic import (
    newmark_displacement,
    yield_acceleration,
)
from pygeotech.slope_stability.slices import (
    SlipCircle,
    critical_circle,
    simple_slope_surface,
    slope_factor_of_safety,
)

__all__: List[str] = [
    "infinite_slope_factor",
    "SlipCircle",
    "simple_slope_surface",
    "slope_factor_of_safety",
    "critical_circle",
    "yield_acceleration",
    "newmark_displacement",
    "LayeredSoil",
    "SlopeLayer",
    "PolylineSurface",
    "advanced_factor_of_safety",
    "plot_slope_circle",
    "plot_layered_slope",
]

_LAZY = {"plot_slope_circle", "plot_layered_slope"}


def __getattr__(name: str):
    if name in _LAZY:
        from pygeotech.slope_stability import plotting

        return getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
