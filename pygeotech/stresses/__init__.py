"""Stresses: geostatic profiles and load-induced elastic stresses.

Public API
----------
Geostatic (in-situ):
    :class:`SoilLayer`, :class:`SoilProfile`, :class:`StressState`.
Load-induced (elastic half-space):
    :func:`boussinesq_point`, :func:`westergaard_point`,
    :func:`boussinesq_rectangle`, :func:`boussinesq_circle_center`,
    :func:`influence_factor_rectangle`, :func:`induced_stress_area`.
Plotting (imported lazily; requires matplotlib):
    :func:`plot_geostatic_profile`, :func:`plot_pressure_bulb`.
"""

from typing import List

from pygeotech.stresses.geostatic import SoilLayer, SoilProfile, StressState
from pygeotech.stresses.induced import (
    boussinesq_circle_center,
    boussinesq_point,
    boussinesq_rectangle,
    induced_stress_area,
    influence_factor_rectangle,
    westergaard_point,
)

__all__: List[str] = [
    "SoilLayer",
    "SoilProfile",
    "StressState",
    "boussinesq_point",
    "westergaard_point",
    "boussinesq_rectangle",
    "boussinesq_circle_center",
    "influence_factor_rectangle",
    "induced_stress_area",
    "plot_geostatic_profile",
    "plot_pressure_bulb",
]

_LAZY = {"plot_geostatic_profile", "plot_pressure_bulb"}


def __getattr__(name: str):  # PEP 562 lazy import of plotting helpers.
    if name in _LAZY:
        from pygeotech.stresses import plotting

        return getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
