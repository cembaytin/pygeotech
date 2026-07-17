"""Unsaturated soil mechanics: SWCC and shear strength.

Public API
----------
Soil-water characteristic curve and permeability:
    :func:`van_genuchten_saturation`, :func:`van_genuchten_water_content`,
    :func:`fredlund_xing_water_content`, :func:`relative_permeability_vg`.
Shear strength:
    :func:`unsaturated_shear_strength` (phi^b),
    :func:`unsaturated_shear_strength_vanapalli` (S_e based).
Plotting (lazy; needs matplotlib):
    :func:`plot_swcc`.
"""

from typing import List

from pygeotech.unsaturated.strength import (
    unsaturated_shear_strength,
    unsaturated_shear_strength_vanapalli,
)
from pygeotech.unsaturated.swcc import (
    fredlund_xing_water_content,
    relative_permeability_vg,
    van_genuchten_saturation,
    van_genuchten_water_content,
)

__all__: List[str] = [
    "van_genuchten_saturation",
    "van_genuchten_water_content",
    "fredlund_xing_water_content",
    "relative_permeability_vg",
    "unsaturated_shear_strength",
    "unsaturated_shear_strength_vanapalli",
    "plot_swcc",
]

_LAZY = {"plot_swcc"}


def __getattr__(name: str):
    if name in _LAZY:
        from pygeotech.unsaturated import plotting

        return getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
