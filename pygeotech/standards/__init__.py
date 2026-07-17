"""Design-code adapters over the code-agnostic mechanics cores.

The domain modules compute *characteristic* (unfactored) soil parameters
and resistances; the adapters here apply the partial or load-and-
resistance factors of a chosen design code, realising pyGeotech's
multi-code strategy (see :class:`pygeotech.core.DesignStandard`).

Public API
----------
Eurocode 7 (EN 1997-1):
    :class:`PartialFactorSet`, :func:`factor_set_for`,
    :func:`design_shear_strength`, :func:`design_action`,
    :func:`design_bearing_resistance`, :func:`design_sliding_resistance`.
AASHTO LRFD:
    :class:`ResistanceFactors`, :data:`AASHTO_RESISTANCE`,
    :func:`factored_resistance`, :func:`strength_i_load`.
"""

from typing import List

from pygeotech.standards.aashto import (
    AASHTO_RESISTANCE,
    ResistanceFactors,
    factored_resistance,
    strength_i_load,
)
from pygeotech.standards.eurocode7 import (
    PartialFactorSet,
    design_action,
    design_bearing_resistance,
    design_shear_strength,
    design_sliding_resistance,
    factor_set_for,
)

__all__: List[str] = [
    "PartialFactorSet",
    "factor_set_for",
    "design_shear_strength",
    "design_action",
    "design_bearing_resistance",
    "design_sliding_resistance",
    "ResistanceFactors",
    "AASHTO_RESISTANCE",
    "factored_resistance",
    "strength_i_load",
]
