"""Settlement: immediate (elastic) and semi-empirical methods.

Public API
----------
Elastic:
    :func:`steinbrenner_influence`, :func:`elastic_settlement`.
Semi-empirical (granular soils):
    :func:`schmertmann_settlement` (CPT),
    :func:`burland_burbidge_settlement` (SPT).
Consolidation settlement lives in :mod:`pygeotech.consolidation`.
Plotting (lazy; needs matplotlib):
    :func:`plot_strain_influence`.
"""

from typing import List

from pygeotech.settlement.elastic import elastic_settlement, steinbrenner_influence
from pygeotech.settlement.empirical import (
    burland_burbidge_settlement,
    schmertmann_settlement,
)

__all__: List[str] = [
    "steinbrenner_influence",
    "elastic_settlement",
    "schmertmann_settlement",
    "burland_burbidge_settlement",
    "plot_strain_influence",
]

_LAZY = {"plot_strain_influence"}


def __getattr__(name: str):
    if name in _LAZY:
        from pygeotech.settlement import plotting

        return getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
