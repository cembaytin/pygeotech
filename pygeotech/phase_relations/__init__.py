"""Phase relations: weight-volume identities, index properties, USCS.

Public API
----------
:class:`Soil`
    Rule-based solver for weight-volume (phase) relationships.
:func:`classify_uscs` / :class:`USCSResult`
    ASTM D2487 Unified Soil Classification System engine.
:func:`classify_aashto` / :class:`AASHTOResult`
    AASHTO M145 classification with group index.
:func:`plot_plasticity_chart`
    Casagrande plasticity chart (imported lazily so that using the
    solver never requires a matplotlib backend).
"""

from typing import List

from pygeotech.phase_relations.aashto import (
    AASHTOResult,
    classify_aashto,
    group_index,
)
from pygeotech.phase_relations.classification import (
    USCSResult,
    a_line,
    classify_uscs,
    u_line,
)
from pygeotech.phase_relations.soil import (
    GAMMA_W,
    InconsistentInputError,
    Soil,
    porosity_from_void_ratio,
    void_ratio_from_porosity,
)

__all__: List[str] = [
    "GAMMA_W",
    "InconsistentInputError",
    "Soil",
    "USCSResult",
    "AASHTOResult",
    "a_line",
    "classify_uscs",
    "classify_aashto",
    "group_index",
    "plot_plasticity_chart",
    "porosity_from_void_ratio",
    "u_line",
    "void_ratio_from_porosity",
]


def __getattr__(name: str):  # PEP 562 lazy import of plotting helpers.
    if name == "plot_plasticity_chart":
        from pygeotech.phase_relations.plotting import plot_plasticity_chart

        return plot_plasticity_chart
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
