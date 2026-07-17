"""Rock mechanics: classification and the Hoek-Brown criterion.

Public API
----------
Classification:
    :func:`rock_mass_rating` (RMR89) with :class:`RMRResult` and the
    :func:`ucs_rating` / :func:`rqd_rating` / :func:`spacing_rating`
    helpers; :func:`q_system` (Barton Q); :func:`gsi_from_rmr`.
Strength:
    :func:`hoek_brown_parameters`, :func:`hoek_brown_strength`,
    :class:`HoekBrownParameters`.
"""

from typing import List

from pygeotech.rock_mechanics.classification import (
    RMRResult,
    gsi_from_rmr,
    q_system,
    rock_mass_rating,
    rqd_rating,
    spacing_rating,
    ucs_rating,
)
from pygeotech.rock_mechanics.hoek_brown import (
    HoekBrownParameters,
    hoek_brown_parameters,
    hoek_brown_strength,
)

__all__: List[str] = [
    "rock_mass_rating",
    "RMRResult",
    "ucs_rating",
    "rqd_rating",
    "spacing_rating",
    "q_system",
    "gsi_from_rmr",
    "hoek_brown_parameters",
    "hoek_brown_strength",
    "HoekBrownParameters",
    "plot_hoek_brown",
]

_LAZY = {"plot_hoek_brown"}


def __getattr__(name: str):
    if name in _LAZY:
        from pygeotech.rock_mechanics import plotting

        return getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
