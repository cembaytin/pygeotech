"""Site & material characterization: in-situ test interpretation.

Public API
----------
SPT:
    :func:`correct_n60`, :func:`overburden_factor`, :func:`corrected_n1_60`,
    :func:`relative_density_spt`, :func:`friction_angle_spt`,
    :func:`undrained_strength_spt`, :func:`shear_wave_velocity_spt`,
    :class:`SPTLog`.
CPT:
    :func:`corrected_qt`, :func:`soil_behaviour_type_index`,
    :func:`soil_behaviour_type`, :func:`undrained_strength_cpt`,
    :func:`friction_angle_cpt`, :func:`relative_density_cpt`,
    :func:`unit_weight_cpt`, :class:`CPTLog`, :class:`CPTResult`.
Plotting (lazy; needs matplotlib):
    :func:`plot_cpt_profile`, :func:`plot_spt_profile`.
"""

from typing import List

from pygeotech.characterization.cpt import (
    CPTLog,
    CPTResult,
    corrected_qt,
    friction_angle_cpt,
    relative_density_cpt,
    soil_behaviour_type,
    soil_behaviour_type_index,
    undrained_strength_cpt,
    unit_weight_cpt,
)
from pygeotech.characterization.laboratory import (
    compression_index,
    cv_casagrande,
    cv_taylor,
    preconsolidation_bilinear,
    proctor_optimum,
    zero_air_voids_curve,
)
from pygeotech.characterization.spt import (
    SPTLog,
    correct_n60,
    corrected_n1_60,
    friction_angle_spt,
    overburden_factor,
    relative_density_spt,
    shear_wave_velocity_spt,
    undrained_strength_spt,
)

__all__: List[str] = [
    # SPT
    "correct_n60", "overburden_factor", "corrected_n1_60",
    "relative_density_spt", "friction_angle_spt", "undrained_strength_spt",
    "shear_wave_velocity_spt", "SPTLog",
    # CPT
    "corrected_qt", "soil_behaviour_type_index", "soil_behaviour_type",
    "undrained_strength_cpt", "friction_angle_cpt", "relative_density_cpt",
    "unit_weight_cpt", "CPTLog", "CPTResult",
    # laboratory
    "cv_casagrande", "cv_taylor", "compression_index",
    "preconsolidation_bilinear", "proctor_optimum", "zero_air_voids_curve",
    # plotting
    "plot_cpt_profile", "plot_spt_profile",
]

_LAZY = {"plot_cpt_profile", "plot_spt_profile"}


def __getattr__(name: str):
    if name in _LAZY:
        from pygeotech.characterization import plotting

        return getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
