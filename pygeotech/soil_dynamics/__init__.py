"""Soil dynamics and geotechnical earthquake engineering.

Public API
----------
Liquefaction triggering (Idriss & Boulanger simplified procedure):
    :func:`cyclic_stress_ratio`, :func:`crr_from_spt`, :func:`crr_from_cpt`,
    :func:`fines_correction_spt`, :func:`magnitude_scaling_factor`,
    :func:`overburden_correction_ksigma`, :func:`stress_reduction_coefficient`,
    :func:`liquefaction_factor_of_safety`, :class:`LiquefactionResult`.
Site response and dynamic properties:
    :func:`site_natural_period`, :func:`site_natural_frequency`,
    :func:`transfer_function_amplitude`, :func:`peak_amplification`,
    :func:`max_shear_modulus`, :func:`shear_wave_velocity`.
"""

from typing import List

from pygeotech.soil_dynamics.liquefaction import (
    LiquefactionResult,
    crr_from_cpt,
    crr_from_spt,
    cyclic_stress_ratio,
    fines_correction_spt,
    liquefaction_factor_of_safety,
    magnitude_scaling_factor,
    overburden_correction_ksigma,
    stress_reduction_coefficient,
)
from pygeotech.soil_dynamics.site_response import (
    max_shear_modulus,
    peak_amplification,
    shear_wave_velocity,
    site_natural_frequency,
    site_natural_period,
    transfer_function_amplitude,
)

__all__: List[str] = [
    "cyclic_stress_ratio",
    "crr_from_spt",
    "crr_from_cpt",
    "fines_correction_spt",
    "magnitude_scaling_factor",
    "overburden_correction_ksigma",
    "stress_reduction_coefficient",
    "liquefaction_factor_of_safety",
    "LiquefactionResult",
    "site_natural_period",
    "site_natural_frequency",
    "transfer_function_amplitude",
    "peak_amplification",
    "max_shear_modulus",
    "shear_wave_velocity",
    "plot_liquefaction_chart",
    "plot_transfer_function",
]

_LAZY = {"plot_liquefaction_chart", "plot_transfer_function"}


def __getattr__(name: str):
    if name in _LAZY:
        from pygeotech.soil_dynamics import plotting

        return getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
