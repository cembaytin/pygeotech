"""Shear strength: Mohr-Coulomb envelope fitting and stress paths.

Public API
----------
:class:`MohrCoulomb`
    Failure envelope with least-squares fitting from direct-shear or
    triaxial data.
:func:`stress_path_pq`, :func:`principal_stresses_at_failure`
    Stress-path construction and failure-stress helpers.
Plotting (lazy; needs matplotlib):
    :func:`plot_mohr_circles`, :func:`plot_stress_path`.
"""

from typing import List

from pygeotech.shear_strength.mohr_coulomb import (
    MohrCoulomb,
    principal_stresses_at_failure,
    stress_path_pq,
)

__all__: List[str] = [
    "MohrCoulomb",
    "stress_path_pq",
    "principal_stresses_at_failure",
    "plot_mohr_circles",
    "plot_stress_path",
]

_LAZY = {"plot_mohr_circles", "plot_stress_path"}


def __getattr__(name: str):
    if name in _LAZY:
        from pygeotech.shear_strength import plotting

        return getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
