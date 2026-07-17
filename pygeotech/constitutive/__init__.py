"""Constitutive modelling: advanced stress-strain-strength behaviour.

Public API
----------
:class:`CamClayParameters`, :class:`TriaxialResult`, :func:`triaxial_test`
    Modified Cam-Clay critical-state model with a drained/undrained
    triaxial stress-point (element-test) driver.
:class:`MohrCoulombParameters`, :func:`mc_triaxial_test`
    Linear-elastic perfectly-plastic Mohr-Coulomb model with the same
    triaxial driver (validated against q_f/p' = 6 sinφ/(3-sinφ)).
Plotting (lazy; needs matplotlib):
    :func:`plot_triaxial`.
"""

from typing import List

from pygeotech.constitutive.cam_clay import (
    CamClayParameters,
    TriaxialResult,
    triaxial_test,
)
from pygeotech.constitutive.mohr_coulomb import (
    MohrCoulombParameters,
    mc_triaxial_test,
)

__all__: List[str] = [
    "CamClayParameters",
    "TriaxialResult",
    "triaxial_test",
    "MohrCoulombParameters",
    "mc_triaxial_test",
    "plot_triaxial",
]

_LAZY = {"plot_triaxial"}


def __getattr__(name: str):
    if name in _LAZY:
        from pygeotech.constitutive import plotting

        return getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
