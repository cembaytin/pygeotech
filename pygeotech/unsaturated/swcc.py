"""Soil-water characteristic curve (SWCC) and unsaturated permeability.

**van Genuchten (1980):** effective saturation as a function of matric
suction :math:`\\psi`,

.. math::

    S_e = \\left[1 + (\\alpha\\psi)^n\\right]^{-m}, \\quad m = 1 - 1/n,
    \\qquad \\theta = \\theta_r + (\\theta_s - \\theta_r)\\,S_e .

**Fredlund & Xing (1994):**

.. math::

    \\theta(\\psi) = C(\\psi)\\,\\frac{\\theta_s}
    {\\left[\\ln\\!\\big(e + (\\psi/a)^n\\big)\\right]^m},

with the correction factor :math:`C(\\psi)` that forces
:math:`\\theta \\to 0` at the residual suction.

**Relative permeability (van Genuchten-Mualem):**

.. math:: k_r(S_e) = S_e^{\\,L}
    \\left[1 - \\big(1 - S_e^{1/m}\\big)^m\\right]^2, \\quad L = 0.5 .
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np

__all__ = [
    "van_genuchten_saturation",
    "van_genuchten_water_content",
    "fredlund_xing_water_content",
    "relative_permeability_vg",
]

Number = Union[float, np.ndarray]


def van_genuchten_saturation(
    suction: Number, alpha: float, n: float
) -> Number:
    """Effective saturation :math:`S_e` (van Genuchten).

    Parameters
    ----------
    suction
        Matric suction :math:`\\psi` [kPa] (>= 0).
    alpha
        Inverse of the air-entry value [1/kPa].
    n
        Pore-size distribution parameter (> 1).
    """
    psi = np.maximum(np.asarray(suction, dtype=float), 0.0)
    m = 1.0 - 1.0 / n
    return (1.0 + (alpha * psi) ** n) ** (-m)


def van_genuchten_water_content(
    suction: Number,
    theta_s: float,
    theta_r: float,
    alpha: float,
    n: float,
) -> Number:
    """Volumetric water content :math:`\\theta` (van Genuchten)."""
    se = van_genuchten_saturation(suction, alpha, n)
    return theta_r + (theta_s - theta_r) * se


def fredlund_xing_water_content(
    suction: Number,
    theta_s: float,
    a: float,
    n: float,
    m: float,
    residual_suction: float = 1.0e6,
) -> Number:
    """Volumetric water content :math:`\\theta` (Fredlund & Xing).

    Parameters
    ----------
    suction
        Matric suction :math:`\\psi` [kPa].
    theta_s
        Saturated volumetric water content.
    a
        Fitting parameter related to the air-entry value [kPa].
    n, m
        Fitting parameters controlling the slope and residual curvature.
    residual_suction
        Suction at which the water content is essentially zero [kPa].
    """
    psi = np.maximum(np.asarray(suction, dtype=float), 1e-6)
    correction = 1.0 - (np.log(1.0 + psi / residual_suction)
                        / math.log(1.0 + 1.0e6 / residual_suction))
    return correction * theta_s / (np.log(np.e + (psi / a) ** n)) ** m


def relative_permeability_vg(suction: Number, alpha: float, n: float) -> Number:
    """Relative hydraulic conductivity :math:`k_r` (van Genuchten-Mualem)."""
    se = van_genuchten_saturation(suction, alpha, n)
    m = 1.0 - 1.0 / n
    return np.sqrt(se) * (1.0 - (1.0 - se ** (1.0 / m)) ** m) ** 2
