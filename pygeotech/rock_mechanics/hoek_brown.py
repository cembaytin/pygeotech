"""Generalized Hoek-Brown failure criterion (2002 edition).

.. math::

    \\sigma_1 = \\sigma_3 + \\sigma_{ci}
        \\left(m_b\\frac{\\sigma_3}{\\sigma_{ci}} + s\\right)^a ,

with the rock-mass parameters from the Geological Strength Index (GSI),
the intact material constant :math:`m_i` and the disturbance factor
:math:`D`:

.. math::

    m_b = m_i\\exp\\!\\frac{GSI - 100}{28 - 14D}, \\quad
    s = \\exp\\!\\frac{GSI - 100}{9 - 3D}, \\quad
    a = \\tfrac{1}{2} + \\tfrac{1}{6}\\big(e^{-GSI/15} - e^{-20/3}\\big).

The uniaxial compressive and tensile strengths of the rock mass are
:math:`\\sigma_c = \\sigma_{ci}\\,s^a` and
:math:`\\sigma_t = -s\\,\\sigma_{ci}/m_b`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

import numpy as np

__all__ = [
    "HoekBrownParameters",
    "hoek_brown_parameters",
    "hoek_brown_strength",
]

Number = Union[float, np.ndarray]


@dataclass(frozen=True)
class HoekBrownParameters:
    """Generalized Hoek-Brown rock-mass parameters."""

    mb: float
    s: float
    a: float
    sigma_ci: float

    def uniaxial_compressive_strength(self) -> float:
        """Rock-mass UCS :math:`\\sigma_c = \\sigma_{ci} s^a` [same unit]."""
        return self.sigma_ci * self.s ** self.a

    def tensile_strength(self) -> float:
        """Rock-mass tensile strength :math:`-s\\,\\sigma_{ci}/m_b`."""
        return -self.s * self.sigma_ci / self.mb

    def major_principal_stress(self, sigma3: Number) -> Number:
        """:math:`\\sigma_1` at failure for a confining stress ``sigma3``."""
        sigma3 = np.asarray(sigma3, dtype=float)
        return sigma3 + self.sigma_ci * (
            self.mb * sigma3 / self.sigma_ci + self.s) ** self.a


def hoek_brown_parameters(
    gsi: float,
    mi: float,
    sigma_ci: float,
    disturbance: float = 0.0,
) -> HoekBrownParameters:
    """Rock-mass Hoek-Brown parameters from GSI, ``mi`` and ``D``.

    Parameters
    ----------
    gsi
        Geological Strength Index (10-100).
    mi
        Intact-rock material constant.
    sigma_ci
        Intact uniaxial compressive strength [MPa].
    disturbance
        Blast/excavation disturbance factor :math:`D` (0 undisturbed to 1).

    Examples
    --------
    >>> p = hoek_brown_parameters(gsi=50, mi=10, sigma_ci=80)
    >>> round(p.mb, 3), round(p.s, 4), round(p.a, 3)
    (1.677, 0.0039, 0.506)
    """
    if not 0.0 <= disturbance <= 1.0:
        raise ValueError("disturbance D must lie in [0, 1].")
    mb = mi * math.exp((gsi - 100.0) / (28.0 - 14.0 * disturbance))
    s = math.exp((gsi - 100.0) / (9.0 - 3.0 * disturbance))
    a = 0.5 + (1.0 / 6.0) * (math.exp(-gsi / 15.0) - math.exp(-20.0 / 3.0))
    return HoekBrownParameters(mb=mb, s=s, a=a, sigma_ci=sigma_ci)


def hoek_brown_strength(
    sigma3: Number,
    sigma_ci: float,
    mb: float,
    s: float,
    a: float,
) -> Number:
    """Major principal stress :math:`\\sigma_1` at failure [same unit]."""
    sigma3 = np.asarray(sigma3, dtype=float)
    return sigma3 + sigma_ci * (mb * sigma3 / sigma_ci + s) ** a
