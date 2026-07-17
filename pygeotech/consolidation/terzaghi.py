"""Terzaghi one-dimensional consolidation theory.

Governing equation
-------------------
For a saturated clay layer with constant coefficient of consolidation
:math:`c_v`, the excess pore-water pressure :math:`u_e(z, t)` obeys the
diffusion equation

.. math::

    \\frac{\\partial u_e}{\\partial t}
        = c_v \\frac{\\partial^2 u_e}{\\partial z^2}.

With the dimensionless time factor and depth,

.. math::

    T_v = \\frac{c_v\\, t}{H_{dr}^2}, \\qquad Z = \\frac{z}{H_{dr}},

the classical Fourier-series solution for an initially uniform excess
pressure :math:`u_0` is

.. math::

    \\frac{u_e(Z, T_v)}{u_0}
      = \\sum_{m=0}^{\\infty} \\frac{2}{M}\\,\\sin(M Z)\\,e^{-M^2 T_v},
      \\qquad M = \\frac{\\pi}{2}(2m + 1),

and the average degree of consolidation is

.. math::

    U(T_v) = 1 - \\sum_{m=0}^{\\infty}\\frac{2}{M^2}\\,e^{-M^2 T_v}.

:math:`H_{dr}` is the longest drainage path: :math:`H/2` for a layer that
drains from both faces, :math:`H` for single-face drainage.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

__all__ = [
    "average_degree_of_consolidation",
    "time_factor_from_degree",
    "excess_pressure_ratio",
    "degree_of_consolidation_at_depth",
    "Consolidation1D",
]


def average_degree_of_consolidation(time_factor: float, terms: int = 200) -> float:
    """Average degree of consolidation :math:`U` for a time factor.

    Parameters
    ----------
    time_factor
        Dimensionless time factor :math:`T_v` (>= 0).
    terms
        Number of series terms.

    Returns
    -------
    float
        Average degree of consolidation in [0, 1].

    Examples
    --------
    >>> round(average_degree_of_consolidation(0.197), 3)
    0.5
    """
    if time_factor <= 0.0:
        return 0.0
    total = 0.0
    for m in range(terms):
        big_m = math.pi / 2.0 * (2 * m + 1)
        total += 2.0 / big_m ** 2 * math.exp(-big_m ** 2 * time_factor)
    return 1.0 - total


def time_factor_from_degree(degree: float) -> float:
    """Time factor :math:`T_v` from the average degree of consolidation.

    Uses Terzaghi's standard approximations

    .. math::

        T_v = \\frac{\\pi}{4}U^2 \\;(U \\le 0.60), \\qquad
        T_v = 1.781 - 0.933\\log_{10}(100 - U\\%) \\;(U > 0.60).

    Parameters
    ----------
    degree
        Average degree of consolidation as a fraction in [0, 1).
    """
    if not 0.0 <= degree < 1.0:
        raise ValueError("degree of consolidation must lie in [0, 1).")
    if degree <= 0.60:
        return math.pi / 4.0 * degree ** 2
    return 1.781 - 0.933 * math.log10(100.0 - degree * 100.0)


def excess_pressure_ratio(
    depth_ratio: float, time_factor: float, terms: int = 200
) -> float:
    """Excess pore-pressure ratio :math:`u_e/u_0` at normalised depth.

    Parameters
    ----------
    depth_ratio
        Normalised depth :math:`Z = z / H_{dr}` (0 to 2 for double
        drainage, 0 to 1 for single drainage).
    time_factor
        Time factor :math:`T_v`.
    terms
        Number of series terms.
    """
    if time_factor <= 0.0:
        return 1.0
    total = 0.0
    for m in range(terms):
        big_m = math.pi / 2.0 * (2 * m + 1)
        total += (2.0 / big_m * math.sin(big_m * depth_ratio)
                  * math.exp(-big_m ** 2 * time_factor))
    return total


def degree_of_consolidation_at_depth(
    depth_ratio: float, time_factor: float, terms: int = 200
) -> float:
    """Local degree of consolidation :math:`U_z = 1 - u_e/u_0`."""
    return 1.0 - excess_pressure_ratio(depth_ratio, time_factor, terms)


class Consolidation1D:
    """One-dimensional Terzaghi consolidation of a single clay layer.

    Parameters
    ----------
    cv
        Coefficient of consolidation :math:`c_v` [m^2/year] (any time
        unit is fine as long as it matches the queried times).
    layer_thickness
        Full layer thickness :math:`H` [m].
    drainage
        ``"double"`` (drains top and bottom, :math:`H_{dr}=H/2`) or
        ``"single"`` (:math:`H_{dr}=H`).

    Examples
    --------
    >>> clay = Consolidation1D(cv=2.0, layer_thickness=4.0)
    >>> round(clay.time_factor(1.0), 3)   # H_dr = 2 m
    0.5
    >>> round(clay.time_for_degree(0.90), 2)
    1.7
    """

    def __init__(
        self, cv: float, layer_thickness: float, drainage: str = "double"
    ) -> None:
        if cv <= 0.0 or layer_thickness <= 0.0:
            raise ValueError("cv and layer_thickness must be positive.")
        if drainage not in ("double", "single"):
            raise ValueError("drainage must be 'double' or 'single'.")
        self.cv = cv
        self.thickness = layer_thickness
        self.drainage = drainage
        self.drainage_path = (
            layer_thickness / 2.0 if drainage == "double" else layer_thickness
        )

    def time_factor(self, t: float) -> float:
        """Time factor :math:`T_v = c_v t / H_{dr}^2`."""
        return self.cv * t / self.drainage_path ** 2

    def average_degree(self, t: float) -> float:
        """Average degree of consolidation at time ``t``."""
        return average_degree_of_consolidation(self.time_factor(t))

    def time_for_degree(self, degree: float) -> float:
        """Time required to reach an average degree of consolidation."""
        return time_factor_from_degree(degree) * self.drainage_path ** 2 / self.cv

    def isochrone(
        self, t: float, n_points: int = 101
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Excess-pressure isochrone at time ``t``.

        Returns
        -------
        (z, ue_ratio)
            Physical depth from the top of the layer [m] and the excess
            pore-pressure ratio :math:`u_e/u_0` along it.
        """
        tv = self.time_factor(t)
        z = np.linspace(0.0, self.thickness, n_points)
        ratio = np.array(
            [excess_pressure_ratio(zi / self.drainage_path, tv) for zi in z]
        )
        return z, ratio

    def __repr__(self) -> str:
        return (f"Consolidation1D(cv={self.cv:g}, H={self.thickness:g}, "
                f"drainage={self.drainage!r}, H_dr={self.drainage_path:g})")
