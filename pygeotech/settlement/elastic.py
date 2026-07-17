"""Immediate (elastic) settlement of shallow foundations.

For a uniformly loaded flexible area on an elastic layer the settlement is

.. math::

    S_e = q\\,B'\\,\\frac{1 - \\nu^2}{E_s}\\,I_s\\,I_F,

where :math:`B'` is the least dimension of the contributing rectangle and
:math:`I_s` is Steinbrenner's influence factor for the corner of a
uniformly loaded rectangle on a layer of thickness :math:`H` over a rigid
base,

.. math::

    I_s = I_1 + \\frac{1 - 2\\nu}{1 - \\nu}\\,I_2 ,

with :math:`I_1, I_2` functions of :math:`m' = L/B` and :math:`n' = H/B`
(Steinbrenner, 1934; Fox depth factor :math:`I_F`). Settlement under the
centre is obtained by superposing four quarter-rectangles.

As :math:`H \\to \\infty` and for a square corner, :math:`I_s \\to 0.561`
(the classical half-space value), which the implementation reproduces.
"""

from __future__ import annotations

import math
from typing import Optional

__all__ = ["steinbrenner_influence", "elastic_settlement"]

_LARGE = 1.0e6      # numerical stand-in for an infinite layer thickness


def steinbrenner_influence(m: float, n: float, nu: float) -> float:
    """Steinbrenner corner influence factor :math:`I_s`.

    Parameters
    ----------
    m
        Aspect ratio :math:`L/B` of the loaded rectangle (>= 1).
    n
        Depth ratio :math:`H/B` of the compressible layer.
    nu
        Poisson's ratio of the soil.

    Examples
    --------
    >>> round(steinbrenner_influence(1.0, 1e6, 0.5), 3)   # square, half-space
    0.561
    """
    if m < 1.0:
        m = 1.0 / m if m > 0 else 1.0
    r1 = math.sqrt(m ** 2 + 1.0)
    r2 = math.sqrt(m ** 2 + n ** 2 + 1.0)
    rn = math.sqrt(m ** 2 + n ** 2)
    i1 = (1.0 / math.pi) * (
        m * math.log((1.0 + r1) * rn / (m * (1.0 + r2)))
        + math.log((m + r1) * math.sqrt(1.0 + n ** 2) / (m + r2))
    )
    i2 = (n / (2.0 * math.pi)) * math.atan(m / (n * r2))
    return i1 + (1.0 - 2.0 * nu) / (1.0 - nu) * i2


def elastic_settlement(
    pressure: float,
    width: float,
    length: Optional[float] = None,
    youngs_modulus: float = 20000.0,
    poisson_ratio: float = 0.3,
    layer_thickness: float = math.inf,
    position: str = "center",
    rigid: bool = False,
    depth_factor: float = 1.0,
) -> float:
    """Immediate elastic settlement of a uniformly loaded rectangle.

    Parameters
    ----------
    pressure
        Net contact pressure :math:`q` [kPa].
    width, length
        Foundation plan dimensions :math:`B, L` [m]; ``length=None`` (or
        equal to ``width``) gives a square.
    youngs_modulus
        Drained/undrained Young's modulus :math:`E_s` [kPa].
    poisson_ratio
        Poisson's ratio :math:`\\nu`.
    layer_thickness
        Thickness of the compressible layer below the base [m]
        (``inf`` = elastic half-space).
    position
        ``"center"`` or ``"corner"``.
    rigid
        If ``True``, apply the usual 0.93 rigidity reduction to the
        centre (flexible) settlement.
    depth_factor
        Fox embedment factor :math:`I_F` (1.0 at the surface).

    Returns
    -------
    float
        Settlement [m] (consistent with ``pressure``/``E`` in kPa).
    """
    length = width if length is None else length
    n_layer = _LARGE if math.isinf(layer_thickness) else layer_thickness
    factor = (1.0 - poisson_ratio ** 2) / youngs_modulus * depth_factor

    if position == "corner":
        b = min(width, length)
        m = max(width, length) / b
        i_s = steinbrenner_influence(m, n_layer / b, poisson_ratio)
        return pressure * b * factor * i_s
    if position == "center":
        b = min(width, length) / 2.0
        m = (max(width, length) / 2.0) / b
        i_s = steinbrenner_influence(m, n_layer / b, poisson_ratio)
        settlement = 4.0 * pressure * b * factor * i_s
        return 0.93 * settlement if rigid else settlement
    raise ValueError("position must be 'center' or 'corner'.")
