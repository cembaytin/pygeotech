"""Infinite-slope stability analysis.

For a planar failure surface parallel to the ground at depth :math:`z`
in a slope inclined at :math:`\\beta` to the horizontal, the factor of
safety is

.. math::

    F = \\frac{c' + (\\gamma z\\cos^2\\beta - u)\\tan\\phi'}
             {\\gamma z\\sin\\beta\\cos\\beta},

where :math:`u` is the pore pressure on the failure plane. Special cases:

* Dry cohesionless soil (:math:`c'=0, u=0`): :math:`F = \\tan\\phi'/\\tan\\beta`.
* Seepage parallel to the slope with the water table at the surface:
  :math:`u = \\gamma_w z\\cos^2\\beta`, so

  .. math:: F = \\frac{c' + \\gamma' z\\cos^2\\beta\\tan\\phi'}
                     {\\gamma_{sat} z\\sin\\beta\\cos\\beta}.
"""

from __future__ import annotations

import math

from pygeotech.constants import GAMMA_W

__all__ = ["infinite_slope_factor"]


def infinite_slope_factor(
    slope_angle: float,
    friction_angle: float,
    cohesion: float = 0.0,
    gamma: float = 18.0,
    depth: float = 1.0,
    seepage: bool = False,
    gamma_sat: float = None,
    gamma_w: float = GAMMA_W,
) -> float:
    """Factor of safety of an infinite slope.

    Parameters
    ----------
    slope_angle
        Slope inclination :math:`\\beta` [deg].
    friction_angle
        Effective friction angle :math:`\\phi'` [deg].
    cohesion
        Effective cohesion :math:`c'` [kPa].
    gamma
        Moist unit weight [kN/m^3] (dry case).
    depth
        Depth to the failure plane :math:`z` [m].
    seepage
        If ``True``, full seepage parallel to the slope with the water
        table at the surface (uses ``gamma_sat`` and buoyant weight).
    gamma_sat
        Saturated unit weight [kN/m^3] (required if ``seepage``).
    gamma_w
        Unit weight of water [kN/m^3].

    Returns
    -------
    float
        Factor of safety.

    Examples
    --------
    >>> round(infinite_slope_factor(20.0, 30.0), 3)   # dry cohesionless
    1.586
    """
    beta = math.radians(slope_angle)
    phi = math.radians(friction_angle)
    cos_b, sin_b = math.cos(beta), math.sin(beta)

    if seepage:
        if gamma_sat is None:
            raise ValueError("gamma_sat is required when seepage=True.")
        unit_weight = gamma_sat
        effective_normal = (gamma_sat - gamma_w) * depth * cos_b ** 2
        driving = gamma_sat * depth * sin_b * cos_b
    else:
        unit_weight = gamma
        effective_normal = gamma * depth * cos_b ** 2
        driving = gamma * depth * sin_b * cos_b

    resisting = cohesion + effective_normal * math.tan(phi)
    if driving <= 0.0:
        raise ValueError("slope_angle must be positive.")
    return resisting / driving
