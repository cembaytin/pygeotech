"""Shear strength of unsaturated soils (extended Mohr-Coulomb).

Fredlund et al. (1978) extended the Mohr-Coulomb criterion with an
independent contribution from matric suction:

.. math::

    \\tau_f = c' + (\\sigma_n - u_a)\\tan\\phi'
             + (u_a - u_w)\\tan\\phi^b ,

where :math:`\\phi^b` is the rate of strength increase with suction.
Vanapalli et al. (1996) related :math:`\\tan\\phi^b` to the effective
degree of saturation, giving

.. math::

    \\tau_f = c' + (\\sigma_n - u_a)\\tan\\phi'
             + (u_a - u_w)\\,S_e\\,\\tan\\phi' ,

so that the suction contribution vanishes as the soil dries to residual.
"""

from __future__ import annotations

import math

__all__ = ["unsaturated_shear_strength", "unsaturated_shear_strength_vanapalli"]


def unsaturated_shear_strength(
    net_normal_stress: float,
    matric_suction: float,
    cohesion: float,
    friction_angle: float,
    phi_b: float,
) -> float:
    """Unsaturated shear strength with an explicit :math:`\\phi^b` [kPa].

    Parameters
    ----------
    net_normal_stress
        Net normal stress :math:`(\\sigma_n - u_a)` [kPa].
    matric_suction
        Matric suction :math:`(u_a - u_w)` [kPa].
    cohesion
        Effective cohesion :math:`c'` [kPa].
    friction_angle
        Effective friction angle :math:`\\phi'` [deg].
    phi_b
        Suction friction angle :math:`\\phi^b` [deg] (<= ``friction_angle``).
    """
    return (cohesion + net_normal_stress * math.tan(math.radians(friction_angle))
            + matric_suction * math.tan(math.radians(phi_b)))


def unsaturated_shear_strength_vanapalli(
    net_normal_stress: float,
    matric_suction: float,
    effective_saturation: float,
    cohesion: float,
    friction_angle: float,
) -> float:
    """Vanapalli et al. (1996) unsaturated shear strength [kPa].

    Uses the effective saturation :math:`S_e` as the suction coefficient,
    :math:`\\tan\\phi^b = S_e\\tan\\phi'`.
    """
    tan_phi = math.tan(math.radians(friction_angle))
    return (cohesion + net_normal_stress * tan_phi
            + matric_suction * effective_saturation * tan_phi)
