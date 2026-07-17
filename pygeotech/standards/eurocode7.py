"""Eurocode 7 (EN 1997-1) partial-factor design adapters.

The mechanics cores return *characteristic* (unfactored) soil parameters
and resistances; this module applies the EC7 partial factors so a single
physics implementation can be checked under any Design Approach.

Partial factor sets (EN 1997-1, Annex A):

======  =====================================================
Set     Factors
======  =====================================================
A1      gamma_G = 1.35, gamma_Q = 1.5   (actions)
A2      gamma_G = 1.0,  gamma_Q = 1.3
M1      gamma_phi = gamma_c = gamma_cu = 1.0   (materials)
M2      gamma_phi = gamma_c = 1.25, gamma_cu = 1.4
R1      gamma_Rv = gamma_Rh = 1.0   (resistances)
R2      gamma_Rv = 1.4, gamma_Rh = 1.1
R3      gamma_Rv = 1.0, gamma_Rh = 1.0
======  =====================================================

Design Approaches: DA1-C1 = A1+M1+R1, DA1-C2 = A2+M2+R1, DA2 = A1+M1+R2,
DA3 = A2(geotechnical)+M2+R3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from pygeotech.core import DesignStandard, PyGeotechError

__all__ = [
    "PartialFactorSet",
    "factor_set_for",
    "design_shear_strength",
    "design_action",
    "design_bearing_resistance",
    "design_sliding_resistance",
]


@dataclass(frozen=True)
class PartialFactorSet:
    """A combined EC7 partial-factor set for a Design Approach."""

    name: str
    gamma_g: float          # permanent action (unfavourable)
    gamma_q: float          # variable action (unfavourable)
    gamma_phi: float        # tan(phi')
    gamma_c: float          # effective cohesion c'
    gamma_cu: float         # undrained strength cu
    gamma_rv: float         # bearing / vertical resistance
    gamma_rh: float         # sliding / horizontal resistance


#: Design-Approach partial-factor sets (EN 1997-1).
_DA_SETS = {
    DesignStandard.EUROCODE7_DA1: PartialFactorSet(
        "DA1-C2", gamma_g=1.0, gamma_q=1.3, gamma_phi=1.25, gamma_c=1.25,
        gamma_cu=1.4, gamma_rv=1.0, gamma_rh=1.0),
    DesignStandard.EUROCODE7_DA2: PartialFactorSet(
        "DA2", gamma_g=1.35, gamma_q=1.5, gamma_phi=1.0, gamma_c=1.0,
        gamma_cu=1.0, gamma_rv=1.4, gamma_rh=1.1),
    DesignStandard.EUROCODE7_DA3: PartialFactorSet(
        "DA3", gamma_g=1.0, gamma_q=1.3, gamma_phi=1.25, gamma_c=1.25,
        gamma_cu=1.4, gamma_rv=1.0, gamma_rh=1.0),
}


def factor_set_for(standard: DesignStandard) -> PartialFactorSet:
    """Return the :class:`PartialFactorSet` for an EC7 Design Approach.

    Notes
    -----
    ``EUROCODE7_DA1`` returns the usually governing Combination 2 (A2+M2);
    Combination 1 (A1+M1) applies partial factors only to the actions.
    """
    if standard not in _DA_SETS:
        raise PyGeotechError(f"{standard} is not a Eurocode 7 Design Approach.")
    return _DA_SETS[standard]


def design_shear_strength(
    factors: PartialFactorSet,
    friction_angle: float = 0.0,
    cohesion: float = 0.0,
    undrained_strength: float = 0.0,
) -> Tuple[float, float, float]:
    """Design soil strengths from characteristic values.

    Returns ``(phi_d [deg], c_d [kPa], cu_d [kPa])`` with
    :math:`\\tan\\phi_d = \\tan\\phi_k/\\gamma_\\phi`,
    :math:`c_d = c_k/\\gamma_c`, :math:`c_{u,d} = c_{u,k}/\\gamma_{cu}`.
    """
    phi_d = math.degrees(math.atan(
        math.tan(math.radians(friction_angle)) / factors.gamma_phi))
    return phi_d, cohesion / factors.gamma_c, undrained_strength / factors.gamma_cu


def design_action(
    factors: PartialFactorSet, permanent: float, variable: float = 0.0
) -> float:
    """Design action :math:`E_d = \\gamma_G G_k + \\gamma_Q Q_k`."""
    return factors.gamma_g * permanent + factors.gamma_q * variable


def design_bearing_resistance(
    factors: PartialFactorSet, characteristic_resistance: float
) -> float:
    """Design bearing resistance :math:`R_d = R_k/\\gamma_{Rv}`."""
    return characteristic_resistance / factors.gamma_rv


def design_sliding_resistance(
    factors: PartialFactorSet, characteristic_resistance: float
) -> float:
    """Design sliding resistance :math:`R_d = R_k/\\gamma_{Rh}`."""
    return characteristic_resistance / factors.gamma_rh
