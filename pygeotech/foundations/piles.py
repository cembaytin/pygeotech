"""Axial capacity of single piles and pile groups.

Ultimate axial capacity is the sum of shaft (skin) friction and base
resistance:

.. math:: Q_u = Q_s + Q_b
        = \\int_0^L f_s\\,(\\pi D)\\,dz + q_b\\,A_b .

* **alpha-method** (undrained, total stress, clays):
  :math:`f_s = \\alpha s_u`, :math:`q_b = N_c s_{u,b}` with
  :math:`N_c = 9`. The adhesion factor follows API RP2A,

  .. math:: \\alpha = \\begin{cases}
      0.5\\,\\psi^{-0.5} & \\psi \\le 1\\\\
      0.5\\,\\psi^{-0.25} & \\psi > 1
      \\end{cases},\\quad \\psi = s_u/\\sigma'_v,\\; \\alpha\\le 1 .

* **beta-method** (drained, effective stress, sands and clays):
  :math:`f_s = \\beta\\,\\sigma'_v` with :math:`\\beta = K\\tan\\delta`,
  and :math:`q_b = N_q\\,\\sigma'_{v,b}`.

Group capacity is the lesser of (single-pile capacity x number x
efficiency) and block failure; the Converse-Labarre formula gives the
group efficiency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "alpha_api",
    "PileCapacityResult",
    "pile_capacity_alpha",
    "pile_capacity_beta",
    "group_efficiency_converse_labarre",
    "downdrag_force",
]


def alpha_api(su: float, sigma_v_eff: float) -> float:
    """API RP2A adhesion factor :math:`\\alpha` (capped at 1)."""
    if sigma_v_eff <= 0.0:
        return 1.0
    psi = su / sigma_v_eff
    alpha = 0.5 * psi ** -0.5 if psi <= 1.0 else 0.5 * psi ** -0.25
    return min(1.0, alpha)


@dataclass(frozen=True)
class PileCapacityResult:
    """Axial pile-capacity breakdown (all forces in kN)."""

    shaft: float
    base: float
    ultimate: float
    allowable: float
    factor_of_safety: float

    def __str__(self) -> str:
        return (f"Qs = {self.shaft:.0f} kN | Qb = {self.base:.0f} kN | "
                f"Qult = {self.ultimate:.0f} kN | "
                f"Qall = {self.allowable:.0f} kN (FS = {self.factor_of_safety:g})")


def _integrate_shaft(depth: np.ndarray, unit_friction: np.ndarray,
                     perimeter: float) -> float:
    """Trapezoidal integration of shaft friction over depth [kN]."""
    return float(np.trapz(unit_friction, depth)) * perimeter


def pile_capacity_alpha(
    diameter: float,
    depth: Sequence[float],
    su: Sequence[float],
    sigma_v_eff: Optional[Sequence[float]] = None,
    alpha: Optional[float] = None,
    base_su: Optional[float] = None,
    nc_base: float = 9.0,
    factor_of_safety: float = 3.0,
) -> PileCapacityResult:
    """Axial capacity of a pile in clay by the alpha-method.

    Parameters
    ----------
    diameter
        Pile diameter :math:`D` [m].
    depth
        Depths from head (0) to tip, increasing [m].
    su
        Undrained shear strength at each depth [kPa].
    sigma_v_eff
        Effective vertical stress at each depth [kPa]; if given the API
        :math:`\\alpha` is used, otherwise a constant ``alpha`` is applied.
    alpha
        Constant adhesion factor (used when ``sigma_v_eff`` is ``None``);
        defaults to 1.0.
    base_su
        Undrained strength at the tip [kPa]; defaults to the last ``su``.
    nc_base
        Base bearing-capacity factor (9 for deep piles in clay).
    factor_of_safety
        Global factor of safety.
    """
    depth = np.asarray(depth, dtype=float)
    su = np.asarray(su, dtype=float)
    perimeter = math.pi * diameter
    area = math.pi * diameter ** 2 / 4.0

    if sigma_v_eff is not None:
        sve = np.asarray(sigma_v_eff, dtype=float)
        alpha_arr = np.array([alpha_api(float(s), float(v))
                              for s, v in zip(su, sve)])
    else:
        alpha_arr = np.full_like(su, 1.0 if alpha is None else alpha)

    shaft = _integrate_shaft(depth, alpha_arr * su, perimeter)
    su_tip = float(su[-1]) if base_su is None else base_su
    base = nc_base * su_tip * area
    ultimate = shaft + base
    return PileCapacityResult(shaft, base, ultimate,
                              ultimate / factor_of_safety, factor_of_safety)


def pile_capacity_beta(
    diameter: float,
    depth: Sequence[float],
    sigma_v_eff: Sequence[float],
    beta: Optional[float] = None,
    k_earth: Optional[float] = None,
    delta: Optional[float] = None,
    nq: float = 20.0,
    base_sigma_v_eff: Optional[float] = None,
    limit_unit_base: Optional[float] = None,
    factor_of_safety: float = 3.0,
) -> PileCapacityResult:
    """Axial capacity of a pile by the beta (effective-stress) method.

    Parameters
    ----------
    diameter
        Pile diameter [m].
    depth, sigma_v_eff
        Depths [m] and effective vertical stresses [kPa].
    beta
        :math:`\\beta = K\\tan\\delta`; if ``None`` it is formed from
        ``k_earth`` and ``delta``.
    k_earth, delta
        Lateral earth-pressure coefficient :math:`K` and pile-soil
        friction angle :math:`\\delta` [deg].
    nq
        Base bearing-capacity factor.
    base_sigma_v_eff
        Effective stress at the tip [kPa]; defaults to the last value.
    limit_unit_base
        Optional cap on the unit base resistance [kPa].
    factor_of_safety
        Global factor of safety.
    """
    depth = np.asarray(depth, dtype=float)
    sve = np.asarray(sigma_v_eff, dtype=float)
    if beta is None:
        if k_earth is None or delta is None:
            raise ValueError("provide beta, or both k_earth and delta.")
        beta = k_earth * math.tan(math.radians(delta))
    perimeter = math.pi * diameter
    area = math.pi * diameter ** 2 / 4.0

    shaft = _integrate_shaft(depth, beta * sve, perimeter)
    sv_tip = float(sve[-1]) if base_sigma_v_eff is None else base_sigma_v_eff
    unit_base = nq * sv_tip
    if limit_unit_base is not None:
        unit_base = min(unit_base, limit_unit_base)
    base = unit_base * area
    ultimate = shaft + base
    return PileCapacityResult(shaft, base, ultimate,
                              ultimate / factor_of_safety, factor_of_safety)


def downdrag_force(
    diameter: float,
    depth: Sequence[float],
    sigma_v_eff: Sequence[float],
    beta: float = 0.25,
    neutral_plane_depth: Optional[float] = None,
) -> float:
    """Negative-skin-friction (downdrag) drag load on a pile [kN].

    Above the neutral plane the settling soil drags the pile down; the
    drag load is the integral of the negative unit friction
    :math:`f_n = \\beta\\,\\sigma'_v` over the pile perimeter down to the
    neutral plane.

    Parameters
    ----------
    diameter
        Pile diameter [m].
    depth, sigma_v_eff
        Depths [m] and effective vertical stresses [kPa].
    beta
        Effective-stress friction coefficient :math:`\\beta = K\\tan\\delta`.
    neutral_plane_depth
        Depth of the neutral plane [m]; if ``None`` the whole supplied
        profile is taken as the dragging zone.

    Returns
    -------
    float
        Downdrag (drag) load :math:`Q_n` [kN].
    """
    depth = np.asarray(depth, dtype=float)
    sve = np.asarray(sigma_v_eff, dtype=float)
    if neutral_plane_depth is not None:
        mask = depth <= neutral_plane_depth
        depth, sve = depth[mask], sve[mask]
    perimeter = math.pi * diameter
    return float(np.trapz(beta * sve, depth)) * perimeter


def group_efficiency_converse_labarre(
    rows: int, cols: int, spacing: float, diameter: float
) -> float:
    """Converse-Labarre pile-group efficiency (0-1).

    .. math:: E_g = 1 - \\frac{\\theta}{90}
        \\frac{(n-1)m + (m-1)n}{m\\,n},
        \\qquad \\theta = \\arctan(D/s).
    """
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be >= 1.")
    if rows == 1 and cols == 1:
        return 1.0
    theta = math.degrees(math.atan(diameter / spacing))
    m, n = rows, cols
    factor = ((n - 1) * m + (m - 1) * n) / (m * n)
    return 1.0 - theta / 90.0 * factor
