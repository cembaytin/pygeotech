"""Interpretation of laboratory tests (oedometer and compaction).

Oedometer / consolidation test
------------------------------
The coefficient of consolidation :math:`c_v` from a single load increment:

* Casagrande log-time method: :math:`c_v = 0.197\\,H_{dr}^2 / t_{50}`.
* Taylor square-root-time method: :math:`c_v = 0.848\\,H_{dr}^2 / t_{90}`.

Compression indices come from the slopes of the :math:`e`-:math:`\\log
\\sigma'` curve; the preconsolidation pressure :math:`\\sigma'_p` is taken
here as the intersection of straight-line fits to the recompression and
virgin-compression branches (an objective bilinear estimate).

Compaction test
--------------
The Proctor curve peak gives the maximum dry unit weight and optimum
water content; the zero-air-voids line is

.. math:: \\gamma_{d,zav} = \\frac{G_s \\gamma_w}{1 + w\\,G_s}.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

from pygeotech.constants import GAMMA_W

__all__ = [
    "cv_casagrande",
    "cv_taylor",
    "compression_index",
    "preconsolidation_bilinear",
    "proctor_optimum",
    "zero_air_voids_curve",
]


def cv_casagrande(t50: float, drainage_path: float) -> float:
    """Coefficient of consolidation from :math:`t_{50}` (Casagrande).

    Parameters
    ----------
    t50
        Time to 50 % consolidation for the load increment [any time unit].
    drainage_path
        Longest drainage path :math:`H_{dr}` at that increment [m]
        (half the specimen height for double drainage).

    Returns
    -------
    float
        :math:`c_v` in ``length^2 / time`` (m^2 per the ``t50`` unit).
    """
    if t50 <= 0.0 or drainage_path <= 0.0:
        raise ValueError("t50 and drainage_path must be positive.")
    return 0.197 * drainage_path ** 2 / t50


def cv_taylor(t90: float, drainage_path: float) -> float:
    """Coefficient of consolidation from :math:`t_{90}` (Taylor)."""
    if t90 <= 0.0 or drainage_path <= 0.0:
        raise ValueError("t90 and drainage_path must be positive.")
    return 0.848 * drainage_path ** 2 / t90


def compression_index(
    sigma: Sequence[float], void_ratio: Sequence[float]
) -> float:
    """Slope :math:`-\\Delta e / \\Delta\\log_{10}\\sigma'` over the data.

    Fit through the supplied points (pass just the virgin-branch points to
    get :math:`C_c`, or the recompression points to get :math:`C_r`).

    Parameters
    ----------
    sigma
        Effective consolidation stresses :math:`\\sigma'` [kPa].
    void_ratio
        Corresponding void ratios :math:`e`.
    """
    s = np.asarray(sigma, dtype=float)
    e = np.asarray(void_ratio, dtype=float)
    if s.size < 2:
        raise ValueError("need at least two points.")
    slope, _ = np.polyfit(np.log10(s), e, 1)
    return -slope


def preconsolidation_bilinear(
    sigma: Sequence[float],
    void_ratio: Sequence[float],
    n_recompression: int,
    n_virgin: int,
) -> float:
    """Preconsolidation pressure :math:`\\sigma'_p` (bilinear intersection).

    Straight lines are fitted to the first ``n_recompression`` points
    (recompression branch) and the last ``n_virgin`` points (virgin
    branch) in :math:`e`-:math:`\\log\\sigma'` space; their intersection is
    returned as :math:`\\sigma'_p`.

    Parameters
    ----------
    sigma, void_ratio
        Full loading curve (increasing stress) [kPa, -].
    n_recompression, n_virgin
        Number of leading / trailing points defining each branch.
    """
    s = np.log10(np.asarray(sigma, dtype=float))
    e = np.asarray(void_ratio, dtype=float)
    if n_recompression < 2 or n_virgin < 2:
        raise ValueError("each branch needs at least two points.")
    m1, b1 = np.polyfit(s[:n_recompression], e[:n_recompression], 1)
    m2, b2 = np.polyfit(s[-n_virgin:], e[-n_virgin:], 1)
    if abs(m1 - m2) < 1e-12:
        raise ValueError("branches are parallel; cannot locate sigma_p.")
    log_sp = (b2 - b1) / (m1 - m2)
    return float(10.0 ** log_sp)


def proctor_optimum(
    water_content: Sequence[float], dry_unit_weight: Sequence[float]
) -> Tuple[float, float]:
    """Optimum water content and maximum dry unit weight (Proctor).

    A quadratic is fitted to the compaction points and its apex returned.

    Parameters
    ----------
    water_content
        Moulding water contents (decimal or %, returned in the same unit).
    dry_unit_weight
        Corresponding dry unit weights [kN/m^3].

    Returns
    -------
    (w_opt, gamma_d_max)
    """
    w = np.asarray(water_content, dtype=float)
    gd = np.asarray(dry_unit_weight, dtype=float)
    if w.size < 3:
        raise ValueError("need at least three compaction points.")
    a, b, c = np.polyfit(w, gd, 2)
    if a >= 0.0:
        raise ValueError("compaction data do not form a peak (a >= 0).")
    w_opt = -b / (2.0 * a)
    gamma_d_max = a * w_opt ** 2 + b * w_opt + c
    return float(w_opt), float(gamma_d_max)


def zero_air_voids_curve(
    water_content: Sequence[float],
    specific_gravity: float,
    gamma_w: float = GAMMA_W,
) -> np.ndarray:
    """Zero-air-voids dry unit weight for each water content [kN/m^3].

    Parameters
    ----------
    water_content
        Water contents as decimals (0.15 = 15 %).
    specific_gravity
        Specific gravity of solids :math:`G_s`.
    gamma_w
        Unit weight of water [kN/m^3].
    """
    w = np.asarray(water_content, dtype=float)
    return specific_gravity * gamma_w / (1.0 + w * specific_gravity)
