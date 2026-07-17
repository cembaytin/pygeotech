"""Consolidation settlement of clay layers.

Primary consolidation settlement of a compressible layer of thickness
:math:`H`, initial void ratio :math:`e_0`, under an effective-stress
increase from :math:`\\sigma'_0` to :math:`\\sigma'_0 + \\Delta\\sigma'`:

* Normally consolidated (:math:`\\sigma'_p \\le \\sigma'_0`):

  .. math:: s_c = \\frac{C_c H}{1 + e_0}\\log_{10}
      \\frac{\\sigma'_0 + \\Delta\\sigma'}{\\sigma'_0}.

* Overconsolidated, staying below preconsolidation
  (:math:`\\sigma'_0 + \\Delta\\sigma' \\le \\sigma'_p`):

  .. math:: s_c = \\frac{C_r H}{1 + e_0}\\log_{10}
      \\frac{\\sigma'_0 + \\Delta\\sigma'}{\\sigma'_0}.

* Overconsolidated, crossing preconsolidation:

  .. math:: s_c = \\frac{C_r H}{1 + e_0}\\log_{10}\\frac{\\sigma'_p}{\\sigma'_0}
      + \\frac{C_c H}{1 + e_0}\\log_{10}
      \\frac{\\sigma'_0 + \\Delta\\sigma'}{\\sigma'_p}.

Secondary compression (creep) after the end of primary consolidation:

.. math:: s_s = \\frac{C_\\alpha H}{1 + e_p}\\log_{10}\\frac{t_2}{t_1}.
"""

from __future__ import annotations

import math
from typing import Optional

__all__ = [
    "primary_consolidation_settlement",
    "secondary_compression_settlement",
    "overconsolidation_ratio",
]


def primary_consolidation_settlement(
    thickness: float,
    void_ratio: float,
    sigma0: float,
    delta_sigma: float,
    cc: float,
    cr: Optional[float] = None,
    sigma_p: Optional[float] = None,
) -> float:
    """Primary consolidation settlement :math:`s_c` [m].

    Parameters
    ----------
    thickness
        Compressible layer thickness :math:`H` [m].
    void_ratio
        Initial in-situ void ratio :math:`e_0` [-].
    sigma0
        Initial effective vertical stress :math:`\\sigma'_0` [kPa].
    delta_sigma
        Effective vertical stress increment :math:`\\Delta\\sigma'` [kPa].
    cc
        Compression index :math:`C_c` [-].
    cr
        Recompression index :math:`C_r` [-]; required when the soil is
        overconsolidated (``sigma_p`` given). Defaults to ``cc`` for a
        normally consolidated soil.
    sigma_p
        Preconsolidation pressure :math:`\\sigma'_p` [kPa]. If ``None``
        or ``<= sigma0`` the soil is treated as normally consolidated.

    Returns
    -------
    float
        Settlement [m].
    """
    if min(thickness, void_ratio, sigma0, cc) <= 0.0 or delta_sigma < 0.0:
        raise ValueError("thickness, e0, sigma0, cc must be > 0; "
                         "delta_sigma >= 0.")
    sigma_f = sigma0 + delta_sigma
    factor = thickness / (1.0 + void_ratio)

    if sigma_p is None or sigma_p <= sigma0:
        return factor * cc * math.log10(sigma_f / sigma0)

    if cr is None:
        raise ValueError("cr is required for an overconsolidated soil "
                         "(sigma_p > sigma0).")
    if sigma_f <= sigma_p:
        return factor * cr * math.log10(sigma_f / sigma0)
    return factor * (cr * math.log10(sigma_p / sigma0)
                     + cc * math.log10(sigma_f / sigma_p))


def secondary_compression_settlement(
    thickness: float,
    void_ratio_p: float,
    c_alpha: float,
    t1: float,
    t2: float,
) -> float:
    """Secondary compression settlement :math:`s_s` [m].

    Parameters
    ----------
    thickness
        Layer thickness [m].
    void_ratio_p
        Void ratio at the end of primary consolidation :math:`e_p` [-].
    c_alpha
        Secondary compression index :math:`C_\\alpha` [-].
    t1, t2
        Start (end of primary) and end times; ``t2 > t1 > 0``.
    """
    if t1 <= 0.0 or t2 <= t1:
        raise ValueError("require t2 > t1 > 0.")
    return thickness / (1.0 + void_ratio_p) * c_alpha * math.log10(t2 / t1)


def overconsolidation_ratio(sigma_p: float, sigma0: float) -> float:
    """Overconsolidation ratio :math:`OCR = \\sigma'_p / \\sigma'_0`."""
    if sigma0 <= 0.0:
        raise ValueError("sigma0 must be positive.")
    return sigma_p / sigma0
