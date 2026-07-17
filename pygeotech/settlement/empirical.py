"""Semi-empirical settlement of footings on granular soils.

* **Schmertmann (1978)** strain-influence-factor method from CPT
  :math:`q_c`:

  .. math:: S_e = C_1 C_2\\,\\Delta p \\sum \\frac{I_z}{E_s}\\,\\Delta z ,

  with :math:`E_s = \\alpha q_c` (:math:`\\alpha = 2.5` axisymmetric,
  :math:`3.5` plane strain), a triangular strain-influence diagram peaking
  at :math:`I_{zp} = 0.5 + 0.1\\sqrt{\\Delta p/\\sigma'_{vp}}`, an
  embedment correction :math:`C_1 = 1 - 0.5\\,\\sigma'_{v0}/\\Delta p \\ge
  0.5` and a creep factor :math:`C_2 = 1 + 0.2\\log_{10}(t/0.1)`.

* **Burland & Burbidge (1985)** from SPT :math:`\\bar N_{60}`:

  .. math:: s = f_s f_l f_t\\,B^{0.7}\\,I_c\\,q',
      \\qquad I_c = \\frac{1.71}{\\bar N_{60}^{1.4}},

  with shape, thickness and time factors; ``s`` in mm for ``q'`` in kPa
  and ``B`` in m.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

__all__ = ["schmertmann_settlement", "burland_burbidge_settlement"]


def _schmertmann_single(
    dp: float,
    sigma_v0_eff_base: float,
    width: float,
    depth: np.ndarray,
    qc: np.ndarray,
    peak_stress: float,
    alpha: float,
    iz_surface: float,
    zp_factor: float,
    zmax_factor: float,
    c1: float,
    c2: float,
    n_grid: int,
) -> float:
    """Settlement for one geometry (axisymmetric or plane strain)."""
    zp = zp_factor * width
    zmax = zmax_factor * width
    izp = 0.5 + 0.1 * math.sqrt(max(dp, 0.0) / peak_stress)
    z = np.linspace(0.0, zmax, n_grid)
    qc_i = np.interp(z, depth, qc)
    es = alpha * np.maximum(qc_i, 1e-6)
    iz = np.where(
        z <= zp,
        iz_surface + (izp - iz_surface) * z / zp,
        izp * np.maximum(0.0, (zmax - z) / (zmax - zp)),
    )
    strain = np.trapz(iz / es, z)
    return c1 * c2 * dp * float(strain)


def schmertmann_settlement(
    net_pressure: float,
    sigma_v0_eff_base: float,
    width: float,
    depth: Sequence[float],
    qc: Sequence[float],
    length: Optional[float] = None,
    time_years: float = 0.1,
    peak_effective_stress: Optional[float] = None,
    n_grid: int = 400,
) -> float:
    """Schmertmann CPT settlement of a footing [m].

    Parameters
    ----------
    net_pressure
        Net applied pressure :math:`\\Delta p = q - \\sigma'_{v0}` at the
        founding level [kPa].
    sigma_v0_eff_base
        Effective overburden :math:`\\sigma'_{v0}` at the founding level
        [kPa] (for the :math:`C_1` correction).
    width
        Foundation width :math:`B` [m].
    depth, qc
        CPT tip-resistance profile *below the founding level*: depths [m]
        and :math:`q_c` [kPa].
    length
        Foundation length :math:`L` [m]; ``None`` = square. Intermediate
        ``L/B`` is linearly interpolated between the axisymmetric and
        plane-strain solutions.
    time_years
        Time since loading [years] for the creep factor (>= 0.1).
    peak_effective_stress
        :math:`\\sigma'_{vp}` at the depth of peak influence [kPa];
        defaults to ``sigma_v0_eff_base``.
    n_grid
        Integration points.

    Returns
    -------
    float
        Settlement [m].
    """
    depth = np.asarray(depth, dtype=float)
    qc = np.asarray(qc, dtype=float)
    if net_pressure <= 0.0:
        raise ValueError("net_pressure must be positive.")
    peak = peak_effective_stress if peak_effective_stress else sigma_v0_eff_base
    c1 = max(0.5, 1.0 - 0.5 * sigma_v0_eff_base / net_pressure)
    c2 = 1.0 + 0.2 * math.log10(max(time_years, 0.1) / 0.1)

    axi = _schmertmann_single(
        net_pressure, sigma_v0_eff_base, width, depth, qc, peak,
        alpha=2.5, iz_surface=0.1, zp_factor=0.5, zmax_factor=2.0,
        c1=c1, c2=c2, n_grid=n_grid)
    if length is None or length <= width:
        return axi
    plane = _schmertmann_single(
        net_pressure, sigma_v0_eff_base, width, depth, qc, peak,
        alpha=3.5, iz_surface=0.2, zp_factor=1.0, zmax_factor=4.0,
        c1=c1, c2=c2, n_grid=n_grid)
    weight = min(1.0, (length / width - 1.0) / 9.0)   # L/B 1->10
    return (1.0 - weight) * axi + weight * plane


def burland_burbidge_settlement(
    net_pressure: float,
    width: float,
    n60_avg: float,
    length: Optional[float] = None,
    preconsolidation: Optional[float] = None,
    layer_thickness: Optional[float] = None,
    time_years: Optional[float] = None,
) -> float:
    """Burland & Burbidge SPT settlement of a footing [mm].

    Parameters
    ----------
    net_pressure
        Net foundation pressure :math:`q'` [kPa].
    width
        Foundation width :math:`B` [m].
    n60_avg
        Mean :math:`N_{60}` over the influence depth :math:`z_I = B^{0.75}`.
    length
        Foundation length :math:`L` [m]; ``None`` = square.
    preconsolidation
        Preconsolidation pressure :math:`\\sigma'_p` [kPa]; if given the
        overconsolidated formulation is used.
    layer_thickness
        Thickness :math:`H` of the compressible layer [m]; if
        ``H < z_I`` the thickness factor :math:`f_l` is applied.
    time_years
        Time [years] for the creep factor :math:`f_t` (> 3 to matter).

    Returns
    -------
    float
        Settlement [mm].
    """
    if n60_avg <= 0.0 or width <= 0.0 or net_pressure <= 0.0:
        raise ValueError("n60_avg, width and net_pressure must be positive.")
    compressibility = 1.71 / n60_avg ** 1.4
    b07 = width ** 0.7

    l_over_b = 1.0 if length is None else max(1.0, length / width)
    fs = (1.25 * l_over_b / (l_over_b + 0.25)) ** 2

    fl = 1.0
    z_influence = width ** 0.75
    if layer_thickness is not None and layer_thickness < z_influence:
        ratio = layer_thickness / z_influence
        fl = ratio * (2.0 - ratio)

    ft = 1.0
    if time_years is not None and time_years > 3.0:
        ft = 1.0 + 0.3 + 0.2 * math.log10(time_years / 3.0)

    if preconsolidation is None or net_pressure <= preconsolidation:
        if preconsolidation is not None:       # over-consolidated, q' <= sp
            core = (compressibility / 3.0) * net_pressure
        else:                                   # normally consolidated
            core = compressibility * net_pressure
    else:                                       # over-consolidated, q' > sp
        core = compressibility * (net_pressure - (2.0 / 3.0) * preconsolidation)
    return fs * fl * ft * b07 * core
