"""Cone Penetration Test (CPTu) interpretation after Robertson.

Corrected tip resistance (accounting for pore pressure acting on the cone
shoulder, net area ratio :math:`a`):

.. math:: q_t = q_c + (1 - a)\\,u_2 .

Normalised parameters and the soil-behaviour-type index (Robertson, 1990;
2009):

.. math::

    Q_{tn} = \\frac{q_t - \\sigma_{v0}}{p_a}
             \\left(\\frac{p_a}{\\sigma'_{v0}}\\right)^{n}, \\qquad
    F_r = \\frac{f_s}{q_t - \\sigma_{v0}}\\times 100\\%,

.. math::

    I_c = \\sqrt{(3.47 - \\log_{10} Q_{tn})^2
                + (\\log_{10} F_r + 1.22)^2},

where the stress exponent :math:`n = 0.381\\,I_c + 0.05\\,\\sigma'_{v0}/p_a
- 0.15 \\le 1` is found by iteration with :math:`I_c`.

Correlations
------------
* Undrained strength: :math:`s_u = (q_t - \\sigma_{v0})/N_{kt}`,
  :math:`N_{kt}\\approx 10\\!-\\!18`.
* Friction angle (Robertson & Campanella, 1983):
  :math:`\\phi' = \\arctan[0.1 + 0.38\\log_{10}(q_t/\\sigma'_{v0})]`.
* Relative density (Kulhawy & Mayne, 1990):
  :math:`D_r = \\sqrt{Q_{tn}/305}`.
* Unit weight (Robertson & Cabal, 2010):
  :math:`\\gamma/\\gamma_w = 0.27\\log_{10}R_f + 0.36\\log_{10}(q_t/p_a)
  + 1.236`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from pygeotech.constants import GAMMA_W, P_ATM

__all__ = [
    "corrected_qt",
    "soil_behaviour_type_index",
    "soil_behaviour_type",
    "undrained_strength_cpt",
    "friction_angle_cpt",
    "relative_density_cpt",
    "unit_weight_cpt",
    "CPTLog",
    "CPTResult",
]

#: Robertson (2010) soil-behaviour-type zones keyed by the Ic upper bound.
_SBT_ZONES: Tuple[Tuple[float, int, str], ...] = (
    (1.31, 7, "Gravelly sand to dense sand"),
    (2.05, 6, "Sands: clean sand to silty sand"),
    (2.60, 5, "Sand mixtures: silty sand to sandy silt"),
    (2.95, 4, "Silt mixtures: clayey silt to silty clay"),
    (3.60, 3, "Clays: silty clay to clay"),
    (math.inf, 2, "Organic soils: peat"),
)


def corrected_qt(qc: float, u2: float = 0.0, area_ratio: float = 0.8) -> float:
    """Corrected cone tip resistance :math:`q_t` [same units as ``qc``]."""
    return qc + (1.0 - area_ratio) * u2


def soil_behaviour_type_index(
    qt: float,
    fs: float,
    sigma_v0: float,
    sigma_v0_eff: float,
    pa: float = P_ATM,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Tuple[float, float, float]:
    """Robertson soil-behaviour-type index :math:`I_c` (iterative).

    Parameters
    ----------
    qt
        Corrected tip resistance [kPa].
    fs
        Sleeve friction [kPa].
    sigma_v0, sigma_v0_eff
        Total and effective vertical stress [kPa].
    pa
        Atmospheric pressure [kPa].

    Returns
    -------
    (Ic, Qtn, n)
        Soil-behaviour-type index, normalised tip resistance and the
        converged stress exponent.
    """
    net = qt - sigma_v0
    if net <= 0.0:
        raise ValueError("qt must exceed the total overburden sigma_v0.")
    fr = fs / net * 100.0
    fr = max(fr, 1e-3)
    n = 1.0
    ic = 0.0
    qtn = net / pa
    for _ in range(max_iter):
        qtn = (net / pa) * (pa / sigma_v0_eff) ** n
        ic = math.sqrt((3.47 - math.log10(qtn)) ** 2
                       + (math.log10(fr) + 1.22) ** 2)
        n_new = min(1.0, 0.381 * ic + 0.05 * (sigma_v0_eff / pa) - 0.15)
        if abs(n_new - n) < tol:
            n = n_new
            break
        n = n_new
    return ic, qtn, n


def soil_behaviour_type(ic: float) -> Tuple[int, str]:
    """Map a soil-behaviour-type index :math:`I_c` to a zone and label.

    Examples
    --------
    >>> soil_behaviour_type(1.9)
    (6, 'Sands: clean sand to silty sand')
    >>> soil_behaviour_type(3.1)[0]
    3
    """
    for upper, zone, label in _SBT_ZONES:
        if ic < upper:
            return zone, label
    return _SBT_ZONES[-1][1], _SBT_ZONES[-1][2]


def undrained_strength_cpt(
    qt: float, sigma_v0: float, nkt: float = 14.0
) -> float:
    """Undrained shear strength :math:`s_u` [kPa] for clays."""
    return (qt - sigma_v0) / nkt


def friction_angle_cpt(qt: float, sigma_v0_eff: float, pa: float = P_ATM) -> float:
    """Effective friction angle :math:`\\phi'` [deg] for sands.

    Robertson & Campanella (1983):
    :math:`\\phi' = \\arctan[0.1 + 0.38\\log_{10}(q_t/\\sigma'_{v0})]`.
    """
    if sigma_v0_eff <= 0.0:
        raise ValueError("effective stress must be positive.")
    return math.degrees(math.atan(0.1 + 0.38 * math.log10(qt / sigma_v0_eff)))


def relative_density_cpt(qtn: float) -> float:
    """Relative density :math:`D_r` (fraction) from :math:`Q_{tn}`.

    Kulhawy & Mayne (1990): :math:`D_r = \\sqrt{Q_{tn}/305}`, capped at 1.
    """
    return min(1.0, math.sqrt(max(0.0, qtn) / 305.0))


def unit_weight_cpt(
    rf: float, qt: float, pa: float = P_ATM, gamma_w: float = GAMMA_W
) -> float:
    """Estimated total unit weight :math:`\\gamma` [kN/m^3].

    Robertson & Cabal (2010) from the friction ratio ``rf`` [%] and
    corrected tip resistance.
    """
    rf = max(rf, 0.1)
    ratio = 0.27 * math.log10(rf) + 0.36 * math.log10(max(qt, pa) / pa) + 1.236
    return ratio * gamma_w


@dataclass(frozen=True)
class CPTResult:
    """Processed CPT profiles (all arrays aligned with ``depth``)."""

    depth: np.ndarray
    qt: np.ndarray
    friction_ratio: np.ndarray
    unit_weight: np.ndarray
    sigma_v0: np.ndarray
    sigma_v0_eff: np.ndarray
    ic: np.ndarray
    sbt_zone: np.ndarray
    undrained_strength: np.ndarray
    friction_angle: np.ndarray
    relative_density: np.ndarray


class CPTLog:
    """A CPTu sounding with automatic Robertson processing.

    Parameters
    ----------
    depth
        Measurement depths [m].
    qc
        Measured cone tip resistance [kPa].
    fs
        Sleeve friction [kPa].
    u2
        Pore pressure behind the cone [kPa]; defaults to 0 (CPT, not CPTu).
    area_ratio
        Net cone area ratio :math:`a` (typically 0.7-0.85).
    unit_weight
        Soil unit weight [kN/m^3] (scalar or per-depth). If ``None`` it is
        estimated from the CPT itself (Robertson & Cabal, 2010).
    water_table_depth
        Depth to the water table [m] (``inf`` = dry).
    nkt
        Cone factor for undrained strength.
    pa, gamma_w
        Atmospheric pressure and unit weight of water.
    """

    def __init__(
        self,
        depth: Sequence[float],
        qc: Sequence[float],
        fs: Sequence[float],
        u2: Optional[Sequence[float]] = None,
        area_ratio: float = 0.8,
        unit_weight: Optional[Sequence[float]] = None,
        water_table_depth: float = math.inf,
        nkt: float = 14.0,
        pa: float = P_ATM,
        gamma_w: float = GAMMA_W,
    ) -> None:
        self.depth = np.asarray(depth, dtype=float)
        self.qc = np.asarray(qc, dtype=float)
        self.fs = np.asarray(fs, dtype=float)
        self.u2 = (np.zeros_like(self.depth) if u2 is None
                   else np.asarray(u2, dtype=float))
        self.area_ratio = area_ratio
        self.water_table_depth = water_table_depth
        self.nkt = nkt
        self.pa = pa
        self.gamma_w = gamma_w
        self.qt = self.qc + (1.0 - area_ratio) * self.u2
        self.friction_ratio = np.where(
            self.qt > 0, self.fs / self.qt * 100.0, 0.0)
        if unit_weight is None:
            self.unit_weight = np.array([
                unit_weight_cpt(float(rf), float(qt), pa, gamma_w)
                for rf, qt in zip(self.friction_ratio, self.qt)])
        else:
            self.unit_weight = np.broadcast_to(
                np.asarray(unit_weight, dtype=float), self.depth.shape).copy()

    def _stresses(self) -> Tuple[np.ndarray, np.ndarray]:
        """Total and effective vertical stress profiles [kPa]."""
        z = self.depth
        gamma = self.unit_weight
        sigma = np.zeros_like(z)
        sigma[0] = gamma[0] * z[0]
        for i in range(1, len(z)):
            sigma[i] = sigma[i - 1] + 0.5 * (gamma[i - 1] + gamma[i]) * (
                z[i] - z[i - 1])
        u = np.where(z > self.water_table_depth,
                     self.gamma_w * (z - self.water_table_depth), 0.0)
        return sigma, sigma - u

    def process(self) -> CPTResult:
        """Compute Ic, SBT and the correlation profiles."""
        sigma_v0, sigma_v0_eff = self._stresses()
        n = len(self.depth)
        ic = np.zeros(n)
        sbt = np.zeros(n, dtype=int)
        su = np.zeros(n)
        phi = np.zeros(n)
        dr = np.zeros(n)
        for i in range(n):
            sv, sve = float(sigma_v0[i]), float(sigma_v0_eff[i])
            qt = float(self.qt[i])
            try:
                ic_i, qtn_i, _ = soil_behaviour_type_index(
                    qt, float(self.fs[i]), sv, sve, self.pa)
            except ValueError:
                ic_i, qtn_i = math.nan, math.nan
            ic[i] = ic_i
            sbt[i] = soil_behaviour_type(ic_i)[0] if math.isfinite(ic_i) else 0
            su[i] = undrained_strength_cpt(qt, sv, self.nkt)
            phi[i] = (friction_angle_cpt(qt, sve, self.pa) if sve > 0
                      else math.nan)
            dr[i] = relative_density_cpt(qtn_i) if math.isfinite(qtn_i) else math.nan
        return CPTResult(
            depth=self.depth, qt=self.qt, friction_ratio=self.friction_ratio,
            unit_weight=self.unit_weight, sigma_v0=sigma_v0,
            sigma_v0_eff=sigma_v0_eff, ic=ic, sbt_zone=sbt,
            undrained_strength=su, friction_angle=phi, relative_density=dr)
