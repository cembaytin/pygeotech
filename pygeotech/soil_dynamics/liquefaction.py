"""Simplified liquefaction-triggering analysis (Idriss & Boulanger).

The simplified procedure compares the earthquake-induced cyclic stress
ratio (CSR) with the soil's cyclic resistance ratio (CRR):

.. math::

    CSR = 0.65\\,\\frac{\\sigma_{v0}}{\\sigma'_{v0}}\\,\\frac{a_{max}}{g}\\,r_d ,

with the stress-reduction coefficient :math:`r_d` after Idriss (1999).
The resistance at :math:`M=7.5`, :math:`\\sigma'_v = 1` atm, from the
clean-sand penetration resistance (Idriss & Boulanger, 2008/2014):

.. math::

    CRR_{7.5} = \\exp\\!\\Big[\\tfrac{(N_1)_{60cs}}{14.1}
      + \\big(\\tfrac{(N_1)_{60cs}}{126}\\big)^2
      - \\big(\\tfrac{(N_1)_{60cs}}{23.6}\\big)^3
      + \\big(\\tfrac{(N_1)_{60cs}}{25.4}\\big)^4 - 2.8\\Big].

The factor of safety is
:math:`FS = CRR_{7.5}\\,MSF\\,K_\\sigma / CSR`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "stress_reduction_coefficient",
    "cyclic_stress_ratio",
    "fines_correction_spt",
    "crr_from_spt",
    "crr_from_cpt",
    "magnitude_scaling_factor",
    "overburden_correction_ksigma",
    "LiquefactionResult",
    "liquefaction_factor_of_safety",
]


def stress_reduction_coefficient(depth: float, magnitude: float) -> float:
    """Stress-reduction coefficient :math:`r_d` (Idriss, 1999).

    Valid to ~34 m; beyond that the 34 m value is held.
    """
    z = min(depth, 34.0)
    alpha = -1.012 - 1.126 * math.sin(z / 11.73 + 5.133)
    beta = 0.106 + 0.118 * math.sin(z / 11.28 + 5.142)
    return math.exp(alpha + beta * magnitude)


def cyclic_stress_ratio(
    a_max: float,
    sigma_v0: float,
    sigma_v0_eff: float,
    depth: float,
    magnitude: float,
    g: float = 9.81,
) -> float:
    """Earthquake cyclic stress ratio :math:`CSR` at a depth.

    Parameters
    ----------
    a_max
        Peak ground surface acceleration [m/s^2].
    sigma_v0, sigma_v0_eff
        Total and effective vertical stress [kPa].
    depth
        Depth [m].
    magnitude
        Earthquake moment magnitude :math:`M_w`.
    """
    rd = stress_reduction_coefficient(depth, magnitude)
    return 0.65 * (sigma_v0 / sigma_v0_eff) * (a_max / g) * rd


def fines_correction_spt(n1_60: float, fines_content: float) -> float:
    """Clean-sand equivalent :math:`(N_1)_{60cs}` (Idriss & Boulanger, 2008).

    Parameters
    ----------
    n1_60
        Overburden- and energy-corrected blow count :math:`(N_1)_{60}`.
    fines_content
        Fines content FC [%].
    """
    fc = max(fines_content, 0.0)
    delta = math.exp(1.63 + 9.7 / (fc + 0.01)
                     - (15.7 / (fc + 0.01)) ** 2)
    return n1_60 + delta


def crr_from_spt(n1_60cs: float) -> float:
    """Clean-sand :math:`CRR_{7.5}` from SPT :math:`(N_1)_{60cs}`."""
    n = min(n1_60cs, 37.0)          # beyond ~37 the soil is non-liquefiable
    return math.exp(n / 14.1 + (n / 126.0) ** 2 - (n / 23.6) ** 3
                    + (n / 25.4) ** 4 - 2.8)


def crr_from_cpt(qc1n_cs: float) -> float:
    """Clean-sand :math:`CRR_{7.5}` from CPT :math:`q_{c1N,cs}`."""
    q = min(qc1n_cs, 211.0)
    return math.exp(q / 540.0 + (q / 67.0) ** 2 - (q / 80.0) ** 3
                    + (q / 114.0) ** 4 - 3.0)


def magnitude_scaling_factor(magnitude: float) -> float:
    """Magnitude scaling factor MSF (Idriss & Boulanger, 2008)."""
    return min(1.8, 6.9 * math.exp(-magnitude / 4.0) - 0.058)


def overburden_correction_ksigma(
    sigma_v0_eff: float, n1_60cs: float, pa: float = 101.325
) -> float:
    """Overburden correction :math:`K_\\sigma` (Idriss & Boulanger, 2008)."""
    c_sigma = 1.0 / (18.9 - 2.55 * math.sqrt(min(n1_60cs, 37.0)))
    c_sigma = min(c_sigma, 0.3)
    return min(1.1, 1.0 - c_sigma * math.log(sigma_v0_eff / pa))


@dataclass(frozen=True)
class LiquefactionResult:
    """Liquefaction-triggering result at one depth."""

    csr: float
    crr: float
    factor_of_safety: float
    n1_60cs: float

    @property
    def liquefiable(self) -> bool:
        """``True`` if the factor of safety is below 1."""
        return self.factor_of_safety < 1.0

    def __str__(self) -> str:
        flag = "LIQUEFIES" if self.liquefiable else "stable"
        return (f"CSR = {self.csr:.3f} | CRR = {self.crr:.3f} | "
                f"FS = {self.factor_of_safety:.2f} ({flag})")


def liquefaction_factor_of_safety(
    n1_60: float,
    fines_content: float,
    a_max: float,
    magnitude: float,
    sigma_v0: float,
    sigma_v0_eff: float,
    depth: float,
    g: float = 9.81,
    pa: float = 101.325,
) -> LiquefactionResult:
    """Factor of safety against liquefaction (SPT-based, Idriss-Boulanger).

    Combines CSR, the clean-sand CRR, the magnitude scaling factor and the
    overburden correction:
    :math:`FS = CRR_{7.5}\\,MSF\\,K_\\sigma / CSR`.

    Examples
    --------
    >>> res = liquefaction_factor_of_safety(
    ...     n1_60=12, fines_content=10, a_max=0.3 * 9.81, magnitude=7.5,
    ...     sigma_v0=110, sigma_v0_eff=70, depth=6)
    >>> res.factor_of_safety > 0
    True
    """
    n1_60cs = fines_correction_spt(n1_60, fines_content)
    csr = cyclic_stress_ratio(a_max, sigma_v0, sigma_v0_eff, depth,
                              magnitude, g)
    crr = crr_from_spt(n1_60cs)
    msf = magnitude_scaling_factor(magnitude)
    ksigma = overburden_correction_ksigma(sigma_v0_eff, n1_60cs, pa)
    fs = crr * msf * ksigma / csr if csr > 0 else math.inf
    return LiquefactionResult(csr=csr, crr=crr, factor_of_safety=fs,
                              n1_60cs=n1_60cs)
