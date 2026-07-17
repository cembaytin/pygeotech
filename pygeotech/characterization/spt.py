"""Standard Penetration Test (SPT) corrections and correlations.

Corrections
-----------
The field blow count :math:`N` is first corrected to a 60 % energy ratio,

.. math:: N_{60} = N\\,\\frac{E_R}{60}\\,C_R\\,C_B\\,C_S,

with the rod-length (:math:`C_R`), borehole-diameter (:math:`C_B`) and
sampler (:math:`C_S`) factors after Skempton (1986). It is then corrected
for effective overburden to a reference stress of one atmosphere,

.. math:: (N_1)_{60} = C_N\\,N_{60},
    \\qquad C_N = \\sqrt{p_a / \\sigma'_{v0}} \\le C_{N,\\max}

(Liao & Whitman, 1986).

Correlations (sands)
--------------------
* Relative density (Skempton, 1986):
  :math:`D_r = \\sqrt{(N_1)_{60} / 60}`.
* Friction angle (Hatanaka & Uchida, 1996):
  :math:`\\phi' = \\sqrt{20\\,(N_1)_{60}} + 20`  [deg];
  or the Peck–Hanson–Thornburn regression (Wolff, 1989):
  :math:`\\phi' = 27.1 + 0.30\\,N_{60} - 5.4\\times10^{-4} N_{60}^2`.

Correlations (clays)
--------------------
* Undrained shear strength (Stroud, 1974):
  :math:`s_u = f_1 N_{60}` with :math:`f_1 \\approx 4\\!-\\!6` kPa.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from pygeotech.constants import P_ATM

__all__ = [
    "correct_n60",
    "overburden_factor",
    "corrected_n1_60",
    "relative_density_spt",
    "friction_angle_spt",
    "undrained_strength_spt",
    "shear_wave_velocity_spt",
    "SPTLog",
]


def correct_n60(
    n_field: float,
    energy_ratio: float = 60.0,
    rod_length_factor: float = 1.0,
    borehole_factor: float = 1.0,
    sampler_factor: float = 1.0,
) -> float:
    """Energy-correct a field blow count to :math:`N_{60}`.

    Parameters
    ----------
    n_field
        Field blow count :math:`N`.
    energy_ratio
        Hammer energy ratio :math:`E_R` [%] (e.g. 60 for a safety hammer,
        ~45 for a donut hammer, ~80 for automatic hammers).
    rod_length_factor, borehole_factor, sampler_factor
        Skempton (1986) correction factors :math:`C_R, C_B, C_S`.
    """
    return (n_field * energy_ratio / 60.0
            * rod_length_factor * borehole_factor * sampler_factor)


def overburden_factor(
    sigma_v_eff: float,
    pa: float = P_ATM,
    cap: float = 1.7,
) -> float:
    """Overburden correction factor :math:`C_N` (Liao & Whitman, 1986).

    Parameters
    ----------
    sigma_v_eff
        Effective vertical stress :math:`\\sigma'_{v0}` [kPa].
    pa
        Reference (atmospheric) pressure [kPa].
    cap
        Upper limit on :math:`C_N` (typically 1.7-2.0).
    """
    if sigma_v_eff <= 0.0:
        raise ValueError("effective stress must be positive.")
    return min(cap, math.sqrt(pa / sigma_v_eff))


def corrected_n1_60(
    n60: float, sigma_v_eff: float, pa: float = P_ATM, cap: float = 1.7
) -> float:
    """Overburden- and energy-corrected blow count :math:`(N_1)_{60}`."""
    return overburden_factor(sigma_v_eff, pa, cap) * n60


def relative_density_spt(n1_60: float) -> float:
    """Relative density :math:`D_r` (fraction) from :math:`(N_1)_{60}`.

    Uses Skempton's (1986) :math:`(N_1)_{60}/D_r^2 \\approx 60`, capped at 1.

    Examples
    --------
    >>> round(relative_density_spt(60.0), 3)
    1.0
    """
    return min(1.0, math.sqrt(max(0.0, n1_60) / 60.0))


def friction_angle_spt(
    n1_60: Optional[float] = None,
    n60: Optional[float] = None,
    method: str = "hatanaka",
) -> float:
    """Effective friction angle :math:`\\phi'` [deg] from SPT.

    Parameters
    ----------
    n1_60
        Corrected :math:`(N_1)_{60}` (required for ``"hatanaka"``).
    n60
        :math:`N_{60}` (required for ``"wolff"``).
    method
        ``"hatanaka"`` (Hatanaka & Uchida, 1996) or ``"wolff"``
        (Wolff, 1989 regression of the PHT chart).

    Examples
    --------
    >>> round(friction_angle_spt(n1_60=20.0, method="hatanaka"), 1)
    40.0
    """
    if method == "hatanaka":
        if n1_60 is None:
            raise ValueError("method 'hatanaka' requires n1_60.")
        return math.sqrt(20.0 * max(0.0, n1_60)) + 20.0
    if method == "wolff":
        if n60 is None:
            raise ValueError("method 'wolff' requires n60.")
        return 27.1 + 0.30 * n60 - 5.4e-4 * n60 ** 2
    raise ValueError("method must be 'hatanaka' or 'wolff'.")


def undrained_strength_spt(n60: float, factor: float = 4.5) -> float:
    """Undrained shear strength :math:`s_u` [kPa] for clays (Stroud, 1974).

    :math:`s_u = f_1 N_{60}`; ``factor`` :math:`f_1` is ~4-6 kPa
    (lower for high-plasticity clays).
    """
    return factor * n60


def shear_wave_velocity_spt(n60: float) -> float:
    """Shear-wave velocity :math:`V_s` [m/s] from :math:`N_{60}`.

    Uses the all-soils regression of Imai & Tonouchi (1982),
    :math:`V_s = 97\\,N_{60}^{0.314}`.
    """
    return 97.0 * max(0.0, n60) ** 0.314


@dataclass
class SPTLog:
    """A depth profile of SPT blow counts with automatic processing.

    Parameters
    ----------
    depth
        Test depths [m].
    n_field
        Field blow counts :math:`N` at each depth.
    profile
        Optional :class:`~pygeotech.stresses.geostatic.SoilProfile` used to
        evaluate the effective stress at each depth (for the overburden
        correction). Alternatively pass ``sigma_v_eff`` directly.
    sigma_v_eff
        Effective vertical stresses [kPa] aligned with ``depth`` (used if
        ``profile`` is not given).
    energy_ratio
        Hammer energy ratio [%].
    rod_length_factor, borehole_factor, sampler_factor
        Skempton correction factors.
    """

    depth: Sequence[float]
    n_field: Sequence[float]
    profile: Optional[object] = None
    sigma_v_eff: Optional[Sequence[float]] = None
    energy_ratio: float = 60.0
    rod_length_factor: float = 1.0
    borehole_factor: float = 1.0
    sampler_factor: float = 1.0
    pa: float = P_ATM

    def __post_init__(self) -> None:
        self.depth = np.asarray(self.depth, dtype=float)
        self.n_field = np.asarray(self.n_field, dtype=float)
        if self.depth.shape != self.n_field.shape:
            raise ValueError("depth and n_field must have the same length.")
        if self.sigma_v_eff is None:
            if self.profile is None:
                raise ValueError("provide either 'profile' or 'sigma_v_eff'.")
            self.sigma_v_eff = np.array(
                [self.profile.effective_stress(float(z)) for z in self.depth])
        else:
            self.sigma_v_eff = np.asarray(self.sigma_v_eff, dtype=float)

    def n60(self) -> np.ndarray:
        """Energy-corrected :math:`N_{60}` profile."""
        return np.array([
            correct_n60(n, self.energy_ratio, self.rod_length_factor,
                        self.borehole_factor, self.sampler_factor)
            for n in self.n_field])

    def n1_60(self) -> np.ndarray:
        """Overburden-corrected :math:`(N_1)_{60}` profile."""
        n60 = self.n60()
        return np.array([
            corrected_n1_60(n, float(s), self.pa)
            for n, s in zip(n60, self.sigma_v_eff)])

    def friction_angle(self, method: str = "hatanaka") -> np.ndarray:
        """Friction-angle profile [deg] (see :func:`friction_angle_spt`)."""
        if method == "wolff":
            return np.array([friction_angle_spt(n60=n, method="wolff")
                             for n in self.n60()])
        return np.array([friction_angle_spt(n1_60=n, method="hatanaka")
                         for n in self.n1_60()])

    def relative_density(self) -> np.ndarray:
        """Relative-density profile (fraction)."""
        return np.array([relative_density_spt(n) for n in self.n1_60()])
