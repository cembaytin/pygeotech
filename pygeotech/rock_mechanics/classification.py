"""Rock-mass classification: RMR, the Q-system and GSI.

* **RMR** (Bieniawski, 1989): the sum of ratings for intact strength,
  RQD, discontinuity spacing, discontinuity condition and groundwater,
  plus an orientation adjustment.
* **Q-system** (Barton et al., 1974):
  :math:`Q = (RQD/J_n)(J_r/J_a)(J_w/SRF)`.
* **GSI** (Geological Strength Index): estimated from the basic RMR as
  :math:`GSI = RMR_{89} - 5` for :math:`RMR_{89} > 23`.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ucs_rating",
    "rqd_rating",
    "spacing_rating",
    "RMRResult",
    "rock_mass_rating",
    "q_system",
    "gsi_from_rmr",
]


def ucs_rating(ucs: float) -> int:
    """Bieniawski rating for intact uniaxial compressive strength [MPa]."""
    if ucs > 250:
        return 15
    if ucs > 100:
        return 12
    if ucs > 50:
        return 7
    if ucs > 25:
        return 4
    if ucs > 5:
        return 2
    if ucs > 1:
        return 1
    return 0


def rqd_rating(rqd: float) -> int:
    """Bieniawski rating for the rock quality designation RQD [%]."""
    if rqd > 90:
        return 20
    if rqd > 75:
        return 17
    if rqd > 50:
        return 13
    if rqd > 25:
        return 8
    return 3


def spacing_rating(spacing: float) -> int:
    """Bieniawski rating for discontinuity spacing [m]."""
    if spacing > 2.0:
        return 20
    if spacing > 0.6:
        return 15
    if spacing > 0.2:
        return 10
    if spacing > 0.06:
        return 8
    return 5


@dataclass(frozen=True)
class RMRResult:
    """Rock Mass Rating outcome."""

    rmr: int
    rock_class: str
    description: str

    def __str__(self) -> str:
        return f"RMR = {self.rmr} (Class {self.rock_class}: {self.description})"


_RMR_CLASSES = (
    (81, "I", "Very good rock"),
    (61, "II", "Good rock"),
    (41, "III", "Fair rock"),
    (21, "IV", "Poor rock"),
    (0, "V", "Very poor rock"),
)


def rock_mass_rating(
    ucs: float,
    rqd: float,
    spacing: float,
    condition_rating: float,
    groundwater_rating: float,
    orientation_adjustment: float = 0.0,
) -> RMRResult:
    """Compute the Bieniawski (1989) Rock Mass Rating.

    Parameters
    ----------
    ucs
        Intact uniaxial compressive strength [MPa].
    rqd
        Rock quality designation [%].
    spacing
        Mean discontinuity spacing [m].
    condition_rating
        Discontinuity condition rating (0-30).
    groundwater_rating
        Groundwater rating (0-15; 15 = completely dry).
    orientation_adjustment
        Adjustment for discontinuity orientation (<= 0).

    Returns
    -------
    RMRResult

    Examples
    --------
    >>> rock_mass_rating(120, 80, 0.8, 25, 15).rmr
    84
    """
    total = (ucs_rating(ucs) + rqd_rating(rqd) + spacing_rating(spacing)
             + condition_rating + groundwater_rating + orientation_adjustment)
    total = int(round(total))
    for lower, label, desc in _RMR_CLASSES:
        if total >= lower:
            return RMRResult(total, label, desc)
    return RMRResult(total, "V", "Very poor rock")


def q_system(
    rqd: float, jn: float, jr: float, ja: float, jw: float, srf: float
) -> float:
    """Barton rock-mass quality :math:`Q`.

    Parameters
    ----------
    rqd
        Rock quality designation [%] (use >= 10).
    jn, jr, ja, jw, srf
        Joint set number, roughness, alteration, water reduction and
        stress reduction factor.
    """
    if min(jn, ja, srf) <= 0.0:
        raise ValueError("jn, ja and srf must be positive.")
    return (max(rqd, 10.0) / jn) * (jr / ja) * (jw / srf)


def gsi_from_rmr(rmr89: float) -> float:
    """Estimate GSI from the basic RMR89 (valid for RMR > 23)."""
    if rmr89 <= 23:
        raise ValueError("GSI = RMR - 5 is only valid for RMR89 > 23; "
                         "use a GSI chart for weaker rock masses.")
    return rmr89 - 5.0
