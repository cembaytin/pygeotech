"""AASHTO soil classification (AASHTO M145 / ASTM D3282).

Soils are placed into groups A-1 to A-7 (with sub-groups) from the grain
size passing the No. 10 (2.0 mm), No. 40 (0.425 mm) and No. 200 (0.075 mm)
sieves together with the liquid limit and plasticity index of the minus-
No. 40 fraction. The **group index**

.. math::

    GI = (F_{200} - 35)\\,[0.2 + 0.005\\,(LL - 40)]
         + 0.01\\,(F_{200} - 15)\\,(PI - 10)

(reported as a non-negative integer) rates the quality as a subgrade;
higher is poorer. For the A-1, A-3 and A-2-4/A-2-5 groups ``GI`` is taken
as zero, and for A-2-6/A-2-7 only the second (plasticity) term applies.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["AASHTOResult", "group_index", "classify_aashto"]

_DESCRIPTIONS = {
    "A-1-a": "Well-graded gravel/stone fragments (excellent subgrade)",
    "A-1-b": "Coarse sand (excellent subgrade)",
    "A-3": "Fine sand, non-plastic (excellent to good subgrade)",
    "A-2-4": "Silty/clayey gravel and sand (good subgrade)",
    "A-2-5": "Silty/clayey gravel and sand (good subgrade)",
    "A-2-6": "Silty/clayey gravel and sand (good to fair subgrade)",
    "A-2-7": "Silty/clayey gravel and sand (fair subgrade)",
    "A-4": "Silty soil (fair to poor subgrade)",
    "A-5": "Elastic silty soil (poor subgrade)",
    "A-6": "Clayey soil (poor subgrade)",
    "A-7-5": "Clayey soil, moderate plasticity (poor subgrade)",
    "A-7-6": "Clayey soil, high plasticity (poor subgrade)",
}


@dataclass(frozen=True)
class AASHTOResult:
    """Outcome of an AASHTO classification.

    Attributes
    ----------
    group
        AASHTO group symbol, e.g. ``"A-2-6"``.
    group_index
        Group index (integer, >= 0); higher means a poorer subgrade.
    description
        Plain-language soil description and subgrade rating.
    """

    group: str
    group_index: int
    description: str

    def __str__(self) -> str:
        return f"{self.group}({self.group_index}): {self.description}"


def group_index(
    passing_no200: float,
    liquid_limit: float,
    plasticity_index: float,
    partial: bool = False,
) -> int:
    """Compute the AASHTO group index (rounded, non-negative).

    Parameters
    ----------
    passing_no200
        Percent passing the No. 200 sieve [%].
    liquid_limit, plasticity_index
        Atterberg limits [%].
    partial
        If ``True``, return only the plasticity term (used by the
        A-2-6 / A-2-7 groups).
    """
    f = passing_no200
    plasticity_term = 0.01 * (f - 15.0) * (plasticity_index - 10.0)
    if partial:
        return max(0, round(plasticity_term))
    liquid_term = (f - 35.0) * (0.2 + 0.005 * (liquid_limit - 40.0))
    return max(0, round(liquid_term + plasticity_term))


def classify_aashto(
    passing_no10: float,
    passing_no40: float,
    passing_no200: float,
    liquid_limit: float = 0.0,
    plasticity_index: float = 0.0,
) -> AASHTOResult:
    """Classify a soil per the AASHTO system (M145).

    Parameters
    ----------
    passing_no10, passing_no40, passing_no200
        Percent passing the No. 10, No. 40 and No. 200 sieves [%].
    liquid_limit, plasticity_index
        Liquid limit and plasticity index of the minus-No. 40 fraction
        [%] (0 for non-plastic soils).

    Returns
    -------
    AASHTOResult

    Examples
    --------
    >>> classify_aashto(50, 30, 8, liquid_limit=0, plasticity_index=0).group
    'A-1-a'
    >>> classify_aashto(100, 90, 45, liquid_limit=45, plasticity_index=20).group
    'A-7-6'
    """
    f10, f40, f200 = passing_no10, passing_no40, passing_no200
    ll, pi = liquid_limit, plasticity_index

    # ---- Granular materials: 35% or less passing No. 200 -------------
    if f200 <= 35.0:
        if f200 <= 15.0 and f40 <= 30.0 and f10 <= 50.0 and pi <= 6.0:
            group = "A-1-a"
            gi = 0
        elif f200 <= 25.0 and f40 <= 50.0 and pi <= 6.0:
            group = "A-1-b"
            gi = 0
        elif f200 <= 10.0 and pi == 0.0:
            group = "A-3"
            gi = 0
        else:
            # A-2 sub-groups from the plasticity chart of the fines.
            if ll <= 40.0 and pi <= 10.0:
                group, gi = "A-2-4", 0
            elif ll >= 41.0 and pi <= 10.0:
                group, gi = "A-2-5", 0
            elif ll <= 40.0 and pi >= 11.0:
                group = "A-2-6"
                gi = group_index(f200, ll, pi, partial=True)
            else:
                group = "A-2-7"
                gi = group_index(f200, ll, pi, partial=True)
        return AASHTOResult(group, gi, _DESCRIPTIONS[group])

    # ---- Silt-clay materials: more than 35% passing No. 200 ----------
    gi = group_index(f200, ll, pi)
    if ll <= 40.0 and pi <= 10.0:
        group = "A-4"
    elif ll >= 41.0 and pi <= 10.0:
        group = "A-5"
    elif ll <= 40.0 and pi >= 11.0:
        group = "A-6"
    else:                               # LL >= 41 and PI >= 11 -> A-7
        group = "A-7-5" if pi <= (ll - 30.0) else "A-7-6"
    return AASHTOResult(group, gi, _DESCRIPTIONS[group])
