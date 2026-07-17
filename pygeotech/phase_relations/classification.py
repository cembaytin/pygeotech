"""Unified Soil Classification System (USCS, ASTM D2487) engine.

Classification logic
--------------------
With ``F200`` = percent passing the No. 200 (0.075 mm) sieve and
``F4`` = percent passing the No. 4 (4.75 mm) sieve:

* **Fine-grained** (``F200 >= 50``): position on the Casagrande
  plasticity chart relative to the A-line,

  .. math:: PI_A = 0.73\\,(LL - 20)

  - ``LL < 50``:  CL (above A-line, ``PI > 7``), ML (below A-line or
    ``PI < 4``), CL-ML (above A-line, ``4 <= PI <= 7``);
    organic soils -> OL.
  - ``LL >= 50``: CH (on/above A-line) or MH (below); organic -> OH.
  - Points above the U-line, :math:`PI_U = 0.9\\,(LL - 8)`, are flagged
    as probable data errors.

* **Coarse-grained** (``F200 < 50``): gravel fraction
  ``100 - F4`` vs. sand fraction ``F4 - F200`` selects the G/S prefix.

  - ``F200 < 5``  : clean; grading from :math:`C_u = D_{60}/D_{10}` and
    :math:`C_c = D_{30}^2/(D_{10} D_{60})`.
    Well graded (W): ``Cu >= 4`` (gravel) or ``Cu >= 6`` (sand) with
    ``1 <= Cc <= 3``; otherwise poorly graded (P).
  - ``F200 > 12`` : fines symbol from the plasticity chart
    (M / C / C-M dual, e.g. GM, SC, SC-SM).
  - ``5 <= F200 <= 12``: borderline dual symbols (e.g. SP-SM, GW-GC);
    CL-ML fines group with the clayey branch per the ASTM flow chart.

Group *names* follow ASTM D2487 with the common "with sand / with
gravel / sandy / gravelly" modifiers (15 % and 30 % coarse-fraction
thresholds). The rarely governing refinements of the full ASTM naming
flow charts (e.g. "with cobbles") are intentionally out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = ["USCSResult", "classify_uscs", "a_line", "u_line"]


def a_line(liquid_limit: float) -> float:
    """Casagrande A-line ordinate :math:`PI_A = 0.73 (LL - 20)`."""
    return 0.73 * (liquid_limit - 20.0)


def u_line(liquid_limit: float) -> float:
    """Casagrande U-line ordinate :math:`PI_U = 0.9 (LL - 8)`.

    The U-line is the empirical upper bound of natural soils on the
    plasticity chart; points above it usually indicate testing errors.
    """
    return 0.9 * (liquid_limit - 8.0)


_GROUP_NAMES: Dict[str, str] = {
    "GW": "Well-graded gravel",
    "GP": "Poorly graded gravel",
    "GM": "Silty gravel",
    "GC": "Clayey gravel",
    "GC-GM": "Silty, clayey gravel",
    "GW-GM": "Well-graded gravel with silt",
    "GW-GC": "Well-graded gravel with clay",
    "GP-GM": "Poorly graded gravel with silt",
    "GP-GC": "Poorly graded gravel with clay",
    "SW": "Well-graded sand",
    "SP": "Poorly graded sand",
    "SM": "Silty sand",
    "SC": "Clayey sand",
    "SC-SM": "Silty, clayey sand",
    "SW-SM": "Well-graded sand with silt",
    "SW-SC": "Well-graded sand with clay",
    "SP-SM": "Poorly graded sand with silt",
    "SP-SC": "Poorly graded sand with clay",
    "CL": "Lean clay",
    "CL-ML": "Silty clay",
    "ML": "Silt",
    "CH": "Fat clay",
    "MH": "Elastic silt",
    "OL": "Organic clay/silt (LL < 50)",
    "OH": "Organic clay/silt (LL >= 50)",
}


@dataclass(frozen=True)
class USCSResult:
    """Outcome of a USCS classification.

    Attributes
    ----------
    symbol
        USCS group symbol (e.g. ``"SP-SM"``, ``"CL"``).
    group_name
        ASTM-style group name including sand/gravel modifiers.
    fines
        Percent passing the No. 200 sieve [%].
    gravel
        Gravel fraction, percent retained on the No. 4 sieve [%].
    sand
        Sand fraction, percent between No. 4 and No. 200 sieves [%].
    notes
        Diagnostic remarks (borderline cases, U-line warnings, ...).
    """

    symbol: str
    group_name: str
    fines: float
    gravel: float
    sand: float
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def __str__(self) -> str:
        text = f"{self.symbol}: {self.group_name}"
        if self.notes:
            text += "  [" + "; ".join(self.notes) + "]"
        return text


def _fines_symbol(
    liquid_limit: float,
    plasticity_index: float,
    organic: bool,
    notes: List[str],
) -> str:
    """Symbol of the fine fraction from the Casagrande plasticity chart."""
    ll, pi = liquid_limit, plasticity_index
    if pi > u_line(ll) + 1e-9:
        notes.append(
            f"PI = {pi:.1f} plots above the U-line "
            f"(PI_U = {u_line(ll):.1f}); verify the Atterberg data."
        )
    if organic:
        return "OL" if ll < 50.0 else "OH"
    if ll >= 50.0:
        return "CH" if pi >= a_line(ll) else "MH"
    if pi > 7.0 and pi >= a_line(ll):
        return "CL"
    if pi < 4.0 or pi < a_line(ll):
        return "ML"
    return "CL-ML"


def _coefficients(
    cu: Optional[float],
    cc: Optional[float],
    d10: Optional[float],
    d30: Optional[float],
    d60: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """Resolve Cu and Cc, computing them from D10/D30/D60 if needed."""
    if cu is None and None not in (d10, d60):
        cu = d60 / d10  # type: ignore[operator]
    if cc is None and None not in (d10, d30, d60):
        cc = d30**2 / (d10 * d60)  # type: ignore[operator]
    return cu, cc


def _coarse_modifiers(symbol: str, gravel: float, sand: float) -> str:
    """'with sand' / 'with gravel' modifier for coarse-grained names."""
    other, label = (sand, "sand") if symbol[0] == "G" else (gravel, "gravel")
    if other < 15.0:
        return ""
    # Dual-symbol names already end in '... with silt/clay'.
    return f" and {label}" if " with " in _GROUP_NAMES[symbol] else f" with {label}"


def _fine_modifiers(gravel: float, sand: float) -> Tuple[str, str]:
    """(prefix, suffix) modifiers for fine-grained group names."""
    coarse = gravel + sand
    if coarse < 15.0:
        return "", ""
    label = "sand" if sand >= gravel else "gravel"
    if coarse < 30.0:
        return "", f" with {label}"
    return ("sandy " if label == "sand" else "gravelly "), ""


def classify_uscs(
    passing_no200: float,
    passing_no4: float = 100.0,
    *,
    liquid_limit: Optional[float] = None,
    plasticity_index: Optional[float] = None,
    plastic_limit: Optional[float] = None,
    cu: Optional[float] = None,
    cc: Optional[float] = None,
    d10: Optional[float] = None,
    d30: Optional[float] = None,
    d60: Optional[float] = None,
    organic: bool = False,
    nonplastic: bool = False,
) -> USCSResult:
    """Classify a soil per the Unified Soil Classification System.

    Parameters
    ----------
    passing_no200
        Percent passing the No. 200 (0.075 mm) sieve, 0-100 [%].
    passing_no4
        Percent passing the No. 4 (4.75 mm) sieve, 0-100 [%]
        (default 100, i.e. no gravel).
    liquid_limit, plasticity_index
        Atterberg limits of the minus-No. 40 fraction [%]. If
        ``plasticity_index`` is omitted but ``plastic_limit`` is given,
        ``PI = LL - PL`` is used.
    plastic_limit
        Plastic limit [%], alternative to ``plasticity_index``.
    cu, cc
        Uniformity coefficient :math:`C_u = D_{60}/D_{10}` and
        coefficient of curvature :math:`C_c = D_{30}^2/(D_{10}D_{60})`.
        Computed automatically when ``d10``/``d30``/``d60`` are given.
    d10, d30, d60
        Characteristic grain sizes [mm], alternatives to ``cu``/``cc``.
    organic
        ``True`` if the soil is organic (oven-dried LL ratio < 0.75
        per ASTM D2487); routes fine-grained soils to OL/OH.
    nonplastic
        ``True`` if the fines are non-plastic (NP); they are then
        treated as silt (M) without requiring Atterberg limits.

    Returns
    -------
    USCSResult
        Group symbol, group name, fraction breakdown and notes.

    Raises
    ------
    ValueError
        On invalid sieve percentages or when required plasticity or
        grading data are missing.

    Examples
    --------
    >>> classify_uscs(60.0, 90.0, liquid_limit=40, plasticity_index=22).symbol
    'CL'
    >>> classify_uscs(8.0, 95.0, d10=0.1, d30=0.2, d60=0.4,
    ...               nonplastic=True).symbol
    'SP-SM'
    """
    f200, f4 = float(passing_no200), float(passing_no4)
    if not 0.0 <= f200 <= 100.0 or not 0.0 <= f4 <= 100.0:
        raise ValueError("Sieve percentages must lie between 0 and 100.")
    if f200 > f4 + 1e-9:
        raise ValueError(
            f"Percent passing No. 200 ({f200:.1f}) cannot exceed percent "
            f"passing No. 4 ({f4:.1f})."
        )

    if plasticity_index is None and None not in (liquid_limit, plastic_limit):
        plasticity_index = liquid_limit - plastic_limit  # type: ignore[operator]

    gravel = 100.0 - f4
    sand = f4 - f200
    notes: List[str] = []

    def fines_group() -> str:
        if nonplastic:
            return "ML"
        if liquid_limit is None or plasticity_index is None:
            raise ValueError(
                "Atterberg limits (liquid_limit and plasticity_index or "
                "plastic_limit) are required, or set nonplastic=True."
            )
        return _fines_symbol(liquid_limit, plasticity_index, organic, notes)

    # ------------------------------------------------------------------
    # Fine-grained soils: 50 % or more passes the No. 200 sieve.
    # ------------------------------------------------------------------
    if f200 >= 50.0:
        symbol = fines_group()
        prefix, suffix = _fine_modifiers(gravel, sand)
        base = _GROUP_NAMES[symbol]
        name = prefix + base.lower() if prefix else base
        return USCSResult(symbol, (name + suffix).capitalize(),
                          f200, gravel, sand, tuple(notes))

    # ------------------------------------------------------------------
    # Coarse-grained soils.
    # ------------------------------------------------------------------
    prefix_sym = "G" if gravel > sand else "S"
    cu, cc = _coefficients(cu, cc, d10, d30, d60)
    cu_well = 4.0 if prefix_sym == "G" else 6.0

    def grading_letter() -> str:
        if cu is None or cc is None:
            raise ValueError(
                "Gradation data (cu/cc or d10/d30/d60) are required for "
                f"coarse soils with {f200:.0f}% fines (< 5% or 5-12%)."
            )
        return "W" if (cu >= cu_well and 1.0 <= cc <= 3.0) else "P"

    if f200 < 5.0:
        symbol = prefix_sym + grading_letter()
    elif f200 > 12.0:
        fines = fines_group()
        if fines in ("CL", "CH"):
            symbol = prefix_sym + "C"
        elif fines == "CL-ML":
            symbol = f"{prefix_sym}C-{prefix_sym}M"
            notes.append("Fines plot in the CL-ML zone.")
        else:
            symbol = prefix_sym + "M"
            if fines in ("OL", "OH"):
                notes.append("Organic fines; verify with ASTM D2487 OL/OH.")
    else:
        grade = grading_letter()
        fines = fines_group()
        # Per the ASTM flow chart, CL/CH and CL-ML fines take the clayey
        # branch, ML/MH the silty branch.
        second = "M" if fines in ("ML", "MH") else "C"
        symbol = f"{prefix_sym}{grade}-{prefix_sym}{second}"
        notes.append(f"Borderline fines content ({f200:.1f}%): dual symbol.")
        if fines == "CL-ML":
            notes.append("Fines plot in the CL-ML zone.")

    name = _GROUP_NAMES[symbol] + _coarse_modifiers(symbol, gravel, sand)
    return USCSResult(symbol, name, f200, gravel, sand, tuple(notes))
