"""Weight-volume (phase) relationship solver.

Physical background
-------------------
A soil element is idealized as a three-phase system (solids, water, air).
With :math:`\\gamma_w` the unit weight of water, the fundamental
definitions and identities used by the solver are:

.. math::

    e = \\frac{V_v}{V_s}, \\qquad
    n = \\frac{V_v}{V}   = \\frac{e}{1+e}, \\qquad
    w = \\frac{W_w}{W_s}, \\qquad
    S_r = \\frac{V_w}{V_v}

.. math::

    S_r \\, e = w \\, G_s
    \\qquad \\text{(saturation identity)}

.. math::

    \\gamma_d = \\frac{G_s \\gamma_w}{1+e}, \\qquad
    \\gamma   = \\gamma_d (1 + w) = \\frac{(G_s + S_r e)\\,\\gamma_w}{1+e},
    \\qquad
    \\gamma_{sat} = \\frac{(G_s + e)\\,\\gamma_w}{1+e}, \\qquad
    \\gamma' = \\gamma_{sat} - \\gamma_w

The :class:`Soil` class propagates any consistent subset of known
properties through this closed set of identities (a fixed-point,
rule-based resolution), verifies redundant inputs for consistency and
records every derivation step for reproducibility.

Units and conventions
---------------------
* Unit weights in kN/m^3 (default :math:`\\gamma_w` = 9.81 kN/m^3).
* ``w`` (water content) and ``sr`` (degree of saturation) are decimal
  ratios (0.25 = 25 %), not percentages.
* ``e`` (void ratio) and ``n`` (porosity) are dimensionless.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from pygeotech.constants import GAMMA_W

_EPS: float = 1e-12


class InconsistentInputError(ValueError):
    """Raised when redundant inputs violate the phase identities."""


@dataclass(frozen=True)
class _Rule:
    """A single derivation rule ``target = func(*requires)``."""

    target: str
    requires: Tuple[str, ...]
    func: Callable[..., float]
    formula: str
    guard: Optional[Callable[..., bool]] = None


#: Ordered rule base. The solver sweeps this list until a fixed point is
#: reached, so rule order only affects which formula is credited in the
#: derivation log, never the final values.
_RULES: Tuple[_Rule, ...] = (
    _Rule("e", ("n",), lambda n: n / (1.0 - n),
          "e = n / (1 - n)", lambda n: n < 1.0 - _EPS),
    _Rule("n", ("e",), lambda e: e / (1.0 + e),
          "n = e / (1 + e)"),
    _Rule("sr", ("w", "gs", "e"), lambda w, gs, e: w * gs / e,
          "Sr = w Gs / e", lambda w, gs, e: e > _EPS),
    _Rule("w", ("sr", "e", "gs"), lambda sr, e, gs: sr * e / gs,
          "w = Sr e / Gs"),
    _Rule("e", ("w", "gs", "sr"), lambda w, gs, sr: w * gs / sr,
          "e = w Gs / Sr", lambda w, gs, sr: sr > _EPS),
    _Rule("gamma_d", ("gs", "e", "gamma_w"),
          lambda gs, e, gw: gs * gw / (1.0 + e),
          "gamma_d = Gs gamma_w / (1 + e)"),
    _Rule("e", ("gs", "gamma_d", "gamma_w"),
          lambda gs, gd, gw: gs * gw / gd - 1.0,
          "e = Gs gamma_w / gamma_d - 1"),
    _Rule("gs", ("gamma_d", "e", "gamma_w"),
          lambda gd, e, gw: gd * (1.0 + e) / gw,
          "Gs = gamma_d (1 + e) / gamma_w"),
    _Rule("gamma", ("gamma_d", "w"), lambda gd, w: gd * (1.0 + w),
          "gamma = gamma_d (1 + w)"),
    _Rule("gamma_d", ("gamma", "w"), lambda g, w: g / (1.0 + w),
          "gamma_d = gamma / (1 + w)"),
    _Rule("w", ("gamma", "gamma_d"), lambda g, gd: g / gd - 1.0,
          "w = gamma / gamma_d - 1"),
    _Rule("gamma", ("gs", "sr", "e", "gamma_w"),
          lambda gs, sr, e, gw: (gs + sr * e) * gw / (1.0 + e),
          "gamma = (Gs + Sr e) gamma_w / (1 + e)"),
    _Rule("gamma_sat", ("gs", "e", "gamma_w"),
          lambda gs, e, gw: (gs + e) * gw / (1.0 + e),
          "gamma_sat = (Gs + e) gamma_w / (1 + e)"),
    _Rule("e", ("gs", "gamma_sat", "gamma_w"),
          lambda gs, gsat, gw: (gs * gw - gsat) / (gsat - gw),
          "e = (Gs gamma_w - gamma_sat) / (gamma_sat - gamma_w)",
          lambda gs, gsat, gw: gsat > gw + _EPS),
    # For a fully saturated soil the bulk and saturated unit weights match.
    _Rule("gamma_sat", ("gamma", "sr"), lambda g, sr: g,
          "gamma_sat = gamma  (Sr = 1)", lambda g, sr: sr >= 1.0 - 1e-9),
    _Rule("gamma", ("gamma_sat", "sr"), lambda gsat, sr: gsat,
          "gamma = gamma_sat  (Sr = 1)", lambda gsat, sr: sr >= 1.0 - 1e-9),
    _Rule("gamma_sub", ("gamma_sat", "gamma_w"),
          lambda gsat, gw: gsat - gw,
          "gamma' = gamma_sat - gamma_w"),
    _Rule("air_content", ("n", "sr"), lambda n, sr: n * (1.0 - sr),
          "a = n (1 - Sr)"),
)

#: Properties tracked by the solver (order used in reports).
_PROPERTIES: Tuple[str, ...] = (
    "gs", "w", "e", "n", "sr",
    "gamma", "gamma_d", "gamma_sat", "gamma_sub", "air_content",
)

_LABELS: Dict[str, str] = {
    "gs": "Gs   (specific gravity)          [-]",
    "w": "w    (water content)             [-]",
    "e": "e    (void ratio)                [-]",
    "n": "n    (porosity)                  [-]",
    "sr": "Sr   (degree of saturation)      [-]",
    "gamma": "gamma      (bulk unit weight)      [kN/m^3]",
    "gamma_d": "gamma_d    (dry unit weight)       [kN/m^3]",
    "gamma_sat": "gamma_sat  (saturated unit weight) [kN/m^3]",
    "gamma_sub": "gamma'     (buoyant unit weight)   [kN/m^3]",
    "air_content": "a    (volumetric air content)   [-]",
}


class Soil:
    """Phase-relationship state of a soil element.

    Provide any physically consistent subset of the parameters below;
    the constructor derives every property reachable through the phase
    identities and validates redundant inputs.

    Parameters
    ----------
    gs
        Specific gravity of solids :math:`G_s` [-].
    w
        Gravimetric water content :math:`w` (decimal) [-].
    e
        Void ratio :math:`e` [-].
    n
        Porosity :math:`n` (decimal) [-].
    sr
        Degree of saturation :math:`S_r` (decimal, 0-1) [-].
    gamma
        Bulk (moist) unit weight :math:`\\gamma` [kN/m^3].
    gamma_d
        Dry unit weight :math:`\\gamma_d` [kN/m^3].
    gamma_sat
        Saturated unit weight :math:`\\gamma_{sat}` [kN/m^3].
    gamma_w
        Unit weight of water [kN/m^3], default 9.81.
    rtol
        Relative tolerance used when cross-checking redundant inputs.

    Raises
    ------
    InconsistentInputError
        If redundant inputs disagree beyond ``rtol`` or a derived value
        is physically inadmissible (e.g. :math:`S_r > 1`).

    Examples
    --------
    >>> s = Soil(w=0.20, gs=2.70, e=0.80)
    >>> round(s.sr, 3)
    0.675
    >>> round(s.gamma_d, 3)
    14.715
    """

    def __init__(
        self,
        *,
        gs: Optional[float] = None,
        w: Optional[float] = None,
        e: Optional[float] = None,
        n: Optional[float] = None,
        sr: Optional[float] = None,
        gamma: Optional[float] = None,
        gamma_d: Optional[float] = None,
        gamma_sat: Optional[float] = None,
        gamma_w: float = GAMMA_W,
        rtol: float = 1e-2,
    ) -> None:
        self.gamma_w: float = float(gamma_w)
        self.rtol: float = float(rtol)
        self.derivation_log: List[str] = []

        inputs: Dict[str, Optional[float]] = {
            "gs": gs, "w": w, "e": e, "n": n, "sr": sr,
            "gamma": gamma, "gamma_d": gamma_d, "gamma_sat": gamma_sat,
        }
        self._known: Dict[str, float] = {
            key: float(val) for key, val in inputs.items() if val is not None
        }
        self._inputs: Tuple[str, ...] = tuple(self._known)

        self._validate(self._known, stage="input")
        self._solve()
        self._validate(self._known, stage="derived")

    # ------------------------------------------------------------------
    # Core solver
    # ------------------------------------------------------------------
    def _solve(self) -> None:
        """Sweep the rule base until no new property can be derived."""
        known = self._known
        known_ext = dict(known)
        known_ext["gamma_w"] = self.gamma_w

        changed = True
        while changed:
            changed = False
            for rule in _RULES:
                if any(req not in known_ext for req in rule.requires):
                    continue
                args = [known_ext[req] for req in rule.requires]
                if rule.guard is not None and not rule.guard(*args):
                    continue
                value = rule.func(*args)
                if rule.target in known_ext:
                    self._cross_check(rule, known_ext[rule.target], value)
                    continue
                known_ext[rule.target] = value
                known[rule.target] = value
                self.derivation_log.append(
                    f"{rule.target} = {value:.4f}  via  {rule.formula}"
                    f"  [from {', '.join(rule.requires)}]"
                )
                changed = True

    def _cross_check(self, rule: _Rule, existing: float, new: float) -> None:
        """Verify that a redundant derivation matches the stored value."""
        scale = max(abs(existing), abs(new), _EPS)
        if abs(existing - new) > self.rtol * scale:
            raise InconsistentInputError(
                f"Inconsistent phase data: rule '{rule.formula}' yields "
                f"{rule.target} = {new:.4f} but {rule.target} = "
                f"{existing:.4f} is already set "
                f"(relative difference {abs(existing - new) / scale:.2%} "
                f"> rtol = {self.rtol:.2%})."
            )

    def _validate(self, known: Dict[str, float], stage: str) -> None:
        """Range checks on inputs ('input') or the solved state ('derived')."""
        def bad(msg: str) -> None:
            raise InconsistentInputError(f"[{stage}] {msg}")

        tol = 1e-6
        if "w" in known and known["w"] < -tol:
            bad(f"water content w = {known['w']:.4f} cannot be negative.")
        if "e" in known and known["e"] <= 0.0:
            bad(f"void ratio e = {known['e']:.4f} must be positive.")
        if "n" in known and not 0.0 < known["n"] < 1.0:
            bad(f"porosity n = {known['n']:.4f} must lie in (0, 1).")
        if "sr" in known and not -tol <= known["sr"] <= 1.0 + 1e-3:
            bad(
                f"degree of saturation Sr = {known['sr']:.4f} must lie in "
                "[0, 1]; the given data imply an over-saturated state."
            )
        if "gs" in known and known["gs"] <= 1.0:
            bad(f"specific gravity Gs = {known['gs']:.4f} must exceed 1.")
        for key in ("gamma", "gamma_d", "gamma_sat"):
            if key in known and known[key] <= 0.0:
                bad(f"{key} = {known[key]:.4f} kN/m^3 must be positive.")
        if "gamma" in known and "gamma_d" in known:
            if known["gamma_d"] > known["gamma"] * (1.0 + self.rtol):
                bad(
                    f"gamma_d = {known['gamma_d']:.3f} exceeds "
                    f"gamma = {known['gamma']:.3f} kN/m^3."
                )

    # ------------------------------------------------------------------
    # Read-only access
    # ------------------------------------------------------------------
    def _get(self, name: str) -> Optional[float]:
        return self._known.get(name)

    @property
    def gs(self) -> Optional[float]:
        """Specific gravity of solids :math:`G_s` [-]."""
        return self._get("gs")

    @property
    def w(self) -> Optional[float]:
        """Water content :math:`w` (decimal) [-]."""
        return self._get("w")

    @property
    def e(self) -> Optional[float]:
        """Void ratio :math:`e = V_v / V_s` [-]."""
        return self._get("e")

    @property
    def n(self) -> Optional[float]:
        """Porosity :math:`n = e / (1 + e)` [-]."""
        return self._get("n")

    @property
    def sr(self) -> Optional[float]:
        """Degree of saturation :math:`S_r = w G_s / e` (decimal) [-]."""
        return self._get("sr")

    @property
    def gamma(self) -> Optional[float]:
        """Bulk unit weight :math:`\\gamma = \\gamma_d (1+w)` [kN/m^3]."""
        return self._get("gamma")

    @property
    def gamma_d(self) -> Optional[float]:
        """Dry unit weight :math:`\\gamma_d = G_s \\gamma_w/(1+e)` [kN/m^3]."""
        return self._get("gamma_d")

    @property
    def gamma_sat(self) -> Optional[float]:
        """Saturated unit weight :math:`(G_s+e)\\gamma_w/(1+e)` [kN/m^3]."""
        return self._get("gamma_sat")

    @property
    def gamma_sub(self) -> Optional[float]:
        """Buoyant unit weight :math:`\\gamma' = \\gamma_{sat}-\\gamma_w`."""
        return self._get("gamma_sub")

    @property
    def air_content(self) -> Optional[float]:
        """Volumetric air content :math:`a = n (1 - S_r)` [-]."""
        return self._get("air_content")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Optional[float]]:
        """Return every tracked property (``None`` if underdetermined)."""
        return {name: self._get(name) for name in _PROPERTIES}

    def report(self) -> str:
        """Human-readable report of inputs, results and derivation steps."""
        lines: List[str] = ["Soil phase relations", "-" * 60]
        for name in _PROPERTIES:
            value = self._get(name)
            origin = "input" if name in self._inputs else (
                "derived" if value is not None else "unresolved"
            )
            text = f"{value:.4f}" if value is not None else "-"
            lines.append(f"{_LABELS[name]:<46s} = {text:>9s}  ({origin})")
        if self.derivation_log:
            lines.append("-" * 60)
            lines.append("Derivation trace:")
            lines.extend(f"  {step}" for step in self.derivation_log)
        return "\n".join(lines)

    def __repr__(self) -> str:
        parts = ", ".join(
            f"{name}={self._known[name]:.4g}"
            for name in _PROPERTIES if name in self._known
        )
        return f"Soil({parts})"


def void_ratio_from_porosity(n: float) -> float:
    """Return :math:`e = n / (1 - n)` for a porosity ``n`` in (0, 1)."""
    if not 0.0 < n < 1.0:
        raise ValueError(f"porosity n = {n} must lie in (0, 1).")
    return n / (1.0 - n)


def porosity_from_void_ratio(e: float) -> float:
    """Return :math:`n = e / (1 + e)` for a void ratio ``e`` > 0."""
    if e <= 0.0 or not math.isfinite(e):
        raise ValueError(f"void ratio e = {e} must be a positive number.")
    return e / (1.0 + e)
