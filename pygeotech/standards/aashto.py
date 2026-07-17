"""AASHTO LRFD design adapters (load and resistance factor design).

Nominal (characteristic) resistances from the mechanics cores are reduced
by a resistance factor :math:`\\phi`, while loads are amplified by load
factors :math:`\\gamma` and combined per a limit-state group. The factored
resistance must exceed the factored load:

.. math:: \\phi\\,R_n \\ge \\sum \\gamma_i Q_i .

Representative resistance factors (AASHTO LRFD Bridge Design
Specifications) are provided for common geotechnical limit states; the
Strength I load combination is implemented for the load side.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ResistanceFactors",
    "AASHTO_RESISTANCE",
    "factored_resistance",
    "strength_i_load",
]


@dataclass(frozen=True)
class ResistanceFactors:
    """Representative AASHTO LRFD geotechnical resistance factors."""

    bearing_shallow: float = 0.45         # spread footing bearing
    sliding: float = 0.80                 # base sliding (cohesionless)
    pile_side_alpha: float = 0.35         # driven pile skin (alpha method)
    pile_tip: float = 0.40                # driven pile end bearing
    drilled_shaft_side: float = 0.45
    drilled_shaft_tip: float = 0.40


#: Default resistance-factor set.
AASHTO_RESISTANCE = ResistanceFactors()


def factored_resistance(nominal_resistance: float, resistance_factor: float) -> float:
    """Factored resistance :math:`\\phi R_n`."""
    if not 0.0 < resistance_factor <= 1.0:
        raise ValueError("resistance_factor must lie in (0, 1].")
    return resistance_factor * nominal_resistance


def strength_i_load(
    dead_load: float,
    wearing_surface: float = 0.0,
    live_load: float = 0.0,
    minimum: bool = False,
) -> float:
    """Factored load for the AASHTO Strength I combination.

    Parameters
    ----------
    dead_load
        Structural dead load :math:`DC`.
    wearing_surface
        Wearing-surface / utility load :math:`DW`.
    live_load
        Vehicular live load :math:`LL + IM`.
    minimum
        If ``True``, use the minimum (favourable) permanent load factors
        (0.90 for DC, 0.65 for DW); otherwise the maximum (1.25, 1.50).
    """
    if minimum:
        gamma_dc, gamma_dw = 0.90, 0.65
    else:
        gamma_dc, gamma_dw = 1.25, 1.50
    return gamma_dc * dead_load + gamma_dw * wearing_surface + 1.75 * live_load
