"""Radial consolidation by prefabricated vertical drains (PVD).

Hansbo's (1981) solution for radial drainage to a vertical drain in a
unit cell of influence diameter :math:`d_e`:

.. math::

    U_h = 1 - \\exp\\!\\left(-\\frac{8 T_h}{F(n)}\\right),
    \\qquad T_h = \\frac{c_h\\,t}{d_e^2},

with the drain-spacing factor accounting for smear and well resistance,

.. math::

    F(n) = \\ln\\!\\frac{n}{s} + \\frac{k_h}{k_s}\\ln s - \\frac{3}{4},
    \\qquad n = \\frac{d_e}{d_w},\\; s = \\frac{d_s}{d_w}.

Vertical and radial consolidation combine by Carrillo's theorem,
:math:`U = 1 - (1 - U_v)(1 - U_h)`. Influence diameter:
:math:`d_e = 1.05\\,S` (triangular) or :math:`1.13\\,S` (square grid).
"""

from __future__ import annotations

import math

__all__ = [
    "drain_influence_diameter",
    "hansbo_factor",
    "radial_time_factor",
    "radial_degree_of_consolidation",
    "combined_degree_of_consolidation",
]


def drain_influence_diameter(spacing: float, pattern: str = "triangular") -> float:
    """Unit-cell influence diameter :math:`d_e` for a drain grid [m]."""
    if pattern == "triangular":
        return 1.05 * spacing
    if pattern == "square":
        return 1.13 * spacing
    raise ValueError("pattern must be 'triangular' or 'square'.")


def hansbo_factor(n: float, smear_ratio: float = 1.0, kh_ks: float = 1.0) -> float:
    """Hansbo spacing factor :math:`F(n)` with smear.

    Parameters
    ----------
    n
        Diameter ratio :math:`n = d_e/d_w`.
    smear_ratio
        Smear-zone ratio :math:`s = d_s/d_w` (1 = no smear).
    kh_ks
        Ratio of undisturbed to smear-zone horizontal permeability
        :math:`k_h/k_s`.
    """
    if n <= smear_ratio:
        raise ValueError("n must exceed the smear ratio s.")
    return math.log(n / smear_ratio) + kh_ks * math.log(smear_ratio) - 0.75


def radial_time_factor(ch: float, t: float, de: float) -> float:
    """Radial time factor :math:`T_h = c_h t / d_e^2`."""
    return ch * t / de ** 2


def radial_degree_of_consolidation(time_factor: float, f_n: float) -> float:
    """Average radial degree of consolidation :math:`U_h`."""
    if time_factor <= 0.0:
        return 0.0
    return 1.0 - math.exp(-8.0 * time_factor / f_n)


def combined_degree_of_consolidation(u_vertical: float, u_radial: float) -> float:
    """Combined degree of consolidation (Carrillo): ``1-(1-Uv)(1-Uh)``."""
    return 1.0 - (1.0 - u_vertical) * (1.0 - u_radial)
