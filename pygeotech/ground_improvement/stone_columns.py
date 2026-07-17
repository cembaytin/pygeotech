"""Stone-column ground improvement.

The area replacement ratio is :math:`a_s = A_c/A` (column area over the
tributary cell area). Two settlement-improvement estimates are provided:

* **Equilibrium (stress-concentration) method:** with a stress
  concentration ratio :math:`n = \\sigma_c/\\sigma_s`, the settlement
  improvement factor is :math:`\\beta = 1 + a_s(n - 1)` and the treated
  settlement is :math:`s_{treated} = s_{untreated}/\\beta`.

* **Priebe (1995) basic improvement factor** :math:`n_0`, for a rigid
  column material with friction angle :math:`\\phi_c`:

  .. math::

      n_0 = 1 + a_s\\left[\\frac{0.5 + f(\\nu_s, a_s)}
      {K_{ac}\\,f(\\nu_s, a_s)} - 1\\right],
      \\quad K_{ac} = \\tan^2\\!\\left(45 - \\tfrac{\\phi_c}{2}\\right),

  with :math:`f(\\nu_s, a_s) = \\dfrac{(1-\\nu_s)(1-a_s)}
  {1 - 2\\nu_s + a_s}`.
"""

from __future__ import annotations

import math

__all__ = [
    "area_replacement_ratio",
    "settlement_improvement_equilibrium",
    "priebe_improvement_factor",
]


def area_replacement_ratio(column_diameter: float, spacing: float,
                           pattern: str = "triangular") -> float:
    """Area replacement ratio :math:`a_s = A_c/A` for a column grid."""
    ac = math.pi * column_diameter ** 2 / 4.0
    if pattern == "triangular":
        cell = (math.sqrt(3.0) / 2.0) * spacing ** 2
    elif pattern == "square":
        cell = spacing ** 2
    else:
        raise ValueError("pattern must be 'triangular' or 'square'.")
    return ac / cell


def settlement_improvement_equilibrium(
    area_ratio: float, stress_concentration: float
) -> float:
    """Settlement improvement factor :math:`\\beta = 1 + a_s(n-1)`."""
    if not 0.0 < area_ratio < 1.0:
        raise ValueError("area_ratio must lie in (0, 1).")
    return 1.0 + area_ratio * (stress_concentration - 1.0)


def priebe_improvement_factor(
    area_ratio: float,
    column_friction_angle: float,
    poisson_ratio: float = 1.0 / 3.0,
) -> float:
    """Priebe (1995) basic improvement factor :math:`n_0`.

    Parameters
    ----------
    area_ratio
        Area replacement ratio :math:`a_s` (0-1).
    column_friction_angle
        Friction angle of the column material :math:`\\phi_c` [deg].
    poisson_ratio
        Poisson's ratio of the surrounding soil (Priebe uses 1/3).

    Examples
    --------
    >>> round(priebe_improvement_factor(0.2, 40.0), 2)
    2.18
    """
    if not 0.0 < area_ratio < 1.0:
        raise ValueError("area_ratio must lie in (0, 1).")
    nu = poisson_ratio
    f = (1.0 - nu) * (1.0 - area_ratio) / (1.0 - 2.0 * nu + area_ratio)
    k_ac = math.tan(math.radians(45.0 - column_friction_angle / 2.0)) ** 2
    return 1.0 + area_ratio * ((0.5 + f) / (k_ac * f) - 1.0)
