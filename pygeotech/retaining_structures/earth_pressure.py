"""Lateral earth pressures and retaining-wall stability.

Earth-pressure coefficients
---------------------------
Rankine (with a backfill slope :math:`\\beta`):

.. math::

    K_a = \\cos\\beta\\,
    \\frac{\\cos\\beta - \\sqrt{\\cos^2\\beta - \\cos^2\\phi}}
         {\\cos\\beta + \\sqrt{\\cos^2\\beta - \\cos^2\\phi}},
    \\qquad
    K_a = \\tan^2\\!\\left(45 - \\tfrac{\\phi}{2}\\right)\\;(\\beta = 0),

with the passive coefficient obtained by swapping the numerator and
denominator (and :math:`K_p = \\tan^2(45 + \\phi/2)` for level backfill).

Coulomb (wall friction :math:`\\delta`, wall batter :math:`\\theta` from
vertical, backfill slope :math:`\\beta`):

.. math::

    K_a = \\frac{\\cos^2(\\phi - \\theta)}
    {\\cos^2\\theta\\,\\cos(\\delta + \\theta)
    \\left[1 + \\sqrt{\\dfrac{\\sin(\\phi + \\delta)\\sin(\\phi - \\beta)}
    {\\cos(\\delta + \\theta)\\cos(\\theta - \\beta)}}\\right]^2}.

For a cohesive backfill the Rankine active/passive pressures include the
:math:`\\mp 2c\\sqrt{K}` cohesion terms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

__all__ = [
    "rankine_active_coefficient",
    "rankine_passive_coefficient",
    "coulomb_active_coefficient",
    "coulomb_passive_coefficient",
    "active_thrust",
    "GravityWall",
    "WallStability",
]


def rankine_active_coefficient(phi: float, beta: float = 0.0) -> float:
    """Rankine active earth-pressure coefficient :math:`K_a`.

    Examples
    --------
    >>> round(rankine_active_coefficient(30.0), 3)
    0.333
    """
    if beta == 0.0:
        return math.tan(math.radians(45.0 - phi / 2.0)) ** 2
    b = math.radians(beta)
    p = math.radians(phi)
    if math.cos(b) ** 2 < math.cos(p) ** 2:
        raise ValueError("backfill slope beta cannot exceed phi.")
    root = math.sqrt(math.cos(b) ** 2 - math.cos(p) ** 2)
    return math.cos(b) * (math.cos(b) - root) / (math.cos(b) + root)


def rankine_passive_coefficient(phi: float, beta: float = 0.0) -> float:
    """Rankine passive earth-pressure coefficient :math:`K_p`."""
    if beta == 0.0:
        return math.tan(math.radians(45.0 + phi / 2.0)) ** 2
    b = math.radians(beta)
    p = math.radians(phi)
    if math.cos(b) ** 2 < math.cos(p) ** 2:
        raise ValueError("backfill slope beta cannot exceed phi.")
    root = math.sqrt(math.cos(b) ** 2 - math.cos(p) ** 2)
    return math.cos(b) * (math.cos(b) + root) / (math.cos(b) - root)


def coulomb_active_coefficient(
    phi: float, delta: float = 0.0, beta: float = 0.0, theta: float = 0.0
) -> float:
    """Coulomb active earth-pressure coefficient :math:`K_a`.

    Parameters
    ----------
    phi
        Soil friction angle [degrees].
    delta
        Wall friction angle [degrees].
    beta
        Backfill slope from horizontal [degrees].
    theta
        Wall inclination from vertical [degrees] (positive when the wall
        leans into the backfill).
    """
    p, d = math.radians(phi), math.radians(delta)
    b, t = math.radians(beta), math.radians(theta)
    numerator = math.cos(p - t) ** 2
    root_arg = (math.sin(p + d) * math.sin(p - b)
                / (math.cos(d + t) * math.cos(t - b)))
    if root_arg < 0.0:
        raise ValueError("invalid geometry (negative radicand); check "
                         "phi, delta, beta, theta.")
    bracket = (1.0 + math.sqrt(root_arg)) ** 2
    denominator = math.cos(t) ** 2 * math.cos(d + t) * bracket
    return numerator / denominator


def coulomb_passive_coefficient(
    phi: float, delta: float = 0.0, beta: float = 0.0, theta: float = 0.0
) -> float:
    """Coulomb passive earth-pressure coefficient :math:`K_p`."""
    p, d = math.radians(phi), math.radians(delta)
    b, t = math.radians(beta), math.radians(theta)
    numerator = math.cos(p + t) ** 2
    root_arg = (math.sin(p + d) * math.sin(p + b)
                / (math.cos(d - t) * math.cos(t - b)))
    if root_arg < 0.0:
        raise ValueError("invalid geometry (negative radicand).")
    bracket = (1.0 - math.sqrt(root_arg)) ** 2
    denominator = math.cos(t) ** 2 * math.cos(d - t) * bracket
    return numerator / denominator


def active_thrust(
    height: float,
    gamma: float,
    phi: float,
    cohesion: float = 0.0,
    surcharge: float = 0.0,
    water_table_depth: Optional[float] = None,
    gamma_w: float = 9.81,
) -> Tuple[float, float]:
    """Total active thrust on a smooth vertical wall (Rankine).

    Combines soil, surcharge, cohesion and (optionally) hydrostatic water
    pressure. A tension crack is assumed for the negative cohesion zone
    (pressures are clipped at zero).

    Parameters
    ----------
    height
        Wall height :math:`H` [m].
    gamma
        Backfill unit weight [kN/m^3] (use the saturated value below the
        water table).
    phi, cohesion
        Backfill strength parameters [degrees, kPa].
    surcharge
        Uniform surface surcharge :math:`q` [kPa].
    water_table_depth
        Depth of the water table behind the wall [m]; ``None`` for a dry
        backfill.
    gamma_w
        Unit weight of water [kN/m^3].

    Returns
    -------
    (thrust, line_of_action)
        Total horizontal thrust [kN/m] and the height of its resultant
        above the base [m].
    """
    ka = rankine_active_coefficient(phi)
    sqrt_ka = math.sqrt(ka)
    n = 200
    z = [height * i / n for i in range(n + 1)]
    dz = height / n
    gamma_w_eff = gamma_w if water_table_depth is not None else 0.0
    dw = water_table_depth if water_table_depth is not None else math.inf

    def sigma_h(depth: float) -> float:
        sigma_v = surcharge + gamma * depth
        if depth > dw:
            # effective vertical stress uses buoyant weight below WT
            sigma_v = surcharge + gamma * dw + (gamma - gamma_w_eff) * (
                depth - dw)
        pressure = ka * sigma_v - 2.0 * cohesion * sqrt_ka
        u = gamma_w_eff * max(0.0, depth - dw) if math.isfinite(dw) else 0.0
        return max(0.0, pressure) + u

    # Trapezoidal integration of the pressure diagram and its moment.
    thrust = 0.0
    moment = 0.0
    for i in range(n):
        p0, p1 = sigma_h(z[i]), sigma_h(z[i + 1])
        seg = 0.5 * (p0 + p1) * dz
        # centroid height above base of this trapezoidal slice
        lever = height - (z[i] + dz * (p0 + 2.0 * p1) / (3.0 * (p0 + p1))
                          if (p0 + p1) > 0 else z[i] + dz / 2.0)
        thrust += seg
        moment += seg * lever
    line = moment / thrust if thrust > 0 else 0.0
    return thrust, line


@dataclass(frozen=True)
class WallStability:
    """Stability check results for a gravity retaining wall."""

    fs_sliding: float
    fs_overturning: float
    bearing_pressure_max: float
    bearing_pressure_min: float
    resultant_eccentricity: float

    def __str__(self) -> str:
        return (f"FS_sliding = {self.fs_sliding:.2f} | "
                f"FS_overturning = {self.fs_overturning:.2f} | "
                f"q_max = {self.bearing_pressure_max:.1f} kPa | "
                f"e = {self.resultant_eccentricity:.3f} m")


class GravityWall:
    """A gravity retaining wall checked for external stability.

    Parameters
    ----------
    height
        Total wall height :math:`H` [m].
    base_width
        Base slab width :math:`B` [m].
    weight
        Total weight of the wall (and any soil on the heel) per metre run
        [kN/m].
    weight_arm
        Horizontal distance from the toe to the line of action of
        ``weight`` [m].
    base_friction_angle
        Friction angle between the base and the foundation soil [degrees].
    """

    def __init__(
        self,
        height: float,
        base_width: float,
        weight: float,
        weight_arm: float,
        base_friction_angle: float,
    ) -> None:
        self.height = height
        self.base_width = base_width
        self.weight = weight
        self.weight_arm = weight_arm
        self.base_friction_angle = base_friction_angle

    def check(
        self,
        horizontal_thrust: float,
        thrust_arm: float,
        vertical_thrust: float = 0.0,
    ) -> WallStability:
        """Evaluate sliding, overturning and base-pressure stability.

        Parameters
        ----------
        horizontal_thrust
            Horizontal component of the active thrust [kN/m].
        thrust_arm
            Height of the thrust resultant above the base [m].
        vertical_thrust
            Vertical component of the thrust (e.g. from wall friction or a
            sloped backfill), acting at the heel [kN/m].
        """
        normal = self.weight + vertical_thrust
        # Sliding.
        resistance = normal * math.tan(math.radians(self.base_friction_angle))
        fs_sliding = resistance / horizontal_thrust if horizontal_thrust else math.inf
        # Overturning about the toe.
        m_over = horizontal_thrust * thrust_arm
        m_resist = (self.weight * self.weight_arm
                    + vertical_thrust * self.base_width)
        fs_over = m_resist / m_over if m_over else math.inf
        # Base pressure (Meyerhof) from the resultant eccentricity.
        x_resultant = (m_resist - m_over) / normal
        eccentricity = self.base_width / 2.0 - x_resultant
        b = self.base_width
        q_avg = normal / b
        q_max = q_avg * (1.0 + 6.0 * eccentricity / b)
        q_min = q_avg * (1.0 - 6.0 * eccentricity / b)
        return WallStability(
            fs_sliding=fs_sliding,
            fs_overturning=fs_over,
            bearing_pressure_max=q_max,
            bearing_pressure_min=q_min,
            resultant_eccentricity=eccentricity,
        )

    def __repr__(self) -> str:
        return (f"GravityWall(H={self.height} m, B={self.base_width} m, "
                f"W={self.weight} kN/m)")
