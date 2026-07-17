"""Circular slip-surface slope stability by the method of slices.

The sliding mass above a trial circular surface is divided into vertical
slices; for slice ``i`` with base inclination :math:`\\alpha_i`, width
:math:`b_i`, base length :math:`l_i = b_i/\\cos\\alpha_i`, weight
:math:`W_i` and pore pressure :math:`u_i`:

* **Ordinary / Fellenius** (moment equilibrium, no interslice forces):

  .. math:: F = \\frac{\\sum [c'\\,l_i + (W_i\\cos\\alpha_i - u_i l_i)\\tan\\phi']}
                     {\\sum W_i\\sin\\alpha_i}.

* **Bishop simplified** (horizontal interslice forces, iterative):

  .. math:: F = \\frac{\\sum \\dfrac{c' b_i + (W_i - u_i b_i)\\tan\\phi'}{m_{\\alpha,i}}}
                     {\\sum W_i\\sin\\alpha_i},
      \\quad m_{\\alpha,i} = \\cos\\alpha_i + \\frac{\\sin\\alpha_i\\tan\\phi'}{F}.

* **Janbu simplified** (force equilibrium) with an empirical correction
  factor :math:`f_0`.

Pore pressure is specified through the pore-pressure ratio
:math:`r_u = u/(\\gamma h)`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

__all__ = [
    "SlipCircle",
    "simple_slope_surface",
    "slope_factor_of_safety",
    "critical_circle",
]


@dataclass(frozen=True)
class SlipCircle:
    """A trial circular slip surface (centre ``(xc, yc)``, ``radius``)."""

    xc: float
    yc: float
    radius: float


def simple_slope_surface(
    height: float, slope_angle: float
) -> Callable[[np.ndarray], np.ndarray]:
    """Ground-surface profile of a simple slope (toe at the origin).

    The toe sits at ``(0, 0)``; the face rises to the crest at
    ``(H/tan(beta), H)`` and the ground is horizontal beyond the toe
    (``y=0`` for ``x<0``) and the crest (``y=H``).

    Parameters
    ----------
    height
        Slope height :math:`H` [m].
    slope_angle
        Face inclination :math:`\\beta` [deg].

    Returns
    -------
    callable
        A vectorised function ``y_ground(x)``.
    """
    x_crest = height / math.tan(math.radians(slope_angle))

    def ground_y(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.clip(x, 0.0, x_crest) * (height / x_crest)

    return ground_y


def _slices(
    ground_y: Callable[[np.ndarray], np.ndarray],
    circle: SlipCircle,
    n_slices: int,
) -> Optional[Tuple[np.ndarray, ...]]:
    """Return per-slice ``(b, h, alpha, x_mid)`` arrays, or ``None``.

    ``None`` is returned if the circle does not form a valid sliding mass
    (does not dip below the ground surface).
    """
    xc, yc, r = circle.xc, circle.yc, circle.radius
    xs = np.linspace(xc - r, xc + r, 400)
    with np.errstate(invalid="ignore"):
        y_circle = yc - np.sqrt(np.maximum(r ** 2 - (xs - xc) ** 2, 0.0))
    h = ground_y(xs) - y_circle
    inside = h > 1e-9
    if not inside.any():
        return None
    idx = np.where(inside)[0]
    x_left, x_right = xs[idx[0]], xs[idx[-1]]
    if x_right - x_left < 1e-6:
        return None

    edges = np.linspace(x_left, x_right, n_slices + 1)
    x_mid = 0.5 * (edges[:-1] + edges[1:])
    b = np.diff(edges)
    sin_a = (x_mid - xc) / r
    if np.any(np.abs(sin_a) >= 1.0):
        return None
    alpha = np.arcsin(sin_a)
    y_circle_mid = yc - np.sqrt(np.maximum(r ** 2 - (x_mid - xc) ** 2, 0.0))
    h_mid = ground_y(x_mid) - y_circle_mid
    if np.any(h_mid <= 0.0):
        h_mid = np.maximum(h_mid, 0.0)
    return b, h_mid, alpha, x_mid


def slope_factor_of_safety(
    ground_y: Callable[[np.ndarray], np.ndarray],
    circle: SlipCircle,
    gamma: float,
    cohesion: float,
    friction_angle: float,
    ru: float = 0.0,
    n_slices: int = 50,
    method: str = "bishop",
    f0: float = 1.0,
    seismic_coefficient: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    """Factor of safety of a slope for a given circular slip surface.

    Parameters
    ----------
    ground_y
        Vectorised ground-surface function ``y(x)`` (e.g. from
        :func:`simple_slope_surface`).
    circle
        Trial :class:`SlipCircle`.
    gamma, cohesion, friction_angle
        Homogeneous soil unit weight [kN/m^3], cohesion [kPa] and
        friction angle [deg].
    ru
        Pore-pressure ratio :math:`r_u = u/(\\gamma h)`.
    n_slices
        Number of slices.
    method
        ``"fellenius"``, ``"bishop"`` or ``"janbu"``.
    f0
        Janbu correction factor (used only for ``method="janbu"``).
    seismic_coefficient
        Horizontal pseudo-static seismic coefficient :math:`k_h`; the
        inertia force :math:`k_h W_i` at each slice adds to the driving
        moment about the circle centre.

    Returns
    -------
    float
        Factor of safety (``inf`` if the mass is not kinematically
        driving).

    Raises
    ------
    ValueError
        If the circle does not intersect the slope.
    """
    data = _slices(ground_y, circle, n_slices)
    if data is None:
        raise ValueError("the slip circle does not intersect the slope.")
    b, h, alpha, x_mid = data
    phi = math.radians(friction_angle)
    tan_phi = math.tan(phi)

    weight = gamma * h * b
    length = b / np.cos(alpha)
    u = ru * gamma * h                      # pore pressure on the base
    driving = float(np.sum(weight * np.sin(alpha)))

    # Pseudo-static seismic: horizontal inertia k_h W adds a driving moment
    # k_h W (yc - y_cg) about the centre; divide by R to match W sin(alpha).
    if seismic_coefficient != 0.0:
        y_base = circle.yc - np.sqrt(
            np.maximum(circle.radius ** 2 - (x_mid - circle.xc) ** 2, 0.0))
        y_cg = 0.5 * (np.asarray(ground_y(x_mid), dtype=float) + y_base)
        seismic_moment = float(np.sum(
            seismic_coefficient * weight * (circle.yc - y_cg)))
        driving += seismic_moment / circle.radius

    if driving <= 0.0:
        return math.inf

    if method == "fellenius":
        normal = weight * np.cos(alpha) - u * length
        resisting = cohesion * length + np.maximum(normal, 0.0) * tan_phi
        return float(np.sum(resisting)) / driving

    if method in ("bishop", "janbu"):
        numer_const = cohesion * b + (weight - u * b) * tan_phi
        fos = 1.0
        for _ in range(max_iter):
            m_alpha = np.cos(alpha) + np.sin(alpha) * tan_phi / fos
            if method == "bishop":
                fos_new = float(np.sum(numer_const / m_alpha)) / driving
            else:  # Janbu simplified: force equilibrium
                n_alpha = np.cos(alpha) * m_alpha
                shear_sum = float(np.sum(weight * np.tan(alpha)))
                fos_new = f0 * float(np.sum(numer_const / n_alpha)) / shear_sum
            if abs(fos_new - fos) < tol:
                return fos_new
            fos = fos_new
        return fos
    raise ValueError("method must be 'fellenius', 'bishop' or 'janbu'.")


def critical_circle(
    ground_y: Callable[[np.ndarray], np.ndarray],
    gamma: float,
    cohesion: float,
    friction_angle: float,
    height: float,
    slope_angle: float,
    ru: float = 0.0,
    method: str = "bishop",
    n_centers: int = 12,
    n_radii: int = 12,
    n_slices: int = 40,
) -> Tuple[float, SlipCircle]:
    """Grid-search the critical (minimum-FoS) circle for a simple slope.

    The search spans centres above and behind the crest and radii large
    enough to pass near the toe. Returns ``(min_fos, circle)``.
    """
    x_crest = height / math.tan(math.radians(slope_angle))
    xc_range = np.linspace(0.2 * x_crest, 1.5 * x_crest, n_centers)
    yc_range = np.linspace(1.1 * height, 2.5 * height, n_centers)

    best_fos = math.inf
    best_circle: Optional[SlipCircle] = None
    for xc in xc_range:
        for yc in yc_range:
            # radii that bracket the toe region
            dist_to_toe = math.hypot(xc - 0.0, yc - 0.0)
            for r in np.linspace(0.6 * dist_to_toe, 1.05 * dist_to_toe,
                                 n_radii):
                circle = SlipCircle(float(xc), float(yc), float(r))
                try:
                    fos = slope_factor_of_safety(
                        ground_y, circle, gamma, cohesion, friction_angle,
                        ru=ru, n_slices=n_slices, method=method)
                except ValueError:
                    continue
                if math.isfinite(fos) and 0.0 < fos < best_fos:
                    best_fos = fos
                    best_circle = circle
    if best_circle is None:
        raise ValueError("no valid slip circle found in the search grid.")
    return best_fos, best_circle
