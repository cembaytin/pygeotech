"""Advanced limit-equilibrium slope stability.

Extends the method of slices to **layered soils**, **non-circular slip
surfaces** and **Spencer's method** (which satisfies both force and moment
equilibrium).

Spencer (1967) assumes the interslice force resultants act at a single
inclination :math:`\\theta`. For each slice the resultant is

.. math::

    Q_i = \\frac{\\dfrac{1}{F}\\big[c'_i \\ell_i
        + (W_i\\cos\\alpha_i - u_i \\ell_i)\\tan\\phi'_i\\big]
        - W_i\\sin\\alpha_i}
        {\\cos(\\alpha_i-\\theta) + \\dfrac{\\sin(\\alpha_i-\\theta)\\tan\\phi'_i}{F}} ,

and the factor of safety :math:`F` and angle :math:`\\theta` are found by
enforcing force equilibrium :math:`\\sum Q_i = 0` and (for circular
surfaces) moment equilibrium :math:`\\sum Q_i\\cos(\\alpha_i-\\theta)=0`
simultaneously. For a circular surface Spencer's result is very close to
Bishop's simplified value, which is used to validate the implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from pygeotech.slope_stability.slices import SlipCircle

__all__ = [
    "SlopeLayer",
    "LayeredSoil",
    "PolylineSurface",
    "advanced_factor_of_safety",
]


@dataclass(frozen=True)
class SlopeLayer:
    """A soil layer above the elevation ``top`` (down to the next layer)."""

    top: float                  # elevation of the layer's top surface [m]
    gamma: float                # unit weight [kN/m^3]
    cohesion: float             # effective cohesion c' [kPa]
    friction_angle: float       # effective friction angle phi' [deg]


class LayeredSoil:
    """A stack of :class:`SlopeLayer` ordered by elevation.

    A homogeneous slope is a single layer with ``top = +inf``.
    """

    def __init__(self, layers: Sequence[SlopeLayer]) -> None:
        if not layers:
            raise ValueError("at least one layer is required.")
        self.layers: List[SlopeLayer] = sorted(layers, key=lambda l: -l.top)

    @classmethod
    def homogeneous(cls, gamma: float, cohesion: float,
                    friction_angle: float) -> "LayeredSoil":
        """Build a single-layer (homogeneous) soil."""
        return cls([SlopeLayer(math.inf, gamma, cohesion, friction_angle)])

    def _layer_bottom(self, idx: int) -> float:
        return (self.layers[idx + 1].top if idx + 1 < len(self.layers)
                else -math.inf)

    def weight_per_width(self, y_top: float, y_bottom: float) -> float:
        """Weight per unit width of a vertical strip from ``y_top`` down."""
        total = 0.0
        for idx, layer in enumerate(self.layers):
            hi = min(y_top, layer.top)
            lo = max(y_bottom, self._layer_bottom(idx))
            if hi > lo:
                total += layer.gamma * (hi - lo)
        return total

    def strength_at(self, y: float) -> Tuple[float, float]:
        """Return ``(cohesion, friction_angle)`` of the layer at elevation ``y``."""
        for idx, layer in enumerate(self.layers):
            if self._layer_bottom(idx) - 1e-9 <= y <= layer.top + 1e-9:
                return layer.cohesion, layer.friction_angle
        return self.layers[-1].cohesion, self.layers[-1].friction_angle


class PolylineSurface:
    """A non-circular slip surface defined by ordered ``(x, y)`` points."""

    def __init__(self, points: Sequence[Tuple[float, float]]) -> None:
        pts = np.asarray(points, dtype=float)
        order = np.argsort(pts[:, 0])
        self.x = pts[order, 0]
        self.y = pts[order, 1]
        if self.x.size < 2:
            raise ValueError("a polyline surface needs at least two points.")

    @property
    def x_range(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[-1])

    def base_y(self, x: np.ndarray) -> np.ndarray:
        return np.interp(x, self.x, self.y)

    def base_angle(self, x: np.ndarray) -> np.ndarray:
        # Local segment inclination at each x.
        slopes = np.gradient(self.y, self.x)
        return np.arctan(np.interp(x, self.x, slopes))


Surface = Union[SlipCircle, PolylineSurface]


def _build_slices(
    ground_y: Callable[[np.ndarray], np.ndarray],
    surface: Surface,
    n_slices: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return ``(x_mid, width, y_ground, y_base, alpha)`` or ``None``."""
    if isinstance(surface, SlipCircle):
        xc, yc, r = surface.xc, surface.yc, surface.radius
        xs = np.linspace(xc - r, xc + r, 400)
        with np.errstate(invalid="ignore"):
            yb = yc - np.sqrt(np.maximum(r ** 2 - (xs - xc) ** 2, 0.0))
        inside = ground_y(xs) - yb > 1e-9
        if not inside.any():
            return None
        idx = np.where(inside)[0]
        x_left, x_right = xs[idx[0]], xs[idx[-1]]

        def base_y(x):
            return yc - np.sqrt(np.maximum(r ** 2 - (x - xc) ** 2, 0.0))

        def base_angle(x):
            return np.arcsin(np.clip((x - xc) / r, -1.0, 1.0))
    else:
        x_left, x_right = surface.x_range
        base_y = surface.base_y
        base_angle = surface.base_angle

    if x_right - x_left < 1e-6:
        return None
    edges = np.linspace(x_left, x_right, n_slices + 1)
    x_mid = 0.5 * (edges[:-1] + edges[1:])
    width = np.diff(edges)
    y_ground = np.asarray(ground_y(x_mid), dtype=float)
    y_base = np.asarray(base_y(x_mid), dtype=float)
    alpha = np.asarray(base_angle(x_mid), dtype=float)
    height = y_ground - y_base
    if np.any(height <= 0):
        height = np.maximum(height, 0.0)
    return x_mid, width, y_ground, y_base, alpha


def advanced_factor_of_safety(
    ground_y: Callable[[np.ndarray], np.ndarray],
    surface: Surface,
    soil: LayeredSoil,
    ru: float = 0.0,
    n_slices: int = 50,
    method: str = "spencer",
    max_iter: int = 200,
    tol: float = 1e-6,
) -> float:
    """Factor of safety for a layered slope and a general slip surface.

    Parameters
    ----------
    ground_y
        Ground-surface function ``y(x)``.
    surface
        A :class:`~pygeotech.slope_stability.slices.SlipCircle` or a
        :class:`PolylineSurface`.
    soil
        A :class:`LayeredSoil` (use :meth:`LayeredSoil.homogeneous` for a
        single soil).
    ru
        Pore-pressure ratio.
    method
        ``"fellenius"``, ``"bishop"`` (circular only), ``"janbu"`` or
        ``"spencer"``.

    Returns
    -------
    float
        Factor of safety.
    """
    data = _build_slices(ground_y, surface, n_slices)
    if data is None:
        raise ValueError("the slip surface does not intersect the slope.")
    x_mid, width, y_ground, y_base, alpha = data

    height = np.maximum(y_ground - y_base, 0.0)
    weight = np.array([soil.weight_per_width(yt, yb) * b
                       for yt, yb, b in zip(y_ground, y_base, width)])
    strengths = [soil.strength_at(yb) for yb in y_base]
    cohesion = np.array([c for c, _ in strengths])
    tan_phi = np.array([math.tan(math.radians(p)) for _, p in strengths])
    length = width / np.cos(alpha)
    u = ru * np.array([soil.weight_per_width(yt, yb) / max(yt - yb, 1e-9)
                       for yt, yb in zip(y_ground, y_base)]) * height
    driving = float(np.sum(weight * np.sin(alpha)))
    if driving <= 0.0:
        return math.inf

    if method == "fellenius":
        normal = np.maximum(weight * np.cos(alpha) - u * length, 0.0)
        return float(np.sum(cohesion * length + normal * tan_phi)) / driving

    if method in ("bishop", "janbu"):
        numer = cohesion * width + (weight - u * width) * tan_phi
        fos = 1.5
        for _ in range(max_iter):
            m_alpha = np.cos(alpha) + np.sin(alpha) * tan_phi / fos
            if method == "bishop":
                fos_new = float(np.sum(numer / m_alpha)) / driving
            else:
                n_alpha = np.cos(alpha) * m_alpha
                fos_new = float(np.sum(numer / n_alpha)) / float(
                    np.sum(weight * np.tan(alpha)))
            if abs(fos_new - fos) < tol:
                return fos_new
            fos = fos_new
        return fos

    if method == "spencer":
        return _spencer(weight, alpha, length, cohesion, tan_phi, u,
                        driving, max_iter, tol)
    raise ValueError("method must be 'fellenius', 'bishop', 'janbu' or "
                     "'spencer'.")


def _spencer(
    weight: np.ndarray,
    alpha: np.ndarray,
    length: np.ndarray,
    cohesion: np.ndarray,
    tan_phi: np.ndarray,
    u: np.ndarray,
    driving: float,
    max_iter: int,
    tol: float,
) -> float:
    """Spencer's method: solve force and moment equilibrium for (F, theta)."""

    def q_resultant(fos: float, theta: float) -> np.ndarray:
        resist = (cohesion * length
                  + (weight * np.cos(alpha) - u * length) * tan_phi) / fos
        num = resist - weight * np.sin(alpha)
        den = (np.cos(alpha - theta)
               + np.sin(alpha - theta) * tan_phi / fos)
        return num / den

    def solve_f(theta: float, moment: bool) -> float:
        """Find F satisfying force (or moment) equilibrium at fixed theta."""
        fos = 1.5
        for _ in range(max_iter):
            q = q_resultant(fos, theta)
            weights = np.cos(alpha - theta) if moment else np.ones_like(q)
            # Newton-free fixed point: rescale F by the equilibrium residual.
            resist = (cohesion * length
                      + (weight * np.cos(alpha) - u * length) * tan_phi)
            den = (np.cos(alpha - theta)
                   + np.sin(alpha - theta) * tan_phi / fos)
            num_r = np.sum(weights * resist / den)
            num_d = np.sum(weights * weight * np.sin(alpha) / den)
            if abs(num_d) < 1e-12:
                return math.inf
            fos_new = num_r / num_d
            if abs(fos_new - fos) < tol:
                return fos_new
            fos = fos_new
        return fos

    # Root-find theta where F_force(theta) == F_moment(theta).
    thetas = np.radians(np.linspace(-30.0, 30.0, 61))
    best_theta, best_diff = 0.0, math.inf
    prev = None
    for th in thetas:
        ff = solve_f(th, moment=False)
        fm = solve_f(th, moment=True)
        if not (math.isfinite(ff) and math.isfinite(fm)):
            continue
        diff = ff - fm
        if prev is not None and prev[1] * diff < 0:      # sign change -> root
            th0, d0 = prev
            th_root = th0 + (th - th0) * (0.0 - d0) / (diff - d0)
            return solve_f(th_root, moment=True)
        if abs(diff) < best_diff:
            best_diff, best_theta = abs(diff), th
        prev = (th, diff)
    return solve_f(best_theta, moment=True)
