"""Plots for the slope_stability submodule."""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pygeotech.plot_style import academic_style
from pygeotech.slope_stability.slices import SlipCircle

__all__ = ["plot_slope_circle"]


def plot_slope_circle(
    ground_y: Callable[[np.ndarray], np.ndarray],
    circle: SlipCircle,
    factor_of_safety: Optional[float] = None,
    x_range: Optional[Tuple[float, float]] = None,
    n_slices: int = 12,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Figure, Axes]:
    """Draw the slope, the trial slip circle and the slices."""
    xc, yc, r = circle.xc, circle.yc, circle.radius
    if x_range is None:
        x_range = (xc - 1.3 * r, xc + 1.3 * r)
    x = np.linspace(x_range[0], x_range[1], 400)

    with academic_style():
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(x, ground_y(x), color="k", lw=1.6)

        # Slip arc where it lies below the ground surface.
        xa = np.linspace(xc - r, xc + r, 400)
        with np.errstate(invalid="ignore"):
            y_arc = yc - np.sqrt(np.maximum(r ** 2 - (xa - xc) ** 2, 0.0))
        below = y_arc < ground_y(xa) - 1e-9
        ax.plot(xa[below], y_arc[below], color="C3", lw=1.6)
        ax.plot(xc, yc, "+", color="C3", ms=10)

        # Slice boundaries.
        if below.any():
            xl, xr = xa[below][0], xa[below][-1]
            for xe in np.linspace(xl, xr, n_slices + 1):
                y_base = yc - math.sqrt(max(r ** 2 - (xe - xc) ** 2, 0.0))
                ax.plot([xe, xe], [y_base, float(ground_y(np.array([xe]))[0])],
                        color="0.75", lw=0.4)

        ax.set_xlabel("$x$ (m)")
        ax.set_ylabel("$y$ (m)")
        ax.set_aspect("equal", adjustable="box")
        if factor_of_safety is not None:
            ax.set_title(f"Critical circle, $F = {factor_of_safety:.3f}$",
                         fontsize=10)
        if save_path is not None:
            fig.savefig(save_path)
        if show:
            plt.show()
    return fig, ax


def plot_layered_slope(
    ground_y,
    surface,
    soil,
    factor_of_safety=None,
    x_range=None,
    save_path=None,
    show=False,
):
    """Draw a layered slope with a circular or polyline slip surface.

    ``surface`` is a ``SlipCircle`` or ``PolylineSurface``; ``soil`` is a
    ``LayeredSoil``. Layer boundaries are drawn as horizontal lines.
    """
    from pygeotech.slope_stability.advanced import PolylineSurface
    from pygeotech.slope_stability.slices import SlipCircle

    if x_range is None:
        if isinstance(surface, SlipCircle):
            x_range = (surface.xc - 1.3 * surface.radius,
                       surface.xc + 1.3 * surface.radius)
        else:
            x0, x1 = surface.x_range
            x_range = (x0 - 2.0, x1 + 2.0)
    x = np.linspace(x_range[0], x_range[1], 400)

    with academic_style():
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        gy = ground_y(x)
        ax.plot(x, gy, color="k", lw=1.6, zorder=4)
        # Layer boundaries (finite tops only).
        for layer in soil.layers:
            if np.isfinite(layer.top) and gy.min() < layer.top < gy.max():
                ax.axhline(layer.top, color="0.6", lw=0.7, ls="--")
                ax.annotate(
                    rf"$c'={layer.cohesion:g},\ \phi'={layer.friction_angle:g}$",
                    xy=(x_range[0], layer.top), fontsize=7, va="bottom",
                    color="0.4")
        # Slip surface.
        if isinstance(surface, SlipCircle):
            xa = np.linspace(surface.xc - surface.radius,
                             surface.xc + surface.radius, 400)
            ya = surface.yc - np.sqrt(np.maximum(
                surface.radius ** 2 - (xa - surface.xc) ** 2, 0.0))
            below = ya < ground_y(xa) - 1e-9
            ax.plot(xa[below], ya[below], color="C3", lw=1.8, zorder=5)
            ax.plot(surface.xc, surface.yc, "+", color="C3", ms=9)
        else:
            ax.plot(surface.x, surface.y, color="C3", lw=1.8, zorder=5)

        ax.set_xlabel("$x$ (m)")
        ax.set_ylabel("$y$ (m)")
        ax.set_aspect("equal", adjustable="box")
        if factor_of_safety is not None:
            ax.set_title(f"Layered slope, $F = {factor_of_safety:.3f}$",
                         fontsize=10)
        if save_path is not None:
            fig.savefig(save_path)
        if show:
            plt.show()
    return fig, ax
