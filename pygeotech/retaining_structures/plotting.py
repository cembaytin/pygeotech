"""Publication-quality plots for the retaining_structures submodule."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pygeotech.plot_style import academic_style
from pygeotech.retaining_structures.earth_pressure import (
    rankine_active_coefficient,
)

__all__ = ["plot_active_pressure_diagram"]


def plot_active_pressure_diagram(
    height: float,
    gamma: float,
    phi: float,
    cohesion: float = 0.0,
    surcharge: float = 0.0,
    water_table_depth: Optional[float] = None,
    gamma_w: float = 9.81,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Figure, Axes]:
    """Plot the Rankine active lateral-pressure distribution behind a wall.

    Effective earth pressure and hydrostatic water pressure are drawn as
    separate components against depth.
    """
    ka = rankine_active_coefficient(phi)
    sqrt_ka = math.sqrt(ka)
    dw = water_table_depth if water_table_depth is not None else math.inf
    gamma_w_eff = gamma_w if water_table_depth is not None else 0.0

    z = np.linspace(0.0, height, 201)
    earth = np.empty_like(z)
    water = np.empty_like(z)
    for i, depth in enumerate(z):
        if depth <= dw:
            sigma_v = surcharge + gamma * depth
            u = 0.0
        else:
            sigma_v = surcharge + gamma * dw + (gamma - gamma_w_eff) * (depth - dw)
            u = gamma_w_eff * (depth - dw)
        earth[i] = max(0.0, ka * sigma_v - 2.0 * cohesion * sqrt_ka)
        water[i] = u

    with academic_style():
        fig, ax = plt.subplots(figsize=(4.6, 5.2))
        ax.plot(earth, z, color="C3", lw=1.6, label="Effective earth pressure")
        ax.fill_betweenx(z, 0.0, earth, color="C3", alpha=0.15)
        if np.any(water > 0.0):
            ax.plot(water, z, color="C0", lw=1.4, ls="--",
                    label="Water pressure")
            ax.fill_betweenx(z, 0.0, water, color="C0", alpha=0.12)
        ax.set_xlabel("Lateral pressure (kPa)")
        ax.set_ylabel("Depth, $z$ (m)")
        ax.set_xlim(left=0.0)
        ax.set_ylim(height, 0.0)
        ax.legend(loc="lower right")
        if save_path is not None:
            fig.savefig(save_path)
        if show:
            plt.show()
    return fig, ax
