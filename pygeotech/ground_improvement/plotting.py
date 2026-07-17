"""Plots for the ground_improvement submodule."""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pygeotech.consolidation.terzaghi import average_degree_of_consolidation
from pygeotech.ground_improvement.vertical_drains import (
    combined_degree_of_consolidation,
    hansbo_factor,
    radial_degree_of_consolidation,
    radial_time_factor,
)
from pygeotech.plot_style import academic_style

__all__ = ["plot_pvd_consolidation"]


def plot_pvd_consolidation(
    ch: float,
    de: float,
    f_n: float,
    time_max: float,
    cv: Optional[float] = None,
    drainage_path: Optional[float] = None,
    n_points: int = 200,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Figure, Axes]:
    """Plot degree of consolidation vs. time for a PVD-improved soil.

    Shows the radial component and, if vertical parameters are supplied,
    the combined (Carrillo) consolidation, illustrating the acceleration
    provided by the drains.
    """
    t = np.linspace(1e-6, time_max, n_points)
    ur = np.array([radial_degree_of_consolidation(
        radial_time_factor(ch, ti, de), f_n) for ti in t])
    with academic_style():
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        ax.plot(t, ur * 100.0, color="C0", lw=1.6, label="Radial (PVD)")
        if cv is not None and drainage_path is not None:
            uv = np.array([average_degree_of_consolidation(
                cv * ti / drainage_path ** 2) for ti in t])
            comb = combined_degree_of_consolidation(uv, ur)
            ax.plot(t, uv * 100.0, color="0.6", lw=1.2, ls=":",
                    label="Vertical only")
            ax.plot(t, comb * 100.0, color="C3", lw=1.8,
                    label="Combined")
        ax.set_xlabel("Time")
        ax.set_ylabel(r"Degree of consolidation, $U$ (%)")
        ax.set_xlim(0.0, time_max)
        ax.set_ylim(100.0, 0.0)
        ax.legend(loc="lower right")
        if save_path is not None:
            fig.savefig(save_path)
        if show:
            plt.show()
    return fig, ax
