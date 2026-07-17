"""Publication-quality plots for the consolidation submodule."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pygeotech.consolidation.terzaghi import (
    Consolidation1D,
    average_degree_of_consolidation,
    excess_pressure_ratio,
)
from pygeotech.plot_style import academic_style

__all__ = [
    "plot_isochrones",
    "plot_degree_vs_time_factor",
    "plot_time_settlement",
]


def plot_isochrones(
    time_factors: Sequence[float] = (0.05, 0.1, 0.2, 0.4, 0.6, 0.9),
    drainage: str = "double",
    n_points: int = 201,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Figure, Axes]:
    """Plot excess pore-pressure isochrones :math:`u_e/u_0` vs. depth."""
    z_max = 2.0 if drainage == "double" else 1.0
    z_ratio = np.linspace(0.0, z_max, n_points)
    with academic_style():
        fig, ax = plt.subplots(figsize=(4.8, 5.2))
        for tv in time_factors:
            ratio = [excess_pressure_ratio(z, tv) for z in z_ratio]
            ax.plot(ratio, z_ratio / z_max, lw=1.3, label=f"$T_v = {tv:g}$")
        ax.set_xlabel(r"Excess pore-pressure ratio, $u_e / u_0$")
        ax.set_ylabel(r"Normalised depth, $z / H$")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(1.0, 0.0)
        ax.legend(loc="lower left", ncol=2)
        if save_path is not None:
            fig.savefig(save_path)
        if show:
            plt.show()
    return fig, ax


def plot_degree_vs_time_factor(
    tv_max: float = 2.0,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Figure, Axes]:
    """Plot the average degree of consolidation :math:`U(T_v)`."""
    tv = np.linspace(1e-3, tv_max, 300)
    u = np.array([average_degree_of_consolidation(t) for t in tv])
    with academic_style():
        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        ax.plot(tv, u * 100.0, color="k", lw=1.6)
        ax.set_xlabel(r"Time factor, $T_v$")
        ax.set_ylabel(r"Average degree of consolidation, $U$ (%)")
        ax.set_xlim(0.0, tv_max)
        ax.set_ylim(100.0, 0.0)
        ax.grid(True, which="both", lw=0.4, alpha=0.4)
        if save_path is not None:
            fig.savefig(save_path)
        if show:
            plt.show()
    return fig, ax


def plot_time_settlement(
    model: Consolidation1D,
    ultimate_settlement: float,
    times: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Figure, Axes]:
    """Plot the settlement-versus-time curve for a consolidating layer.

    Parameters
    ----------
    model
        A :class:`~pygeotech.consolidation.terzaghi.Consolidation1D`.
    ultimate_settlement
        Final primary consolidation settlement :math:`s_c` [m or mm].
    times
        Times at which to evaluate; defaults to the time to reach 99 %.
    """
    if times is None:
        t99 = model.time_for_degree(0.99)
        times = np.linspace(1e-4, t99, 300)
    settlement = np.array(
        [model.average_degree(t) * ultimate_settlement for t in times]
    )
    with academic_style():
        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        ax.plot(times, settlement, color="C3", lw=1.6)
        ax.set_xlabel("Time")
        ax.set_ylabel("Settlement")
        ax.set_xlim(left=0.0)
        ax.set_ylim(ultimate_settlement * 1.02, 0.0)
        if save_path is not None:
            fig.savefig(save_path)
        if show:
            plt.show()
    return fig, ax
