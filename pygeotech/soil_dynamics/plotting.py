"""Plots for the soil_dynamics submodule."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pygeotech.plot_style import academic_style
from pygeotech.soil_dynamics.liquefaction import crr_from_spt

__all__ = ["plot_liquefaction_chart", "plot_transfer_function"]

#: One data point on the triggering chart: ((N1)60cs, CSR*, liquefied).
LiqPoint = Tuple[float, float, bool]


def plot_liquefaction_chart(
    points: Optional[Sequence[LiqPoint]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Figure, Axes]:
    """Plot the SPT liquefaction-triggering chart (CRR7.5 boundary).

    Points below/right of the curve are non-liquefiable. Each ``points``
    entry is ``((N1)60cs, CSR, liquefied_flag)``.
    """
    n = np.linspace(0.0, 37.0, 300)
    crr = np.array([crr_from_spt(ni) for ni in n])
    with academic_style():
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        ax.plot(n, crr, color="k", lw=1.8,
                label=r"$CRR_{7.5}$ (Idriss-Boulanger)")
        ax.fill_between(n, crr, 0.6, color="C3", alpha=0.10)
        ax.fill_between(n, 0.0, crr, color="C2", alpha=0.10)
        ax.text(6, 0.42, "Liquefaction", color="C3", fontsize=9)
        ax.text(26, 0.1, "No liquefaction", color="C2", fontsize=9)
        if points:
            for n_cs, csr, liq in points:
                ax.scatter(n_cs, csr, s=45,
                           facecolor="C3" if liq else "white",
                           edgecolor="k", linewidth=0.8, zorder=5,
                           marker="o" if liq else "s")
        ax.set_xlabel(r"Corrected blow count, $(N_1)_{60cs}$")
        ax.set_ylabel(r"$CSR$ / $CRR_{7.5}$")
        ax.set_xlim(0.0, 37.0)
        ax.set_ylim(0.0, 0.6)
        ax.legend(loc="upper left")
        if save_path is not None:
            fig.savefig(save_path)
        if show:
            plt.show()
    return fig, ax


def plot_transfer_function(
    thickness: float,
    vs: float,
    damping: float,
    freq_max: float = 15.0,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Figure, Axes]:
    """Plot the 1-D site amplification (transfer-function) spectrum."""
    from pygeotech.soil_dynamics.site_response import (
        site_natural_frequency,
        transfer_function_amplitude,
    )
    f = np.linspace(0.05, freq_max, 600)
    amp = transfer_function_amplitude(f, thickness, vs, damping)
    with academic_style():
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        ax.plot(f, amp, color="C0", lw=1.6)
        for mode in (1, 2, 3):
            ax.axvline(site_natural_frequency(thickness, vs, mode),
                       color="0.7", lw=0.6, ls=":")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplification")
        ax.set_xlim(0.0, freq_max)
        ax.set_ylim(bottom=0.0)
        if save_path is not None:
            fig.savefig(save_path)
        if show:
            plt.show()
    return fig, ax
