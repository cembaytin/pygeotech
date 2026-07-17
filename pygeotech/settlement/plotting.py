"""Plots for the settlement submodule."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pygeotech.plot_style import academic_style

__all__ = ["plot_strain_influence"]


def plot_strain_influence(
    net_pressure: float,
    peak_effective_stress: float,
    width: float,
    plane_strain: bool = False,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Figure, Axes]:
    """Plot the Schmertmann strain-influence diagram :math:`I_z(z)`."""
    iz_surface = 0.2 if plane_strain else 0.1
    zp = (1.0 if plane_strain else 0.5) * width
    zmax = (4.0 if plane_strain else 2.0) * width
    izp = 0.5 + 0.1 * math.sqrt(net_pressure / peak_effective_stress)

    z = np.array([0.0, zp, zmax])
    iz = np.array([iz_surface, izp, 0.0])
    with academic_style():
        fig, ax = plt.subplots(figsize=(4.2, 5.2))
        ax.plot(iz, z, color="C3", lw=1.6)
        ax.fill_betweenx(z, 0.0, iz, color="C3", alpha=0.15)
        ax.axhline(zp, color="0.7", lw=0.6, ls=":")
        ax.annotate(rf"$I_{{zp}} = {izp:.2f}$", xy=(izp, zp),
                    xytext=(izp * 0.5, zp * 0.7), fontsize=9)
        ax.set_xlabel(r"Strain influence factor, $I_z$")
        ax.set_ylabel("Depth below footing, $z$ (m)")
        ax.set_xlim(left=0.0)
        ax.set_ylim(zmax, 0.0)
        kind = "plane strain" if plane_strain else "axisymmetric"
        ax.set_title(f"Schmertmann $I_z$ ({kind})", fontsize=9)
        if save_path is not None:
            fig.savefig(save_path)
        if show:
            plt.show()
    return fig, ax
