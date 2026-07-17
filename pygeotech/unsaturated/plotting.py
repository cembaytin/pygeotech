"""Plots for the unsaturated submodule."""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pygeotech.plot_style import academic_style
from pygeotech.unsaturated.swcc import (
    relative_permeability_vg,
    van_genuchten_water_content,
)

__all__ = ["plot_swcc"]


def plot_swcc(
    theta_s: float,
    theta_r: float,
    alpha: float,
    n: float,
    suction_range: Tuple[float, float] = (0.1, 1.0e6),
    show_permeability: bool = True,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Figure, Axes]:
    """Plot a van Genuchten SWCC (and relative permeability) vs. suction."""
    psi = np.logspace(np.log10(suction_range[0]), np.log10(suction_range[1]),
                      300)
    theta = van_genuchten_water_content(psi, theta_s, theta_r, alpha, n)
    with academic_style():
        fig, ax = plt.subplots(figsize=(5.4, 4.0))
        ax.semilogx(psi, theta, color="C0", lw=1.8,
                    label=r"$\theta(\psi)$")
        ax.set_xlabel(r"Matric suction, $\psi$ (kPa)")
        ax.set_ylabel(r"Volumetric water content, $\theta$")
        ax.set_ylim(0.0, theta_s * 1.05)
        if show_permeability:
            ax2 = ax.twinx()
            kr = relative_permeability_vg(psi, alpha, n)
            ax2.semilogx(psi, kr, color="C3", lw=1.4, ls="--",
                         label=r"$k_r(\psi)$")
            ax2.set_ylabel(r"Relative permeability, $k_r$")
            ax2.set_ylim(0.0, 1.05)
            lines = ax.get_lines() + ax2.get_lines()
            ax.legend(lines, [ln.get_label() for ln in lines],
                      loc="center left")
        else:
            ax.legend(loc="lower left")
        if save_path is not None:
            fig.savefig(save_path)
        if show:
            plt.show()
    return fig, ax
