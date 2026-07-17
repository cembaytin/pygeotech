"""Plots for the rock_mechanics submodule."""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pygeotech.plot_style import academic_style
from pygeotech.rock_mechanics.hoek_brown import HoekBrownParameters

__all__ = ["plot_hoek_brown"]


def plot_hoek_brown(
    params: HoekBrownParameters,
    sigma3_max: Optional[float] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Figure, Axes]:
    """Plot the Hoek-Brown failure envelope in principal-stress space."""
    if sigma3_max is None:
        sigma3_max = 0.25 * params.sigma_ci
    sigma3 = np.linspace(0.0, sigma3_max, 200)
    sigma1 = params.major_principal_stress(sigma3)
    with academic_style():
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        ax.plot(sigma3, sigma1, color="C3", lw=1.8,
                label="Hoek-Brown envelope")
        ax.plot([0.0], [params.uniaxial_compressive_strength()], "ko", ms=4)
        ax.annotate(
            rf"$\sigma_c = {params.uniaxial_compressive_strength():.1f}$ MPa",
            xy=(0.0, params.uniaxial_compressive_strength()),
            xytext=(sigma3_max * 0.05,
                    params.uniaxial_compressive_strength() * 1.15),
            fontsize=9)
        ax.set_xlabel(r"Minor principal stress, $\sigma_3$ (MPa)")
        ax.set_ylabel(r"Major principal stress, $\sigma_1$ (MPa)")
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.legend(loc="upper left")
        if save_path is not None:
            fig.savefig(save_path)
        if show:
            plt.show()
    return fig, ax
