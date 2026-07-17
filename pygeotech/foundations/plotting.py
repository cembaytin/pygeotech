"""Publication-quality plots for the foundations submodule."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pygeotech.foundations.bearing_capacity import bearing_capacity_factors
from pygeotech.plot_style import academic_style

__all__ = ["plot_bearing_capacity_factors"]


def plot_bearing_capacity_factors(
    method: str = "vesic",
    phi_range: Tuple[float, float] = (0.0, 45.0),
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Figure, Axes]:
    """Plot :math:`N_c, N_q, N_\\gamma` versus friction angle (log scale).

    Parameters
    ----------
    method
        Bearing-capacity factor set.
    phi_range
        Range of friction angles [degrees].
    """
    phi = np.linspace(phi_range[0], phi_range[1], 91)
    nc, nq, ng = [], [], []
    for p in phi:
        c, q, g = bearing_capacity_factors(float(p), method)
        nc.append(c)
        nq.append(q)
        ng.append(g)

    with academic_style():
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        ax.plot(phi, nc, color="k", lw=1.6, label=r"$N_c$")
        ax.plot(phi, nq, color="C0", lw=1.6, ls="--", label=r"$N_q$")
        ax.plot(phi, ng, color="C3", lw=1.6, ls="-.",
                label=r"$N_\gamma$")
        ax.set_yscale("log")
        ax.set_xlabel(r"Friction angle, $\phi$ (deg)")
        ax.set_ylabel("Bearing-capacity factor")
        ax.set_xlim(phi_range)
        ax.set_ylim(1.0, 1e3)
        ax.set_title(f"{method.capitalize()} bearing-capacity factors",
                     fontsize=9)
        ax.grid(True, which="both", lw=0.4, alpha=0.4)
        ax.legend(loc="upper left")
        if save_path is not None:
            fig.savefig(save_path)
        if show:
            plt.show()
    return fig, ax
