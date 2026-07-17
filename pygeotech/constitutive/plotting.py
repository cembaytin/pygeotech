"""Plots for the constitutive submodule."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pygeotech.constitutive.cam_clay import TriaxialResult
from pygeotech.plot_style import academic_style

__all__ = ["plot_triaxial"]


def plot_triaxial(
    results: Sequence[Tuple[TriaxialResult, str]],
    critical_state_M: Optional[float] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Figure, Axes]:
    """Plot triaxial stress paths and stress-strain curves.

    Parameters
    ----------
    results
        Sequence of ``(TriaxialResult, label)`` pairs.
    critical_state_M
        If given, the critical-state line :math:`q = M p'` is drawn.
    """
    with academic_style():
        fig, (ax_path, ax_ss) = plt.subplots(1, 2, figsize=(9.0, 4.2))
        p_max = max(float(r.p_eff.max()) for r, _ in results) * 1.05

        if critical_state_M is not None:
            p_line = np.array([0.0, p_max])
            ax_path.plot(p_line, critical_state_M * p_line, color="0.5",
                         lw=1.0, ls="--", label=f"CSL ($M={critical_state_M:g}$)")

        for idx, (res, label) in enumerate(results):
            color = f"C{idx}"
            ax_path.plot(res.p_eff, res.q, color=color, lw=1.6, label=label)
            ax_path.plot(res.p_eff[0], res.q[0], "o", color=color, ms=4)
            ax_ss.plot(res.axial_strain * 100.0, res.q, color=color, lw=1.6,
                       label=label)

        ax_path.set_xlabel("Mean effective stress, $p'$ (kPa)")
        ax_path.set_ylabel("Deviatoric stress, $q$ (kPa)")
        ax_path.set_xlim(left=0.0)
        ax_path.set_ylim(bottom=0.0)
        ax_path.legend(loc="upper left", fontsize=8)
        ax_path.set_title("Effective stress paths", fontsize=9)

        ax_ss.set_xlabel(r"Axial strain, $\varepsilon_a$ (%)")
        ax_ss.set_ylabel("Deviatoric stress, $q$ (kPa)")
        ax_ss.set_xlim(left=0.0)
        ax_ss.set_ylim(bottom=0.0)
        ax_ss.legend(loc="lower right", fontsize=8)
        ax_ss.set_title("Stress-strain response", fontsize=9)

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path)
        if show:
            plt.show()
    return fig, ax_path
