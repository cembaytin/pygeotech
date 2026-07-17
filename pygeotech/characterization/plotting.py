"""Publication-quality plots for the characterization submodule."""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from pygeotech.characterization.cpt import CPTLog, CPTResult
from pygeotech.characterization.spt import SPTLog
from pygeotech.plot_style import academic_style

__all__ = ["plot_cpt_profile", "plot_spt_profile"]

#: Boundaries and colours of the Robertson SBT zones for the Ic track.
_IC_BOUNDS = (1.31, 2.05, 2.60, 2.95, 3.60)


def plot_cpt_profile(
    log: CPTLog,
    result: Optional[CPTResult] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Figure, np.ndarray]:
    """Plot a CPT sounding: qt, friction ratio and Ic (with SBT bands).

    Parameters
    ----------
    log
        The :class:`~pygeotech.characterization.cpt.CPTLog`.
    result
        A pre-computed :class:`CPTResult`; computed from ``log`` if ``None``.
    """
    res = result if result is not None else log.process()
    with academic_style():
        fig, axes = plt.subplots(1, 3, figsize=(7.5, 6.0), sharey=True)
        ax_q, ax_r, ax_ic = axes

        ax_q.plot(res.qt / 1000.0, res.depth, color="k", lw=1.2)
        ax_q.set_xlabel(r"$q_t$ (MPa)")
        ax_q.set_ylabel("Depth, $z$ (m)")
        ax_q.set_xlim(left=0.0)

        ax_r.plot(res.friction_ratio, res.depth, color="C0", lw=1.2)
        ax_r.set_xlabel(r"$R_f$ (%)")
        ax_r.set_xlim(left=0.0)

        ax_ic.plot(res.ic, res.depth, color="C3", lw=1.4)
        for b in _IC_BOUNDS:
            ax_ic.axvline(b, color="0.75", lw=0.6, ls=":")
        ax_ic.set_xlabel(r"$I_c$ (SBT)")
        ax_ic.set_xlim(1.0, 4.0)

        ax_q.set_ylim(res.depth.max(), 0.0)
        if save_path is not None:
            fig.savefig(save_path)
        if show:
            plt.show()
    return fig, axes


def plot_spt_profile(
    log: SPTLog,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Figure, plt.Axes]:
    """Plot an SPT profile: field N, N60 and (N1)60 versus depth."""
    with academic_style():
        fig, ax = plt.subplots(figsize=(4.6, 6.0))
        ax.plot(log.n_field, log.depth, "-o", color="0.6", ms=3, lw=1.0,
                label=r"$N$ (field)")
        ax.plot(log.n60(), log.depth, "-s", color="C0", ms=3, lw=1.2,
                label=r"$N_{60}$")
        ax.plot(log.n1_60(), log.depth, "-^", color="C3", ms=3, lw=1.2,
                label=r"$(N_1)_{60}$")
        ax.set_xlabel("Blow count")
        ax.set_ylabel("Depth, $z$ (m)")
        ax.set_xlim(left=0.0)
        ax.set_ylim(log.depth.max(), 0.0)
        ax.legend(loc="lower right")
        if save_path is not None:
            fig.savefig(save_path)
        if show:
            plt.show()
    return fig, ax
