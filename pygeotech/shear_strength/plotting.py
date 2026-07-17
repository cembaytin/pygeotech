"""Publication-quality plots for the shear_strength submodule."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pygeotech.plot_style import academic_style
from pygeotech.shear_strength.mohr_coulomb import MohrCoulomb

__all__ = ["plot_mohr_circles", "plot_stress_path"]


def plot_mohr_circles(
    sigma3: Sequence[float],
    sigma1: Sequence[float],
    envelope: Optional[MohrCoulomb] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Figure, Axes]:
    """Plot Mohr circles at failure with the Mohr-Coulomb envelope.

    Parameters
    ----------
    sigma3, sigma1
        Minor and major principal stresses at failure [kPa].
    envelope
        A fitted :class:`MohrCoulomb`; if ``None`` it is fitted from the
        supplied triaxial data.
    """
    s3 = np.asarray(sigma3, dtype=float)
    s1 = np.asarray(sigma1, dtype=float)
    if envelope is None:
        envelope = MohrCoulomb.fit_triaxial(s3, s1)

    with academic_style():
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        theta = np.linspace(0.0, np.pi, 180)
        for a, b in zip(s3, s1):
            center = 0.5 * (a + b)
            radius = 0.5 * (b - a)
            ax.plot(center + radius * np.cos(theta),
                    radius * np.sin(theta), color="C0", lw=1.1)

        sigma_max = float(s1.max())
        x = np.array([0.0, sigma_max * 1.05])
        ax.plot(x, envelope.cohesion + x * envelope.tan_phi, color="C3",
                lw=1.6,
                label=(f"$\\tau_f = {envelope.cohesion:.1f} + "
                       f"\\sigma_n\\tan{envelope.friction_angle:.1f}"
                       r"^\circ$"))
        ax.set_xlabel(r"Normal stress, $\sigma_n$ (kPa)")
        ax.set_ylabel(r"Shear stress, $\tau$ (kPa)")
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper left")
        if save_path is not None:
            fig.savefig(save_path)
        if show:
            plt.show()
    return fig, ax


def plot_stress_path(
    p: Sequence[float],
    q: Sequence[float],
    p_eff: Optional[Sequence[float]] = None,
    envelope: Optional[MohrCoulomb] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Figure, Axes]:
    """Plot total (and effective) ``p-q`` stress paths with the K_f line."""
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    with academic_style():
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        ax.plot(p_arr, q_arr, "-o", color="C0", ms=3, lw=1.2,
                label="Total stress path (TSP)")
        if p_eff is not None:
            ax.plot(np.asarray(p_eff, dtype=float), q_arr, "-s", color="C2",
                    ms=3, lw=1.2, label="Effective stress path (ESP)")
        if envelope is not None:
            a, tan_alpha = envelope.kf_line()
            xmax = float(max(p_arr.max(),
                             np.asarray(p_eff).max() if p_eff is not None
                             else 0.0))
            x = np.array([0.0, xmax * 1.05])
            ax.plot(x, a + x * tan_alpha, color="C3", lw=1.4, ls="--",
                    label=r"$K_f$ line")
        ax.set_xlabel(r"$p = (\sigma_1 + \sigma_3)/2$ (kPa)")
        ax.set_ylabel(r"$q = (\sigma_1 - \sigma_3)/2$ (kPa)")
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.legend(loc="upper left")
        if save_path is not None:
            fig.savefig(save_path)
        if show:
            plt.show()
    return fig, ax
