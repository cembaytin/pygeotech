"""Publication-quality plotting for the phase_relations submodule.

Currently implemented: the Casagrande plasticity chart with the A-line
(:math:`PI = 0.73(LL-20)`), the U-line (:math:`PI = 0.9(LL-8)`), the
``LL = 50`` boundary and the CL-ML transition band (``4 <= PI <= 7``).
Figures are rendered under :func:`pygeotech.plot_style.academic_style`
(serif typography, 600 dpi, embedded Type-42 fonts) and can be exported
directly to vector PDF for journal submission.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pygeotech.plot_style import academic_style

__all__ = ["plot_plasticity_chart"]

#: (liquid limit, plasticity index, label) of one sample.
Sample = Tuple[float, float, str]


def _draw_chart_frame(ax: Axes, ll_max: float, pi_max: float) -> None:
    """Draw A-line, U-line, LL = 50 boundary and zone labels."""
    # A-line: horizontal at PI = 4 up to its intersection with the
    # inclined branch (LL = 25.48), then PI = 0.73 (LL - 20).
    ll_a = np.linspace(4.0 / 0.73 + 20.0, ll_max, 200)
    ax.plot([0.0, ll_a[0]], [4.0, 4.0], color="k", lw=1.0)
    ax.plot(ll_a, 0.73 * (ll_a - 20.0), color="k", lw=1.0)
    ax.annotate(
        "A-line", xy=(0.72 * ll_max, 0.73 * (0.72 * ll_max - 20.0) + 1.5),
        rotation=np.degrees(np.arctan(0.73)), rotation_mode="anchor",
        ha="center", fontsize=9,
    )

    # U-line: horizontal at PI = 7, then PI = 0.9 (LL - 8).
    ll_u = np.linspace(7.0 / 0.9 + 8.0, (pi_max / 0.9) + 8.0, 200)
    ax.plot([0.0, ll_u[0]], [7.0, 7.0], color="k", lw=0.8, ls="--")
    ax.plot(ll_u, 0.9 * (ll_u - 8.0), color="k", lw=0.8, ls="--")
    ax.annotate(
        "U-line", xy=(0.32 * ll_max, 0.9 * (0.32 * ll_max - 8.0) + 1.5),
        rotation=np.degrees(np.arctan(0.9)), rotation_mode="anchor",
        ha="center", fontsize=9,
    )

    # LL = 50 boundary between low- and high-plasticity soils.
    ax.axvline(50.0, color="k", lw=0.8, ls=":")

    # CL-ML transition band.
    ax.fill_betweenx(
        [4.0, 7.0], [4.0 / 0.9 + 8.0, 7.0 / 0.9 + 8.0],
        [4.0 / 0.73 + 20.0, 7.0 / 0.73 + 20.0],
        color="0.85", zorder=0,
    )
    ax.annotate("CL-ML", xy=(15.0, 5.4), fontsize=8, ha="center",
                va="center")

    # Zone labels.
    zones = (
        ("CL", 36, 22), ("ML or OL", 42, 6.5), ("CH", 62, 42),
        ("MH or OH", 75, 18),
    )
    for label, x_pos, y_pos in zones:
        if x_pos < ll_max and y_pos < pi_max:
            ax.annotate(label, xy=(x_pos, y_pos), fontsize=10, ha="center",
                        va="center", fontstyle="italic")


def plot_plasticity_chart(
    samples: Optional[Sequence[Sample]] = None,
    *,
    ll_max: float = 100.0,
    pi_max: float = 60.0,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Figure, Axes]:
    """Plot the Casagrande plasticity chart with optional sample points.

    Parameters
    ----------
    samples
        Sequence of ``(liquid_limit, plasticity_index, label)`` tuples
        to scatter on the chart.
    ll_max, pi_max
        Axis limits for LL and PI [%].
    save_path
        If given, the figure is saved there; the extension selects the
        backend (``.pdf`` recommended for vector output, ``.tiff``/
        ``.png`` are rendered at 600 dpi).
    show
        Call ``plt.show()`` after drawing (default ``False`` so the
        function stays script/CI friendly).

    Returns
    -------
    (Figure, Axes)
        The matplotlib figure and axes, for further customization.
    """
    with academic_style():
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        _draw_chart_frame(ax, ll_max, pi_max)

        if samples:
            markers = ("o", "s", "^", "D", "v", "P", "X", "*")
            for idx, (ll, pi, label) in enumerate(samples):
                ax.scatter(
                    ll, pi, s=45, marker=markers[idx % len(markers)],
                    facecolor="C0" if idx % 2 == 0 else "white",
                    edgecolor="k", linewidth=0.8, zorder=5, label=label,
                )
            ax.legend(loc="upper left", fontsize=8)

        ax.set_xlim(0.0, ll_max)
        ax.set_ylim(0.0, pi_max)
        ax.set_xlabel("Liquid limit, $LL$ (%)")
        ax.set_ylabel("Plasticity index, $PI$ (%)")

        if save_path is not None:
            fig.savefig(save_path)
        if show:
            plt.show()
    return fig, ax
