"""Publication-quality plots for the stresses submodule.

* :func:`plot_geostatic_profile` -- the classic total / pore / effective
  vertical-stress-versus-depth diagram for a layered profile.
* :func:`plot_pressure_bulb` -- vertical-stress isobars (the "pressure
  bulb") beneath a strip or rectangular footing for the Boussinesq or
  Westergaard solution.

Both render under :func:`pygeotech.plot_style.academic_style`.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pygeotech.plot_style import academic_style
from pygeotech.stresses.geostatic import SoilProfile
from pygeotech.stresses.induced import (
    boussinesq_rectangle,
    induced_stress_area,
)

__all__ = ["plot_geostatic_profile", "plot_pressure_bulb"]


def plot_geostatic_profile(
    profile: SoilProfile,
    zmax: Optional[float] = None,
    dz: float = 0.05,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Figure, Axes]:
    """Plot total, pore and effective vertical stress against depth.

    Parameters
    ----------
    profile
        The :class:`~pygeotech.stresses.geostatic.SoilProfile` to draw.
    zmax, dz
        Maximum depth and sampling interval [m].
    save_path
        Optional output path (``.pdf`` for vector, ``.png``/``.tiff`` at
        600 dpi).
    show
        Whether to call ``plt.show()``.

    Returns
    -------
    (Figure, Axes)
    """
    depth, sigma, u, sigma_eff = profile.profile(zmax=zmax, dz=dz)
    with academic_style():
        fig, ax = plt.subplots(figsize=(4.5, 5.5))
        ax.plot(sigma, depth, color="C3", lw=1.6,
                label=r"Total, $\sigma_v$")
        ax.plot(u, depth, color="C0", lw=1.4, ls="--",
                label=r"Pore water, $u$")
        ax.plot(sigma_eff, depth, color="k", lw=1.8,
                label=r"Effective, $\sigma'_v$")

        # Layer interfaces and water table as guide lines.
        for top in profile._tops[1:-1]:
            ax.axhline(top, color="0.7", lw=0.6, ls=":")
        if np.isfinite(profile.water_table_depth):
            ax.axhline(profile.water_table_depth, color="C0", lw=0.8,
                       ls="-.", alpha=0.7)
            ax.annotate(
                r"$\nabla$ WT", xy=(0.02, profile.water_table_depth),
                xycoords=("axes fraction", "data"), color="C0",
                va="bottom", fontsize=9,
            )

        ax.set_xlabel("Vertical stress (kPa)")
        ax.set_ylabel("Depth, $z$ (m)")
        ax.set_xlim(left=0.0)
        ax.set_ylim(depth.max(), 0.0)   # depth increases downward
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        ax.legend(loc="lower right")

        if save_path is not None:
            fig.savefig(save_path)
        if show:
            plt.show()
    return fig, ax


def plot_pressure_bulb(
    pressure: float,
    width: float,
    length: Optional[float] = None,
    *,
    method: str = "boussinesq",
    nu: float = 0.0,
    levels: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9),
    half_width_factor: float = 2.5,
    depth_factor: float = 3.0,
    n_grid: int = 120,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Figure, Axes]:
    """Plot vertical-stress isobars (pressure bulb) beneath a footing.

    The isobars are contours of :math:`\\Delta\\sigma_z / q` on the
    vertical centre-plane of the footing.

    Parameters
    ----------
    pressure
        Uniform contact pressure :math:`q` [kPa] (sets the stress scale;
        the contour *labels* are the dimensionless ratio
        :math:`\\Delta\\sigma_z/q`).
    width
        Footing width :math:`B` [m].
    length
        Footing length :math:`L` [m]. If ``None`` a plane-strain strip
        footing is drawn (a long rectangle, ``L = 50 B``).
    method
        ``"boussinesq"`` (closed-form) or ``"westergaard"`` (numerically
        integrated).
    nu
        Poisson's ratio for the Westergaard solution.
    levels
        Contour levels of :math:`\\Delta\\sigma_z / q`.
    half_width_factor, depth_factor
        Extent of the plotting window as multiples of ``B``.
    n_grid
        Grid resolution per axis.
    save_path, show
        Output path and display toggle.

    Returns
    -------
    (Figure, Axes)
    """
    strip = length is None
    plan_length = 50.0 * width if strip else float(length)

    x = np.linspace(-half_width_factor * width, half_width_factor * width,
                    n_grid)
    z = np.linspace(1e-3, depth_factor * width, n_grid)
    grid_x, grid_z = np.meshgrid(x, z)

    if method == "boussinesq":
        # Closed-form, vectorised over the whole grid; the footing spans
        # X in [0, B] and Y in [0, L] so we shift x to centre it.
        ratio = boussinesq_rectangle(
            1.0, width, plan_length, grid_z,
            x=grid_x + width / 2.0, y=plan_length / 2.0,
        )
    elif method == "westergaard":
        ratio = np.empty_like(grid_x)
        for i in range(grid_x.shape[0]):
            for j in range(grid_x.shape[1]):
                ratio[i, j] = induced_stress_area(
                    1.0, width, plan_length, grid_z[i, j],
                    x=grid_x[i, j] + width / 2.0, y=plan_length / 2.0,
                    method="westergaard", nu=nu, n_cells=40,
                )
    else:
        raise ValueError("method must be 'boussinesq' or 'westergaard'.")

    with academic_style():
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        contour = ax.contour(grid_x, grid_z, ratio, levels=list(levels),
                             colors="k", linewidths=0.9)
        ax.clabel(contour, inline=True, fontsize=8, fmt="%.2f")
        filled = ax.contourf(grid_x, grid_z, ratio, levels=list(levels),
                            cmap="Blues", alpha=0.55, extend="max")
        cbar = fig.colorbar(filled, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label(r"$\Delta\sigma_z\,/\,q$")

        # Footing footprint at the surface.
        ax.plot([-width / 2.0, width / 2.0], [0.0, 0.0], color="C3", lw=4,
                solid_capstyle="butt")
        kind = "strip" if strip else f"{width:g}x{plan_length:g} m"
        label = (f"{method.capitalize()} pressure bulb "
                 f"($q = {pressure:g}$ kPa, {kind} footing)")

        ax.set_xlabel("Horizontal distance, $x$ (m)")
        ax.set_ylabel("Depth, $z$ (m)")
        ax.set_ylim(depth_factor * width, 0.0)
        ax.set_title(label, fontsize=9)
        ax.set_aspect("equal", adjustable="box")

        if save_path is not None:
            fig.savefig(save_path)
        if show:
            plt.show()
    return fig, ax
