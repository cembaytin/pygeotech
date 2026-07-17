"""Publication-quality plots for the finite-element engine."""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation

from pygeotech.fem.elasticity import ElasticityFEM
from pygeotech.fem.seepage import SeepageFEM
from pygeotech.plot_style import academic_style

__all__ = ["plot_seepage", "plot_deformed_mesh"]


def plot_seepage(
    solver: SeepageFEM,
    n_equipotentials: int = 12,
    show_velocity: bool = True,
    show_mesh: bool = False,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Figure, Axes]:
    """Plot equipotential lines and Darcy-velocity vectors of a solution.

    Parameters
    ----------
    solver
        A solved :class:`~pygeotech.fem.seepage.SeepageFEM`.
    n_equipotentials
        Number of equipotential (constant-head) contour lines.
    show_velocity
        Overlay element-centroid velocity vectors (flow direction).
    show_mesh
        Draw the triangular mesh edges.
    """
    mesh = solver.mesh
    head = solver.head
    tri = Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)

    with academic_style():
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        if show_mesh:
            ax.triplot(tri, color="0.8", lw=0.3)
        filled = ax.tricontourf(tri, head, levels=n_equipotentials,
                              cmap="viridis", alpha=0.85)
        ax.tricontour(tri, head, levels=n_equipotentials, colors="k",
                    linewidths=0.6)
        cbar = fig.colorbar(filled, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label("Total head, $h$ (m)")

        if show_velocity:
            centroids = mesh.element_centroids()
            vel = solver.velocities()
            speed = np.linalg.norm(vel, axis=1)
            scale = np.where(speed > 0, speed, 1.0)
            ax.quiver(centroids[:, 0], centroids[:, 1],
                      vel[:, 0] / scale, vel[:, 1] / scale,
                      color="white", pivot="mid", width=0.003,
                      scale=40, alpha=0.9)

        ax.set_xlabel("$x$ (m)")
        ax.set_ylabel("$y$ (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Confined seepage: equipotentials and flow direction",
                     fontsize=9)
        if save_path is not None:
            fig.savefig(save_path)
        if show:
            plt.show()
    return fig, ax


def plot_deformed_mesh(
    solver: ElasticityFEM,
    scale: Optional[float] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Figure, Axes]:
    """Plot the deformed mesh (exaggerated) coloured by von Mises stress.

    Parameters
    ----------
    solver
        A solved :class:`~pygeotech.fem.elasticity.ElasticityFEM`.
    scale
        Displacement magnification; auto-scaled from the mesh size when
        ``None``.
    """
    mesh = solver.mesh
    disp = solver.solve()          # (n_nodes, 2) nodal displacements
    vm = solver.von_mises()
    span = mesh.nodes.max(axis=0) - mesh.nodes.min(axis=0)
    if scale is None:
        max_disp = np.abs(disp).max()
        scale = 0.15 * span.max() / max_disp if max_disp > 0 else 1.0
    deformed = mesh.nodes + scale * disp

    with academic_style():
        fig, ax = plt.subplots(figsize=(5.0, 5.0))
        undeformed = Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1],
                                   mesh.elements)
        ax.triplot(undeformed, color="0.8", lw=0.4)
        tri = Triangulation(deformed[:, 0], deformed[:, 1], mesh.elements)
        collection = ax.tripcolor(tri, facecolors=vm, cmap="viridis",
                                  edgecolors="0.4", linewidth=0.2)
        cbar = fig.colorbar(collection, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label("von Mises stress (kPa)")
        ax.set_xlabel("$x$ (m)")
        ax.set_ylabel("$y$ (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Deformed mesh ($\\times{scale:.3g}$ exaggeration)",
                     fontsize=9)
        if save_path is not None:
            fig.savefig(save_path)
        if show:
            plt.show()
    return fig, ax
