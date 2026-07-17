"""Meshing utilities for the pyGeotech finite-element engine.

The engine uses three-node linear triangular elements (constant-strain /
constant-gradient triangles, "T3"/"CST"). This module provides the mesh
container and a structured generator for rectangular domains, which is
enough for confined-seepage, consolidation and plane elasticity problems;
unstructured meshes can be supplied directly to :class:`TriMesh`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

__all__ = ["TriMesh", "rectangular_mesh"]


@dataclass
class TriMesh:
    """A 2-D triangular mesh.

    Attributes
    ----------
    nodes
        ``(n_nodes, 2)`` array of node coordinates ``(x, y)``.
    elements
        ``(n_elements, 3)`` integer array of node indices; each triangle
        is stored counter-clockwise.
    """

    nodes: np.ndarray
    elements: np.ndarray

    def __post_init__(self) -> None:
        self.nodes = np.asarray(self.nodes, dtype=float)
        self.elements = np.asarray(self.elements, dtype=int)
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2:
            raise ValueError("nodes must be an (n, 2) array.")
        if self.elements.ndim != 2 or self.elements.shape[1] != 3:
            raise ValueError("elements must be an (m, 3) array.")

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return self.nodes.shape[0]

    @property
    def n_elements(self) -> int:
        """Number of triangular elements."""
        return self.elements.shape[0]

    def element_areas(self) -> np.ndarray:
        """Signed areas of every element (positive if counter-clockwise)."""
        p = self.nodes[self.elements]           # (m, 3, 2)
        x1, y1 = p[:, 0, 0], p[:, 0, 1]
        x2, y2 = p[:, 1, 0], p[:, 1, 1]
        x3, y3 = p[:, 2, 0], p[:, 2, 1]
        return 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    def element_centroids(self) -> np.ndarray:
        """Centroids of every element, shape ``(n_elements, 2)``."""
        return self.nodes[self.elements].mean(axis=1)

    def nodes_where(
        self, axis: str, value: float, tol: float = 1e-9
    ) -> np.ndarray:
        """Indices of nodes lying on the line ``x = value`` or ``y = value``.

        Parameters
        ----------
        axis
            ``"x"`` or ``"y"``.
        value
            Coordinate of the line.
        tol
            Absolute matching tolerance.
        """
        col = 0 if axis == "x" else 1
        return np.where(np.abs(self.nodes[:, col] - value) < tol)[0]


def rectangular_mesh(
    width: float,
    height: float,
    nx: int,
    ny: int,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> TriMesh:
    """Structured triangular mesh of a rectangle.

    The rectangle is divided into ``nx`` by ``ny`` cells, each split into
    two counter-clockwise triangles.

    Parameters
    ----------
    width, height
        Domain dimensions [m].
    nx, ny
        Number of cells along x and y.
    origin
        Coordinates of the bottom-left corner.

    Returns
    -------
    TriMesh
    """
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1.")
    x0, y0 = origin
    xs = np.linspace(x0, x0 + width, nx + 1)
    ys = np.linspace(y0, y0 + height, ny + 1)
    grid_x, grid_y = np.meshgrid(xs, ys)
    nodes = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    def node_id(i: int, j: int) -> int:
        return j * (nx + 1) + i

    elements: List[Sequence[int]] = []
    for j in range(ny):
        for i in range(nx):
            n00 = node_id(i, j)
            n10 = node_id(i + 1, j)
            n11 = node_id(i + 1, j + 1)
            n01 = node_id(i, j + 1)
            # Two counter-clockwise triangles per cell.
            elements.append((n00, n10, n11))
            elements.append((n00, n11, n01))
    return TriMesh(nodes=nodes, elements=np.array(elements, dtype=int))
