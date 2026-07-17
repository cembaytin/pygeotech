"""Transient 2-D consolidation kernel for the pyGeotech FEM engine.

Excess pore pressure :math:`u_e` diffuses according to

.. math:: \\frac{\\partial u_e}{\\partial t} = c_v\\,\\nabla^2 u_e ,

which, discretised with the same linear-triangle elements as the seepage
kernel, gives the semi-discrete system

.. math:: \\mathbf{M}\\,\\dot{\\mathbf{u}} + c_v\\,\\mathbf{K}\\,\\mathbf{u} = 0 ,

with the consistent capacity (mass) matrix :math:`\\mathbf{M}` and the
Laplacian conductance matrix :math:`\\mathbf{K}`. Backward-Euler time
stepping,

.. math:: (\\mathbf{M} + \\Delta t\\, c_v \\mathbf{K})\\,
    \\mathbf{u}^{n+1} = \\mathbf{M}\\,\\mathbf{u}^{n},

is unconditionally stable. Drained boundaries are imposed as
:math:`u_e = 0`. The kernel reproduces Terzaghi's 1-D average-degree-of-
consolidation curve, which is used to validate it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pygeotech.core import PyGeotechError
from pygeotech.fem.mesh import TriMesh

__all__ = ["ConsolidationFEM"]

#: Consistent-mass element template (multiplied by area/12).
_MASS_TEMPLATE = np.array([[2.0, 1.0, 1.0],
                           [1.0, 2.0, 1.0],
                           [1.0, 1.0, 2.0]])


class ConsolidationFEM:
    """Transient 2-D consolidation solver on a triangular mesh.

    Parameters
    ----------
    mesh
        A :class:`~pygeotech.fem.mesh.TriMesh`.
    cv
        Coefficient of consolidation [m^2/time], scalar or per-element.

    Examples
    --------
    >>> from pygeotech.fem.mesh import rectangular_mesh
    >>> mesh = rectangular_mesh(0.2, 4.0, 1, 40)
    >>> fem = ConsolidationFEM(mesh, cv=1.0)
    >>> _ = fem.set_drainage(mesh.nodes_where("y", 0.0))
    >>> _ = fem.set_drainage(mesh.nodes_where("y", 4.0))
    >>> t, u = fem.solve(u_initial=100.0, dt=0.02, n_steps=50)
    >>> 0.0 < fem.average_degree(u[-1], 100.0) < 1.0
    True
    """

    def __init__(self, mesh: TriMesh, cv: Union[float, Sequence[float]] = 1.0) -> None:
        self.mesh = mesh
        arr = np.asarray(cv, dtype=float)
        if arr.ndim == 0:
            self._cv = np.full(mesh.n_elements, float(arr))
        elif arr.shape == (mesh.n_elements,):
            self._cv = arr.copy()
        else:
            raise ValueError("cv must be a scalar or length-n_elements array.")
        self._fixed: Dict[int, float] = {}

    def set_drainage(self, nodes: Sequence[int]) -> "ConsolidationFEM":
        """Mark nodes as drained (excess pore pressure held at zero)."""
        for n in np.atleast_1d(nodes):
            self._fixed[int(n)] = 0.0
        return self

    def assemble(self) -> Tuple[np.ndarray, np.ndarray]:
        """Assemble the conductance ``K`` and capacity ``M`` matrices."""
        n = self.mesh.n_nodes
        k_global = np.zeros((n, n))
        m_global = np.zeros((n, n))
        coords = self.mesh.nodes
        for e, tri in enumerate(self.mesh.elements):
            i, j, k = tri
            (xi, yi), (xj, yj), (xk, yk) = coords[i], coords[j], coords[k]
            area = 0.5 * ((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
            if abs(area) < 1e-15:
                raise PyGeotechError(f"degenerate element {e}.")
            b = np.array([yj - yk, yk - yi, yi - yj])
            c = np.array([xk - xj, xi - xk, xj - xi])
            ke = self._cv[e] * (np.outer(b, b) + np.outer(c, c)) / (4.0 * area)
            me = abs(area) / 12.0 * _MASS_TEMPLATE
            idx = np.array([i, j, k])
            k_global[np.ix_(idx, idx)] += ke
            m_global[np.ix_(idx, idx)] += me
        return k_global, m_global

    def solve(
        self,
        u_initial: Union[float, Sequence[float]],
        dt: float,
        n_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """March the excess pore pressure forward in time.

        Parameters
        ----------
        u_initial
            Initial excess pore pressure (scalar or nodal array).
        dt
            Time step.
        n_steps
            Number of steps.

        Returns
        -------
        (times, history)
            Times ``(n_steps+1,)`` and nodal excess-pressure fields
            ``(n_steps+1, n_nodes)``.
        """
        n = self.mesh.n_nodes
        u = (np.full(n, float(u_initial)) if np.ndim(u_initial) == 0
             else np.asarray(u_initial, dtype=float).copy())
        fixed_idx = np.array(sorted(self._fixed), dtype=int)
        if fixed_idx.size:
            u[fixed_idx] = 0.0
        free = np.ones(n, dtype=bool)
        free[fixed_idx] = False
        free_idx = np.where(free)[0]

        k_global, m_global = self.assemble()
        system = m_global + dt * k_global
        a_ff = system[np.ix_(free_idx, free_idx)]

        history = [u.copy()]
        times = [0.0]
        for step in range(n_steps):
            rhs = (m_global @ u)[free_idx]
            u_new = u.copy()
            u_new[free_idx] = np.linalg.solve(a_ff, rhs)
            if fixed_idx.size:
                u_new[fixed_idx] = 0.0
            u = u_new
            history.append(u.copy())
            times.append((step + 1) * dt)
        return np.array(times), np.array(history)

    def average_degree(
        self, u_field: np.ndarray, u_initial: float
    ) -> float:
        """Area-weighted average degree of consolidation of a field.

        :math:`U = 1 - \\bar{u}_e / u_0`, with the mean taken over the mesh.
        """
        # Area-weighted nodal average via the lumped mass (row sums of M).
        _, m_global = self.assemble()
        weights = m_global.sum(axis=1)
        mean_u = float(np.sum(weights * u_field) / np.sum(weights))
        return 1.0 - mean_u / u_initial
