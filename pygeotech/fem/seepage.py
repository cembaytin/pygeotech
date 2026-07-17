"""Finite-element solver for 2-D steady-state confined seepage.

Governing equation
------------------
Steady groundwater flow in a (possibly anisotropic, heterogeneous)
saturated porous medium obeys the Laplace/Poisson equation in the total
head :math:`h`,

.. math::

    \\frac{\\partial}{\\partial x}\\!\\left(k_x \\frac{\\partial h}{\\partial x}\\right)
    + \\frac{\\partial}{\\partial y}\\!\\left(k_y \\frac{\\partial h}{\\partial y}\\right)
    + Q = 0 .

Using three-node linear triangles the element conductance matrix is

.. math::

    K^e_{ij} = \\frac{1}{4A}\\left(k_x\\,b_i b_j + k_y\\,c_i c_j\\right),
    \\qquad
    b_i = y_j - y_k,\\; c_i = x_k - x_j,

with :math:`A` the element area and :math:`(i, j, k)` cyclic. Prescribed
heads (Dirichlet) are imposed by static condensation; impervious
boundaries are the natural (zero-flux) condition, so they need no
explicit treatment. Darcy velocities follow from the constant element
gradient, :math:`\\mathbf{v} = -\\mathbf{k}\\,\\nabla h`.

This is the first physics kernel of the pyGeotech finite-element engine;
the same assembly machinery generalises to consolidation (transient
diffusion) and plane elasticity.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from pygeotech.core import PyGeotechError
from pygeotech.fem.mesh import TriMesh

__all__ = ["SeepageFEM"]

Conductivity = Union[float, Tuple[float, float], Sequence[float]]


class SeepageFEM:
    """Steady-state 2-D confined-seepage finite-element solver.

    Parameters
    ----------
    mesh
        A :class:`~pygeotech.fem.mesh.TriMesh`.
    conductivity
        Hydraulic conductivity. Either a scalar (isotropic), a pair
        ``(kx, ky)`` applied to every element, or an array of length
        ``n_elements`` (per-element scalar) / shape ``(n_elements, 2)``
        (per-element anisotropic) for heterogeneous domains.

    Notes
    -----
    Call :meth:`set_head` to prescribe fixed-head boundaries, optionally
    :meth:`add_flux` for nodal inflow, then :meth:`solve`.

    Examples
    --------
    >>> from pygeotech.fem.mesh import rectangular_mesh
    >>> mesh = rectangular_mesh(10.0, 4.0, 10, 4)
    >>> fem = SeepageFEM(mesh, conductivity=1e-5)
    >>> _ = fem.set_head(mesh.nodes_where("x", 0.0), 12.0)
    >>> _ = fem.set_head(mesh.nodes_where("x", 10.0), 10.0)
    >>> head = fem.solve()
    >>> bool(abs(head[mesh.nodes_where("x", 5.0)].mean() - 11.0) < 1e-6)
    True
    """

    def __init__(self, mesh: TriMesh, conductivity: Conductivity = 1.0) -> None:
        self.mesh = mesh
        self._kx, self._ky = self._broadcast_conductivity(conductivity)
        self._fixed: Dict[int, float] = {}
        self._flux = np.zeros(mesh.n_nodes, dtype=float)
        self._head: Optional[np.ndarray] = None
        self._reactions: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _broadcast_conductivity(
        self, conductivity: Conductivity
    ) -> Tuple[np.ndarray, np.ndarray]:
        m = self.mesh.n_elements
        arr = np.asarray(conductivity, dtype=float)
        if arr.ndim == 0:                       # isotropic scalar
            return np.full(m, float(arr)), np.full(m, float(arr))
        if arr.shape == (2,):                   # global (kx, ky)
            return np.full(m, arr[0]), np.full(m, arr[1])
        if arr.shape == (m,):                    # per-element isotropic
            return arr.copy(), arr.copy()
        if arr.shape == (m, 2):                  # per-element anisotropic
            return arr[:, 0].copy(), arr[:, 1].copy()
        raise ValueError(
            "conductivity must be a scalar, (kx, ky), (n_elements,) or "
            "(n_elements, 2) array.")

    def set_head(self, nodes: Sequence[int], value: float) -> "SeepageFEM":
        """Prescribe a fixed total head at the given node indices."""
        for n in np.atleast_1d(nodes):
            self._fixed[int(n)] = float(value)
        self._head = None
        return self

    def add_flux(self, nodes: Sequence[int], value: float) -> "SeepageFEM":
        """Add a nodal inflow ``value`` [m^3/s] at the given nodes."""
        for n in np.atleast_1d(nodes):
            self._flux[int(n)] += float(value)
        self._head = None
        return self

    # ------------------------------------------------------------------
    # Assembly and solution
    # ------------------------------------------------------------------
    def assemble(self) -> np.ndarray:
        """Assemble and return the global conductance matrix ``(n, n)``."""
        mesh = self.mesh
        n = mesh.n_nodes
        k_global = np.zeros((n, n), dtype=float)
        coords = mesh.nodes
        for e, tri in enumerate(mesh.elements):
            i, j, k = tri
            (xi, yi), (xj, yj), (xk, yk) = coords[i], coords[j], coords[k]
            area = 0.5 * ((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
            if abs(area) < 1e-15:
                raise PyGeotechError(f"degenerate element {e} with ~zero area.")
            b = np.array([yj - yk, yk - yi, yi - yj])
            c = np.array([xk - xj, xi - xk, xj - xi])
            ke = (self._kx[e] * np.outer(b, b)
                  + self._ky[e] * np.outer(c, c)) / (4.0 * area)
            idx = np.array([i, j, k])
            k_global[np.ix_(idx, idx)] += ke
        return k_global

    def solve(self) -> np.ndarray:
        """Solve for the nodal total head and return it as an array.

        Returns
        -------
        ndarray
            Total head at every node [m].

        Raises
        ------
        PyGeotechError
            If no fixed-head boundary was prescribed (singular system).
        """
        if not self._fixed:
            raise PyGeotechError(
                "at least one fixed-head boundary must be set before solving.")
        n = self.mesh.n_nodes
        k_global = self.assemble()

        fixed_idx = np.array(sorted(self._fixed), dtype=int)
        fixed_val = np.array([self._fixed[i] for i in fixed_idx], dtype=float)
        free_mask = np.ones(n, dtype=bool)
        free_mask[fixed_idx] = False
        free_idx = np.where(free_mask)[0]

        head = np.zeros(n, dtype=float)
        head[fixed_idx] = fixed_val
        if free_idx.size:
            k_ff = k_global[np.ix_(free_idx, free_idx)]
            k_fc = k_global[np.ix_(free_idx, fixed_idx)]
            rhs = self._flux[free_idx] - k_fc @ fixed_val
            head[free_idx] = np.linalg.solve(k_ff, rhs)

        self._head = head
        # Nodal reactions (net flow) = K h - f; nonzero only at BC nodes.
        self._reactions = k_global @ head - self._flux
        return head

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------
    @property
    def head(self) -> np.ndarray:
        """Nodal total-head array from the last :meth:`solve`."""
        if self._head is None:
            raise PyGeotechError("call solve() first.")
        return self._head

    def gradients(self) -> np.ndarray:
        """Hydraulic gradient ``(dh/dx, dh/dy)`` per element ``(m, 2)``."""
        head = self.head
        mesh = self.mesh
        coords = mesh.nodes
        grads = np.zeros((mesh.n_elements, 2), dtype=float)
        for e, tri in enumerate(mesh.elements):
            i, j, k = tri
            (xi, yi), (xj, yj), (xk, yk) = coords[i], coords[j], coords[k]
            area = 0.5 * ((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
            b = np.array([yj - yk, yk - yi, yi - yj])
            c = np.array([xk - xj, xi - xk, xj - xi])
            he = head[[i, j, k]]
            grads[e, 0] = b @ he / (2.0 * area)
            grads[e, 1] = c @ he / (2.0 * area)
        return grads

    def velocities(self) -> np.ndarray:
        """Darcy velocity ``v = -k grad(h)`` per element ``(m, 2)`` [m/s]."""
        grads = self.gradients()
        vel = np.empty_like(grads)
        vel[:, 0] = -self._kx * grads[:, 0]
        vel[:, 1] = -self._ky * grads[:, 1]
        return vel

    def boundary_flow(self, nodes: Sequence[int]) -> float:
        """Net flow [m^3/s] across the given (fixed-head) boundary nodes.

        Computed from the nodal reactions; a positive value is inflow into
        the domain.
        """
        if self._reactions is None:
            raise PyGeotechError("call solve() first.")
        idx = np.atleast_1d(np.asarray(nodes, dtype=int))
        # Reaction R = K h - f is the flux injected to sustain a fixed head;
        # it is positive where water enters the domain (inflow boundary).
        return float(np.sum(self._reactions[idx]))
