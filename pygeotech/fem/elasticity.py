"""Plane (2-D) linear-elasticity kernel for the pyGeotech FEM engine.

Constant-strain triangles (CST) discretise the elastostatic problem

.. math:: \\nabla\\cdot\\boldsymbol{\\sigma} + \\mathbf{b} = 0,
    \\qquad \\boldsymbol{\\sigma} = \\mathbf{D}\\,\\boldsymbol{\\varepsilon},

with the plane-strain or plane-stress constitutive matrix
:math:`\\mathbf{D}`. Each node carries two displacement degrees of freedom
:math:`(u, v)`; the element stiffness is

.. math:: \\mathbf{K}^e = A\\,t\\,\\mathbf{B}^{\\mathsf T}\\mathbf{D}\\,\\mathbf{B},

where :math:`\\mathbf{B}` (3x6) is the constant strain-displacement matrix.
The global system is assembled and solved in **sparse** form
(``scipy.sparse``), so the engine scales to large meshes. Element stresses
follow from :math:`\\boldsymbol{\\sigma} = \\mathbf{D}\\,\\mathbf{B}\\,
\\mathbf{u}^e`.

The kernel reproduces the closed-form uniaxial-bar solution
(:math:`\\delta = PL/AE`, uniform :math:`\\sigma = P/A`), which is used to
validate it.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from pygeotech.core import PyGeotechError
from pygeotech.fem.mesh import TriMesh

__all__ = ["ElasticityFEM"]


class ElasticityFEM:
    """Plane-strain / plane-stress linear-elastic solver on a triangle mesh.

    Parameters
    ----------
    mesh
        A :class:`~pygeotech.fem.mesh.TriMesh`.
    youngs_modulus
        Young's modulus :math:`E` [kPa] (scalar or per-element).
    poisson_ratio
        Poisson's ratio :math:`\\nu`.
    thickness
        Out-of-plane thickness [m] (plane stress); 1.0 for plane strain.
    plane_strain
        ``True`` for plane strain (default), ``False`` for plane stress.

    Examples
    --------
    >>> from pygeotech.fem.mesh import rectangular_mesh
    >>> mesh = rectangular_mesh(2.0, 10.0, 2, 10)
    >>> fem = ElasticityFEM(mesh, youngs_modulus=1e4, poisson_ratio=0.0,
    ...                     plane_strain=False)
    >>> _ = fem.fix(mesh.nodes_where("y", 0.0))
    >>> _ = fem.add_force(mesh.nodes_where("y", 10.0), fy=50.0)  # total 100 kN
    >>> u = fem.solve()
    >>> bool(u[:, 1].max() > 0)
    True
    """

    def __init__(
        self,
        mesh: TriMesh,
        youngs_modulus: Union[float, np.ndarray] = 1.0e4,
        poisson_ratio: float = 0.3,
        thickness: float = 1.0,
        plane_strain: bool = True,
    ) -> None:
        self.mesh = mesh
        e_arr = np.asarray(youngs_modulus, dtype=float)
        self._e = (np.full(mesh.n_elements, float(e_arr)) if e_arr.ndim == 0
                   else e_arr)
        self.nu = poisson_ratio
        self.thickness = thickness
        self.plane_strain = plane_strain
        self._fixed: Dict[int, float] = {}       # dof -> prescribed value
        self._force = np.zeros(2 * mesh.n_nodes, dtype=float)
        self._disp: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def _constitutive(self, e: float) -> np.ndarray:
        nu = self.nu
        if self.plane_strain:
            factor = e / ((1.0 + nu) * (1.0 - 2.0 * nu))
            return factor * np.array([
                [1.0 - nu, nu, 0.0],
                [nu, 1.0 - nu, 0.0],
                [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0]])
        factor = e / (1.0 - nu ** 2)
        return factor * np.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0]])

    def _b_matrix(self, tri: np.ndarray) -> Tuple[np.ndarray, float]:
        i, j, k = tri
        (xi, yi), (xj, yj), (xk, yk) = self.mesh.nodes[[i, j, k]]
        area = 0.5 * ((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
        b = np.array([yj - yk, yk - yi, yi - yj])
        c = np.array([xk - xj, xi - xk, xj - xi])
        bmat = np.zeros((3, 6))
        for m in range(3):
            bmat[0, 2 * m] = b[m]
            bmat[1, 2 * m + 1] = c[m]
            bmat[2, 2 * m] = c[m]
            bmat[2, 2 * m + 1] = b[m]
        return bmat / (2.0 * area), area

    # ------------------------------------------------------------------
    def fix(self, nodes, x: bool = True, y: bool = True) -> "ElasticityFEM":
        """Constrain node displacements (support). ``x``/``y`` toggle axes."""
        for n in np.atleast_1d(nodes):
            if x:
                self._fixed[2 * int(n)] = 0.0
            if y:
                self._fixed[2 * int(n) + 1] = 0.0
        self._disp = None
        return self

    def add_force(self, nodes, fx: float = 0.0, fy: float = 0.0) -> "ElasticityFEM":
        """Add nodal forces ``(fx, fy)`` [kN] to each of ``nodes``."""
        for n in np.atleast_1d(nodes):
            self._force[2 * int(n)] += fx
            self._force[2 * int(n) + 1] += fy
        self._disp = None
        return self

    # ------------------------------------------------------------------
    def assemble(self) -> coo_matrix:
        """Assemble the global stiffness matrix as a sparse COO matrix."""
        ndof = 2 * self.mesh.n_nodes
        rows, cols, vals = [], [], []
        for e, tri in enumerate(self.mesh.elements):
            bmat, area = self._b_matrix(tri)
            if abs(area) < 1e-15:
                raise PyGeotechError(f"degenerate element {e}.")
            d = self._constitutive(self._e[e])
            ke = abs(area) * self.thickness * (bmat.T @ d @ bmat)
            dofs = np.array([2 * tri[0], 2 * tri[0] + 1,
                             2 * tri[1], 2 * tri[1] + 1,
                             2 * tri[2], 2 * tri[2] + 1])
            for a in range(6):
                for b in range(6):
                    rows.append(dofs[a])
                    cols.append(dofs[b])
                    vals.append(ke[a, b])
        return coo_matrix((vals, (rows, cols)), shape=(ndof, ndof))

    def solve(self) -> np.ndarray:
        """Solve for nodal displacements, returned as an ``(n_nodes, 2)`` array."""
        if not self._fixed:
            raise PyGeotechError("at least one support (fix) is required.")
        ndof = 2 * self.mesh.n_nodes
        k_global = self.assemble().tocsr()
        fixed = np.array(sorted(self._fixed), dtype=int)
        free = np.setdiff1d(np.arange(ndof), fixed)

        u = np.zeros(ndof)
        k_ff = k_global[free][:, free]
        rhs = self._force[free]
        u[free] = spsolve(k_ff.tocsc(), rhs)
        self._disp = u
        return u.reshape(-1, 2)

    # ------------------------------------------------------------------
    def element_stresses(self) -> np.ndarray:
        """Element stresses ``(sigma_xx, sigma_yy, tau_xy)`` per element."""
        if self._disp is None:
            raise PyGeotechError("call solve() first.")
        stresses = np.zeros((self.mesh.n_elements, 3))
        for e, tri in enumerate(self.mesh.elements):
            bmat, _ = self._b_matrix(tri)
            d = self._constitutive(self._e[e])
            dofs = np.array([2 * tri[0], 2 * tri[0] + 1,
                             2 * tri[1], 2 * tri[1] + 1,
                             2 * tri[2], 2 * tri[2] + 1])
            stresses[e] = d @ bmat @ self._disp[dofs]
        return stresses

    def von_mises(self) -> np.ndarray:
        """Element von Mises stress [kPa]."""
        s = self.element_stresses()
        sxx, syy, txy = s[:, 0], s[:, 1], s[:, 2]
        return np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * txy ** 2)
