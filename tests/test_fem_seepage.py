"""Validation tests for the finite-element seepage engine.

The linear-triangle FEM reproduces linear head fields exactly, so a 1-D
confined-flow benchmark (linear head, Dupuit flow rate) is an exact check.
"""

import numpy as np
import pytest

from pygeotech.core import PyGeotechError, SoilMaterial
from pygeotech.fem import SeepageFEM, rectangular_mesh
from pygeotech.fem.mesh import TriMesh


class TestMesh:
    def test_rectangular_mesh_counts(self) -> None:
        mesh = rectangular_mesh(10.0, 4.0, 5, 2)
        assert mesh.n_nodes == (5 + 1) * (2 + 1)
        assert mesh.n_elements == 2 * 5 * 2

    def test_all_areas_positive_ccw(self) -> None:
        mesh = rectangular_mesh(3.0, 2.0, 4, 3)
        assert np.all(mesh.element_areas() > 0.0)

    def test_total_area_conserved(self) -> None:
        mesh = rectangular_mesh(7.0, 5.0, 6, 4)
        assert mesh.element_areas().sum() == pytest.approx(35.0)

    def test_nodes_where(self) -> None:
        mesh = rectangular_mesh(10.0, 4.0, 10, 4)
        left = mesh.nodes_where("x", 0.0)
        assert len(left) == 5
        assert np.allclose(mesh.nodes[left, 0], 0.0)


class TestSeepage1D:
    """Confined flow in a box: left head H1, right head H2, no-flow top/bottom."""

    def _solve(self, nx=12, ny=4, k=1e-5, h1=12.0, h2=10.0):
        w, h = 10.0, 4.0
        mesh = rectangular_mesh(w, h, nx, ny)
        fem = SeepageFEM(mesh, conductivity=k)
        fem.set_head(mesh.nodes_where("x", 0.0), h1)
        fem.set_head(mesh.nodes_where("x", w), h2)
        head = fem.solve()
        return mesh, fem, head, (w, h, k, h1, h2)

    def test_linear_head_field(self) -> None:
        mesh, fem, head, (w, h, k, h1, h2) = self._solve()
        # Exact analytical: h(x) = h1 + (h2 - h1) * x / w
        expected = h1 + (h2 - h1) * mesh.nodes[:, 0] / w
        assert np.allclose(head, expected, atol=1e-9)

    def test_midplane_head(self) -> None:
        mesh, fem, head, (w, h, k, h1, h2) = self._solve()
        mid = mesh.nodes_where("x", 5.0)
        assert head[mid].mean() == pytest.approx(11.0, abs=1e-9)

    def test_uniform_gradient_and_velocity(self) -> None:
        mesh, fem, head, (w, h, k, h1, h2) = self._solve()
        grads = fem.gradients()
        vel = fem.velocities()
        # dh/dx = (h2 - h1)/w = -0.2 ; dh/dy = 0
        assert np.allclose(grads[:, 0], (h2 - h1) / w, atol=1e-9)
        assert np.allclose(grads[:, 1], 0.0, atol=1e-9)
        # v_x = -k dh/dx = k * 0.2
        assert np.allclose(vel[:, 0], -k * (h2 - h1) / w, atol=1e-15)

    def test_flow_rate_matches_darcy(self) -> None:
        mesh, fem, head, (w, h, k, h1, h2) = self._solve()
        # Q = k * i * A = k * (h1 - h2)/w * (h * 1)
        q_expected = k * (h1 - h2) / w * h
        q_in = fem.boundary_flow(mesh.nodes_where("x", 0.0))
        q_out = fem.boundary_flow(mesh.nodes_where("x", w))
        assert q_in == pytest.approx(q_expected, rel=1e-9)
        # Continuity: inflow balances outflow.
        assert q_in == pytest.approx(-q_out, rel=1e-9)

    def test_refinement_still_exact(self) -> None:
        for nx, ny in ((4, 2), (20, 10), (30, 6)):
            mesh, fem, head, (w, h, k, h1, h2) = self._solve(nx=nx, ny=ny)
            expected = h1 + (h2 - h1) * mesh.nodes[:, 0] / w
            assert np.allclose(head, expected, atol=1e-8)


class TestErrorsAndFeatures:
    def test_solve_without_bc_raises(self) -> None:
        mesh = rectangular_mesh(4.0, 2.0, 2, 2)
        with pytest.raises(PyGeotechError):
            SeepageFEM(mesh).solve()

    def test_anisotropic_conductivity(self) -> None:
        # ky != kx must not change a purely horizontal 1-D flow field.
        w, h = 10.0, 4.0
        mesh = rectangular_mesh(w, h, 10, 4)
        fem = SeepageFEM(mesh, conductivity=(1e-4, 1e-6))
        fem.set_head(mesh.nodes_where("x", 0.0), 5.0)
        fem.set_head(mesh.nodes_where("x", w), 0.0)
        head = fem.solve()
        expected = 5.0 * (1.0 - mesh.nodes[:, 0] / w)
        assert np.allclose(head, expected, atol=1e-8)

    def test_material_conductivity_helper(self) -> None:
        clay = SoilMaterial(name="clay", permeability=1e-9)
        assert clay.conductivity() == (1e-9, 1e-9)
        with pytest.raises(PyGeotechError):
            SoilMaterial(name="unknown").conductivity()

    def test_heterogeneous_per_element_k(self) -> None:
        mesh = rectangular_mesh(10.0, 4.0, 10, 4)
        k = np.full(mesh.n_elements, 1e-5)
        fem = SeepageFEM(mesh, conductivity=k)
        fem.set_head(mesh.nodes_where("x", 0.0), 2.0)
        fem.set_head(mesh.nodes_where("x", 10.0), 0.0)
        head = fem.solve()
        assert head[mesh.nodes_where("x", 5.0)].mean() == pytest.approx(1.0,
                                                                        abs=1e-8)
