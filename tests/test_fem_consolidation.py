"""Validation tests for the transient FEM consolidation kernel."""

import numpy as np
import pytest

from pygeotech.consolidation import average_degree_of_consolidation
from pygeotech.fem import ConsolidationFEM, rectangular_mesh


def _terzaghi_column(h=4.0, cv=1.0, ny=80):
    mesh = rectangular_mesh(0.2, h, 1, ny)
    fem = ConsolidationFEM(mesh, cv=cv)
    fem.set_drainage(mesh.nodes_where("y", 0.0))
    fem.set_drainage(mesh.nodes_where("y", h))
    return mesh, fem


class TestTerzaghiValidation:
    def test_average_degree_matches_analytical(self) -> None:
        h, cv = 4.0, 1.0
        mesh, fem = _terzaghi_column(h, cv)
        t, u = fem.solve(u_initial=100.0, dt=0.02, n_steps=200)
        h_dr = h / 2.0
        for step in (25, 50, 100, 200):
            tv = cv * t[step] / h_dr ** 2
            u_fem = fem.average_degree(u[step], 100.0)
            u_analytic = average_degree_of_consolidation(tv)
            assert u_fem == pytest.approx(u_analytic, abs=5e-3)

    def test_monotonic_dissipation(self) -> None:
        mesh, fem = _terzaghi_column()
        t, u = fem.solve(u_initial=100.0, dt=0.02, n_steps=100)
        degrees = [fem.average_degree(u[s], 100.0) for s in range(0, 101, 20)]
        assert all(b >= a for a, b in zip(degrees, degrees[1:]))
        # At t=0 only the drained-boundary nodes are zero, so U starts small.
        assert degrees[0] < 0.05

    def test_drained_boundaries_stay_zero(self) -> None:
        mesh, fem = _terzaghi_column()
        t, u = fem.solve(u_initial=100.0, dt=0.02, n_steps=10)
        top = mesh.nodes_where("y", 0.0)
        assert np.allclose(u[-1][top], 0.0)


class TestFeatures:
    def test_single_drainage_slower(self) -> None:
        # Single drainage (top only) consolidates slower than double.
        h = 4.0
        mesh_d = rectangular_mesh(0.2, h, 1, 60)
        double = ConsolidationFEM(mesh_d, cv=1.0)
        double.set_drainage(mesh_d.nodes_where("y", 0.0))
        double.set_drainage(mesh_d.nodes_where("y", h))
        single = ConsolidationFEM(mesh_d, cv=1.0)
        single.set_drainage(mesh_d.nodes_where("y", 0.0))
        _, ud = double.solve(100.0, 0.05, 40)
        _, us = single.solve(100.0, 0.05, 40)
        assert (double.average_degree(ud[-1], 100.0)
                > single.average_degree(us[-1], 100.0))

    def test_higher_cv_faster(self) -> None:
        mesh, fast = _terzaghi_column(cv=4.0, ny=60)
        _, slow_mesh = _terzaghi_column(cv=1.0, ny=60)
        _, uf = fast.solve(100.0, 0.02, 50)
        _, us = slow_mesh.solve(100.0, 0.02, 50)
        assert (fast.average_degree(uf[-1], 100.0)
                > slow_mesh.average_degree(us[-1], 100.0))

    def test_invalid_cv_shape(self) -> None:
        mesh = rectangular_mesh(1.0, 1.0, 2, 2)
        with pytest.raises(ValueError):
            ConsolidationFEM(mesh, cv=[1.0, 2.0])
