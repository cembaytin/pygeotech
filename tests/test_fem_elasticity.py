"""Validation tests for the plane-elasticity FEM kernel."""

import numpy as np
import pytest

from pygeotech.core import PyGeotechError
from pygeotech.fem import ElasticityFEM, rectangular_mesh


class TestUniaxialBar:
    """Uniaxial bar: fixed base, axial pull -> delta = PL/AE, sigma = P/A."""

    def _bar(self, nu=0.0):
        w, length, t, e, p = 2.0, 10.0, 1.0, 1.0e4, 100.0
        mesh = rectangular_mesh(w, length, 4, 20)
        fem = ElasticityFEM(mesh, youngs_modulus=e, poisson_ratio=nu,
                            thickness=t, plane_strain=False)
        fem.fix(mesh.nodes_where("y", 0.0))
        top = mesh.nodes_where("y", length)
        fem.add_force(top, fy=p / len(top))
        u = fem.solve()
        return mesh, fem, u, (w, length, t, e, p, top)

    def test_axial_stress_uniform(self) -> None:
        mesh, fem, u, (w, length, t, e, p, top) = self._bar()
        syy = fem.element_stresses()[:, 1]
        assert syy.mean() == pytest.approx(p / (w * t), rel=1e-6)
        assert syy.std() / syy.mean() < 0.1        # near-uniform

    def test_tip_displacement(self) -> None:
        mesh, fem, u, (w, length, t, e, p, top) = self._bar()
        delta = p * length / (w * t * e)
        assert u[top, 1].mean() == pytest.approx(delta, rel=0.02)

    def test_load_scaling_linear(self) -> None:
        mesh, fem, u1, (w, length, t, e, p, top) = self._bar()
        # Double the load -> double the displacement (linear elasticity).
        fem.add_force(top, fy=p / len(top))       # add the same again
        u2 = fem.solve()
        assert u2[top, 1].mean() == pytest.approx(2.0 * u1[top, 1].mean(),
                                                  rel=1e-6)


class TestFeatures:
    def test_poisson_lateral_contraction(self) -> None:
        # With nu > 0 an axially stretched bar contracts laterally.
        w, length = 2.0, 10.0
        mesh = rectangular_mesh(w, length, 4, 20)
        fem = ElasticityFEM(mesh, youngs_modulus=1e4, poisson_ratio=0.3,
                            thickness=1.0, plane_strain=False)
        fem.fix(mesh.nodes_where("x", 0.0), x=True, y=False)
        fem.fix(mesh.nodes_where("y", 0.0), x=False, y=True)
        top = mesh.nodes_where("y", length)
        fem.add_force(top, fy=100.0 / len(top))
        u = fem.solve()
        right = mesh.nodes_where("x", w)
        assert u[right, 0].mean() < 0.0            # moves inward

    def test_von_mises_positive(self) -> None:
        mesh = rectangular_mesh(2.0, 4.0, 3, 6)
        fem = ElasticityFEM(mesh, youngs_modulus=1e4, poisson_ratio=0.25)
        fem.fix(mesh.nodes_where("y", 0.0))
        fem.add_force(mesh.nodes_where("y", 4.0), fx=10.0)
        fem.solve()
        assert np.all(fem.von_mises() >= 0.0)

    def test_requires_support(self) -> None:
        mesh = rectangular_mesh(1.0, 1.0, 2, 2)
        with pytest.raises(PyGeotechError):
            ElasticityFEM(mesh, 1e4, 0.3).solve()

    def test_stress_before_solve_raises(self) -> None:
        mesh = rectangular_mesh(1.0, 1.0, 2, 2)
        with pytest.raises(PyGeotechError):
            ElasticityFEM(mesh, 1e4, 0.3).element_stresses()
