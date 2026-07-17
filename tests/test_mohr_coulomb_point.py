"""Tests for the Mohr-Coulomb elastoplastic stress-point driver.

Validated against the analytical triaxial failure ratio q_f/p' = M_phi."""

import math

import pytest

from pygeotech.constitutive import MohrCoulombParameters, mc_triaxial_test


def _mc(phi, c=0.0, psi=0.0):
    return MohrCoulombParameters(youngs_modulus=2.0e4, poisson_ratio=0.3,
                                 friction_angle=phi, cohesion=c,
                                 dilation_angle=psi)


class TestFailureRatio:
    @pytest.mark.parametrize("phi", [20.0, 25.0, 30.0, 35.0, 40.0])
    def test_drained_reaches_M_phi(self, phi) -> None:
        p = _mc(phi)
        res = mc_triaxial_test(p, p0=100.0, drained=True)
        assert res.q[-1] / res.p_eff[-1] == pytest.approx(p.M_phi(), rel=1e-3)

    def test_cohesion_intercept(self) -> None:
        p = _mc(20.0, c=15.0)
        res = mc_triaxial_test(p, p0=100.0, drained=True)
        phi = math.radians(20.0)
        k_c = 6.0 * 15.0 * math.cos(phi) / (3.0 - math.sin(phi))
        assert res.q[-1] == pytest.approx(p.M_phi() * res.p_eff[-1] + k_c,
                                          rel=1e-3)

    def test_M_phi_formula(self) -> None:
        assert _mc(30.0).M_phi() == pytest.approx(1.2, abs=1e-6)


class TestBehaviour:
    def test_elastic_then_plateau(self) -> None:
        # q rises then is capped at the failure line (perfectly plastic).
        p = _mc(30.0)
        res = mc_triaxial_test(p, p0=100.0, drained=True)
        assert res.q[1] > 0.0
        # Late increments add almost no deviatoric stress (plastic plateau).
        assert (res.q[-1] - res.q[-50]) < 0.02 * res.q[-1]

    def test_stronger_soil_higher_strength(self) -> None:
        weak = mc_triaxial_test(_mc(25.0), p0=100.0, drained=True)
        strong = mc_triaxial_test(_mc(38.0), p0=100.0, drained=True)
        assert strong.q[-1] > weak.q[-1]

    def test_undrained_runs(self) -> None:
        res = mc_triaxial_test(_mc(28.0, c=5.0), p0=100.0, drained=False)
        assert res.q[-1] > 0.0
        assert res.volumetric_strain[-1] == pytest.approx(0.0, abs=1e-9)
