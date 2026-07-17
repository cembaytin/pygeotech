"""Unit tests for the shear_strength submodule."""

import math

import numpy as np
import pytest

from pygeotech.shear_strength import (
    MohrCoulomb,
    principal_stresses_at_failure,
    stress_path_pq,
)


class TestMohrCoulomb:
    def test_shear_strength(self) -> None:
        env = MohrCoulomb(cohesion=10.0, friction_angle=30.0)
        assert env.shear_strength(100.0) == pytest.approx(
            10.0 + 100.0 * math.tan(math.radians(30.0)))

    def test_fit_direct_shear_recovers_parameters(self) -> None:
        env_true = MohrCoulomb(cohesion=15.0, friction_angle=28.0)
        sigma = np.array([50.0, 100.0, 150.0, 200.0])
        tau = env_true.cohesion + sigma * env_true.tan_phi
        fitted = MohrCoulomb.fit_direct_shear(sigma, tau)
        assert fitted.cohesion == pytest.approx(15.0, abs=1e-6)
        assert fitted.friction_angle == pytest.approx(28.0, abs=1e-6)

    def test_fit_triaxial_recovers_parameters(self) -> None:
        env_true = MohrCoulomb(cohesion=20.0, friction_angle=25.0)
        sigma3 = np.array([50.0, 100.0, 150.0, 200.0])
        sigma1 = np.array(
            [principal_stresses_at_failure(env_true, s3) for s3 in sigma3])
        fitted = MohrCoulomb.fit_triaxial(sigma3, sigma1)
        assert fitted.cohesion == pytest.approx(20.0, abs=1e-4)
        assert fitted.friction_angle == pytest.approx(25.0, abs=1e-4)

    def test_kf_line_conversion(self) -> None:
        env = MohrCoulomb(cohesion=20.0, friction_angle=25.0)
        a, tan_alpha = env.kf_line()
        assert tan_alpha == pytest.approx(math.sin(math.radians(25.0)))
        assert a == pytest.approx(20.0 * math.cos(math.radians(25.0)))

    def test_frictionless_clay_su(self) -> None:
        # phi = 0: sigma1 - sigma3 = 2 c (undrained strength).
        env = MohrCoulomb(cohesion=40.0, friction_angle=0.0)
        s1 = principal_stresses_at_failure(env, 100.0)
        assert s1 - 100.0 == pytest.approx(80.0)


class TestStressPath:
    def test_total_and_effective_paths(self) -> None:
        s1 = np.array([100.0, 150.0, 200.0])
        s3 = np.array([100.0, 100.0, 100.0])
        u = np.array([0.0, 20.0, 45.0])
        p, q, p_eff = stress_path_pq(s1, s3, u)
        assert np.allclose(p, (s1 + s3) / 2)
        assert np.allclose(q, (s1 - s3) / 2)
        assert np.allclose(p_eff, p - u)

    def test_effective_equals_total_without_u(self) -> None:
        s1 = np.array([120.0, 160.0])
        s3 = np.array([80.0, 80.0])
        p, q, p_eff = stress_path_pq(s1, s3)
        assert np.allclose(p, p_eff)
