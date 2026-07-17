"""Unit tests for the retaining_structures submodule."""

import math

import pytest

from pygeotech.retaining_structures import (
    GravityWall,
    active_thrust,
    coulomb_active_coefficient,
    rankine_active_coefficient,
    rankine_passive_coefficient,
)


class TestCoefficients:
    def test_rankine_level_backfill(self) -> None:
        assert rankine_active_coefficient(30.0) == pytest.approx(1 / 3, abs=1e-3)
        assert rankine_passive_coefficient(30.0) == pytest.approx(3.0, abs=1e-3)

    def test_active_passive_reciprocal(self) -> None:
        ka = rankine_active_coefficient(35.0)
        kp = rankine_passive_coefficient(35.0)
        assert ka * kp == pytest.approx(1.0, abs=1e-6)

    def test_coulomb_reduces_to_rankine(self) -> None:
        # delta = beta = theta = 0 -> Coulomb Ka equals Rankine Ka.
        ka_c = coulomb_active_coefficient(30.0, delta=0.0, beta=0.0, theta=0.0)
        assert ka_c == pytest.approx(rankine_active_coefficient(30.0), abs=1e-6)

    def test_wall_friction_reduces_ka(self) -> None:
        assert coulomb_active_coefficient(30.0, delta=20.0) < \
            coulomb_active_coefficient(30.0, delta=0.0)

    def test_sloped_backfill_increases_ka(self) -> None:
        assert rankine_active_coefficient(30.0, beta=15.0) > \
            rankine_active_coefficient(30.0, beta=0.0)


class TestThrust:
    def test_dry_cohesionless_thrust(self) -> None:
        # Pa = 0.5 Ka gamma H^2 ; line of action at H/3.
        h, gamma, phi = 5.0, 18.0, 30.0
        thrust, line = active_thrust(h, gamma, phi)
        ka = rankine_active_coefficient(phi)
        assert thrust == pytest.approx(0.5 * ka * gamma * h ** 2, rel=1e-3)
        assert line == pytest.approx(h / 3.0, rel=2e-2)

    def test_water_table_adds_pressure(self) -> None:
        dry, _ = active_thrust(5.0, 18.0, 30.0)
        wet, _ = active_thrust(5.0, 20.0, 30.0, water_table_depth=0.0)
        assert wet > dry


class TestGravityWall:
    def test_stability_checks(self) -> None:
        wall = GravityWall(height=5.0, base_width=3.0, weight=300.0,
                           weight_arm=1.5, base_friction_angle=25.0)
        thrust, line = active_thrust(5.0, 18.0, 30.0)
        res = wall.check(horizontal_thrust=thrust, thrust_arm=line)
        # Sliding resistance = W tan(25) / Pa
        expected_slide = (300.0 * math.tan(math.radians(25.0))) / thrust
        assert res.fs_sliding == pytest.approx(expected_slide, rel=1e-6)
        assert res.fs_overturning > 1.0
        assert res.bearing_pressure_max > res.bearing_pressure_min
