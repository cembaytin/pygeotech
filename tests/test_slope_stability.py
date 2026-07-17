"""Unit tests for the slope_stability submodule."""

import math

import numpy as np
import pytest

from pygeotech.slope_stability import (
    SlipCircle,
    critical_circle,
    infinite_slope_factor,
    simple_slope_surface,
    slope_factor_of_safety,
)


class TestInfiniteSlope:
    def test_dry_cohesionless(self) -> None:
        # F = tan(phi)/tan(beta).
        f = infinite_slope_factor(20.0, 30.0)
        assert f == pytest.approx(math.tan(math.radians(30))
                                  / math.tan(math.radians(20)), rel=1e-9)

    def test_at_repose_unity(self) -> None:
        assert infinite_slope_factor(30.0, 30.0) == pytest.approx(1.0)

    def test_seepage_reduces_fos(self) -> None:
        dry = infinite_slope_factor(20.0, 30.0, cohesion=5.0, gamma=20.0,
                                    depth=3.0)
        wet = infinite_slope_factor(20.0, 30.0, cohesion=5.0, depth=3.0,
                                    seepage=True, gamma_sat=20.0)
        assert wet < dry

    def test_cohesion_helps(self) -> None:
        assert (infinite_slope_factor(25.0, 25.0, cohesion=10.0, depth=2.0)
                > infinite_slope_factor(25.0, 25.0, cohesion=0.0, depth=2.0))


class TestMethodOfSlices:
    def _setup(self):
        ground = simple_slope_surface(height=10.0, slope_angle=30.0)
        circle = SlipCircle(xc=12.0, yc=20.0, radius=22.0)
        return ground, circle

    def test_phi_zero_fellenius_equals_bishop(self) -> None:
        # For phi = 0 the two methods are algebraically identical.
        ground, circle = self._setup()
        common = dict(gamma=19.0, cohesion=40.0, friction_angle=0.0,
                      n_slices=60)
        f_fel = slope_factor_of_safety(ground, circle, method="fellenius",
                                       **common)
        f_bis = slope_factor_of_safety(ground, circle, method="bishop",
                                       **common)
        assert f_fel == pytest.approx(f_bis, rel=1e-4)

    def test_bishop_ge_fellenius(self) -> None:
        # Bishop generally returns a slightly higher FoS than Fellenius.
        ground, circle = self._setup()
        common = dict(gamma=19.0, cohesion=15.0, friction_angle=25.0,
                      n_slices=60)
        f_fel = slope_factor_of_safety(ground, circle, method="fellenius",
                                       **common)
        f_bis = slope_factor_of_safety(ground, circle, method="bishop",
                                       **common)
        assert f_bis >= f_fel - 1e-6
        assert f_bis > 0.0

    def test_stronger_soil_higher_fos(self) -> None:
        ground, circle = self._setup()
        weak = slope_factor_of_safety(ground, circle, gamma=19.0,
                                      cohesion=10.0, friction_angle=20.0)
        strong = slope_factor_of_safety(ground, circle, gamma=19.0,
                                        cohesion=30.0, friction_angle=30.0)
        assert strong > weak

    def test_pore_pressure_reduces_fos(self) -> None:
        ground, circle = self._setup()
        dry = slope_factor_of_safety(ground, circle, gamma=19.0,
                                     cohesion=15.0, friction_angle=25.0,
                                     ru=0.0)
        wet = slope_factor_of_safety(ground, circle, gamma=19.0,
                                     cohesion=15.0, friction_angle=25.0,
                                     ru=0.4)
        assert wet < dry

    def test_missing_circle_raises(self) -> None:
        ground = simple_slope_surface(10.0, 30.0)
        # A tiny circle far above the ground never intersects the slope.
        circle = SlipCircle(xc=5.0, yc=100.0, radius=1.0)
        with pytest.raises(ValueError):
            slope_factor_of_safety(ground, circle, gamma=19.0, cohesion=10.0,
                                   friction_angle=25.0)


class TestCriticalCircle:
    def test_search_returns_reasonable_fos(self) -> None:
        ground = simple_slope_surface(height=10.0, slope_angle=30.0)
        fos, circle = critical_circle(
            ground, gamma=19.0, cohesion=20.0, friction_angle=25.0,
            height=10.0, slope_angle=30.0, method="bishop")
        assert 0.5 < fos < 5.0
        assert isinstance(circle, SlipCircle)
        # The minimum must not exceed an arbitrary trial circle's FoS.
        trial = slope_factor_of_safety(ground, SlipCircle(12.0, 20.0, 22.0),
                                       gamma=19.0, cohesion=20.0,
                                       friction_angle=25.0)
        assert fos <= trial + 1e-6
