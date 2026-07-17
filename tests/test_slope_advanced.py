"""Tests for advanced slope stability: layered soils, non-circular, Spencer."""

import math

import pytest

from pygeotech.slope_stability import (
    SlipCircle,
    simple_slope_surface,
    slope_factor_of_safety,
)
from pygeotech.slope_stability.advanced import (
    LayeredSoil,
    PolylineSurface,
    SlopeLayer,
    advanced_factor_of_safety,
)


def _setup():
    ground = simple_slope_surface(height=10.0, slope_angle=30.0)
    circle = SlipCircle(12.0, 20.0, 22.0)
    return ground, circle


class TestReducesToLegacy:
    """The layered engine must match the homogeneous solver exactly."""

    def test_fellenius_matches(self) -> None:
        ground, circle = _setup()
        soil = LayeredSoil.homogeneous(19.0, 15.0, 25.0)
        legacy = slope_factor_of_safety(ground, circle, 19.0, 15.0, 25.0,
                                        method="fellenius")
        adv = advanced_factor_of_safety(ground, circle, soil,
                                        method="fellenius")
        assert adv == pytest.approx(legacy, rel=1e-9)

    def test_bishop_matches(self) -> None:
        ground, circle = _setup()
        soil = LayeredSoil.homogeneous(19.0, 15.0, 25.0)
        legacy = slope_factor_of_safety(ground, circle, 19.0, 15.0, 25.0,
                                        method="bishop")
        adv = advanced_factor_of_safety(ground, circle, soil, method="bishop")
        assert adv == pytest.approx(legacy, rel=1e-6)


class TestSpencer:
    def test_spencer_close_to_bishop_circular(self) -> None:
        # Spencer and Bishop agree within ~2% for a circular surface.
        ground, circle = _setup()
        soil = LayeredSoil.homogeneous(19.0, 15.0, 25.0)
        bishop = advanced_factor_of_safety(ground, circle, soil,
                                           method="bishop")
        spencer = advanced_factor_of_safety(ground, circle, soil,
                                            method="spencer")
        assert spencer == pytest.approx(bishop, rel=0.02)

    def test_spencer_positive(self) -> None:
        ground, circle = _setup()
        soil = LayeredSoil.homogeneous(20.0, 10.0, 28.0)
        assert advanced_factor_of_safety(ground, circle, soil,
                                         method="spencer") > 0.0


class TestLayered:
    def test_weak_lower_layer_reduces_fos(self) -> None:
        ground, circle = _setup()
        homog = LayeredSoil.homogeneous(19.0, 25.0, 32.0)
        layered = LayeredSoil([
            SlopeLayer(1e9, 19.0, 25.0, 32.0),
            SlopeLayer(3.0, 18.0, 5.0, 18.0)])       # weak seam below y=3
        f_homog = advanced_factor_of_safety(ground, circle, homog,
                                            method="bishop")
        f_layered = advanced_factor_of_safety(ground, circle, layered,
                                              method="bishop")
        assert f_layered < f_homog

    def test_strength_and_weight_lookup(self) -> None:
        soil = LayeredSoil([
            SlopeLayer(10.0, 18.0, 20.0, 30.0),
            SlopeLayer(4.0, 20.0, 5.0, 22.0)])
        assert soil.strength_at(6.0) == (20.0, 30.0)     # upper layer
        assert soil.strength_at(2.0) == (5.0, 22.0)      # lower layer
        # Weight from y=10 down to y=0: 18*(10-4) + 20*(4-0) = 188.
        assert soil.weight_per_width(10.0, 0.0) == pytest.approx(188.0)


class TestNonCircular:
    def test_polyline_surface_runs(self) -> None:
        ground = simple_slope_surface(10.0, 30.0)
        poly = PolylineSurface([(0, 0), (4, -2.5), (9, -2), (15, 3), (18, 10)])
        soil = LayeredSoil.homogeneous(19.0, 15.0, 25.0)
        for method in ("fellenius", "janbu", "spencer"):
            fos = advanced_factor_of_safety(ground, poly, soil, method=method)
            assert fos > 0.0

    def test_polyline_sorts_points(self) -> None:
        poly = PolylineSurface([(18, 10), (0, 0), (9, -2)])
        assert poly.x[0] == 0.0 and poly.x[-1] == 18.0

    def test_missing_surface_raises(self) -> None:
        ground = simple_slope_surface(10.0, 30.0)
        soil = LayeredSoil.homogeneous(19.0, 15.0, 25.0)
        with pytest.raises(ValueError):
            advanced_factor_of_safety(ground, SlipCircle(5.0, 100.0, 1.0),
                                      soil, method="bishop")
