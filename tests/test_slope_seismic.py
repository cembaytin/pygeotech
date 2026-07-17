"""Tests for pile downdrag and seismic slope analysis."""

import math

import numpy as np
import pytest

from pygeotech.foundations import downdrag_force
from pygeotech.slope_stability import (
    SlipCircle,
    newmark_displacement,
    simple_slope_surface,
    slope_factor_of_safety,
    yield_acceleration,
)


class TestDowndrag:
    def test_drag_load_integration(self) -> None:
        depth = np.linspace(0.0, 10.0, 51)
        sve = 9.0 * depth
        q = downdrag_force(0.4, depth, sve, beta=0.25)
        expected = 0.25 * np.trapz(sve, depth) * (math.pi * 0.4)
        assert q == pytest.approx(expected, rel=1e-6)

    def test_neutral_plane_truncates(self) -> None:
        depth = np.linspace(0.0, 10.0, 101)
        sve = 9.0 * depth
        full = downdrag_force(0.4, depth, sve, beta=0.25)
        partial = downdrag_force(0.4, depth, sve, beta=0.25,
                                 neutral_plane_depth=5.0)
        assert partial < full


class TestSeismicSlope:
    def test_seismic_reduces_fos(self) -> None:
        ground = simple_slope_surface(height=10.0, slope_angle=30.0)
        circle = SlipCircle(12.0, 20.0, 22.0)
        common = dict(gamma=19.0, cohesion=20.0, friction_angle=25.0)
        static = slope_factor_of_safety(ground, circle, **common)
        seismic = slope_factor_of_safety(ground, circle,
                                         seismic_coefficient=0.15, **common)
        assert seismic < static


class TestNewmark:
    def test_no_displacement_below_yield(self) -> None:
        # Peak acceleration below a_y -> no sliding.
        t = np.linspace(0, 2, 400)
        a = 1.0 * np.sin(2 * math.pi * 2 * t)   # peak 1.0 m/s^2
        d, _ = newmark_displacement(a, dt=t[1] - t[0], yield_accel=2.0)
        assert d == pytest.approx(0.0, abs=1e-12)

    def test_single_pulse_displacement(self) -> None:
        # A constant acceleration pulse a>ay for a duration T.
        dt = 0.001
        n = 1000                       # 1 s pulse
        a = np.full(n, 3.0)            # m/s^2
        ay = 1.0
        d, hist = newmark_displacement(a, dt=dt, yield_accel=ay)
        # Analytic: d = 0.5 (a-ay) T^2 = 0.5*2*1 = 1.0 m.
        assert d == pytest.approx(1.0, rel=1e-2)
        assert hist[-1] == pytest.approx(d)

    def test_displacement_grows_with_shaking(self) -> None:
        dt = 0.01
        t = np.arange(0, 10, dt)
        weak = 1.5 * np.sin(2 * math.pi * t)
        strong = 4.0 * np.sin(2 * math.pi * t)
        dw, _ = newmark_displacement(weak, dt, yield_accel=1.0)
        ds, _ = newmark_displacement(strong, dt, yield_accel=1.0)
        assert ds > dw

    def test_yield_acceleration(self) -> None:
        ay = yield_acceleration(1.3, 20.0)
        assert ay == pytest.approx(0.3 * 9.81 * math.sin(math.radians(20)),
                                   rel=1e-9)
