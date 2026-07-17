"""Tests for the API RP2A sand p-y curves on a laterally loaded pile."""

import math

import pytest

from pygeotech.foundations import (
    api_sand_modulus,
    api_sand_py,
    matlock_clay_py,
    solve_laterally_loaded_pile,
)


class TestApiSandCurve:
    def test_zero_at_surface(self) -> None:
        curve = api_sand_py(0.6, friction_angle=32.0, gamma_eff=10.0)
        assert curve(0.0, 0.01) == 0.0

    def test_resistance_increases_with_depth(self) -> None:
        curve = api_sand_py(0.6, friction_angle=32.0, gamma_eff=10.0)
        assert curve(5.0, 0.02) > curve(1.0, 0.02)

    def test_resistance_increases_with_deflection(self) -> None:
        curve = api_sand_py(0.6, friction_angle=32.0, gamma_eff=10.0)
        assert curve(3.0, 0.05) > curve(3.0, 0.005)

    def test_denser_sand_stiffer(self) -> None:
        loose = api_sand_py(0.6, friction_angle=30.0, gamma_eff=10.0)
        dense = api_sand_py(0.6, friction_angle=38.0, gamma_eff=10.0)
        assert dense(3.0, 0.02) > loose(3.0, 0.02)

    def test_modulus_interpolation(self) -> None:
        # Chart anchors: 30 deg -> 11 MN/m3, 35 deg -> 22 MN/m3.
        assert api_sand_modulus(30.0) == pytest.approx(11.0e3)
        assert api_sand_modulus(35.0) == pytest.approx(22.0e3)
        assert (api_sand_modulus(38.0) > api_sand_modulus(32.0))


class TestPileInSand:
    def test_denser_sand_less_head_deflection(self) -> None:
        ei = 8.0e4
        loose = solve_laterally_loaded_pile(
            15.0, ei, api_sand_py(0.6, 30.0, 10.0), lateral_load=150.0)
        dense = solve_laterally_loaded_pile(
            15.0, ei, api_sand_py(0.6, 38.0, 10.0), lateral_load=150.0)
        assert 0.0 < dense.head_deflection < loose.head_deflection

    def test_more_load_more_deflection(self) -> None:
        ei = 8.0e4
        curve = api_sand_py(0.6, 34.0, 10.0)
        light = solve_laterally_loaded_pile(15.0, ei, curve, lateral_load=50.0)
        heavy = solve_laterally_loaded_pile(15.0, ei, curve, lateral_load=200.0)
        assert heavy.head_deflection > light.head_deflection

    def test_sand_vs_clay_runs(self) -> None:
        ei = 8.0e4
        sand = solve_laterally_loaded_pile(
            15.0, ei, api_sand_py(0.6, 34.0, 10.0), lateral_load=120.0)
        clay = solve_laterally_loaded_pile(
            15.0, ei, matlock_clay_py(0.6, su=40.0, gamma_eff=9.0),
            lateral_load=120.0)
        assert sand.head_deflection > 0.0 and clay.head_deflection > 0.0
