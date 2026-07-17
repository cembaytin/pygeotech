"""Tests for the laterally loaded pile p-y solver (validated vs Hetenyi)."""

import math

import numpy as np
import pytest

from pygeotech.foundations import (
    linear_subgrade_py,
    matlock_clay_py,
    solve_laterally_loaded_pile,
)


class TestLinearHetenyi:
    """Constant subgrade -> Hetenyi's long-beam closed form."""

    def _case(self):
        ei = 1.0e5
        k = 5000.0
        h = 100.0
        beta = (k / (4 * ei)) ** 0.25
        length = 6.0 / beta          # betaL = 6 -> effectively infinite
        return ei, k, h, beta, length

    def test_head_deflection_matches_hetenyi(self) -> None:
        ei, k, h, beta, length = self._case()
        res = solve_laterally_loaded_pile(length, ei, linear_subgrade_py(k),
                                          lateral_load=h, n_elements=300)
        y0 = 2 * h * beta / k
        assert res.head_deflection == pytest.approx(y0, rel=2e-3)

    def test_positive_load_positive_deflection(self) -> None:
        ei, k, h, beta, length = self._case()
        res = solve_laterally_loaded_pile(length, ei, linear_subgrade_py(k),
                                          lateral_load=h, n_elements=200)
        assert res.head_deflection > 0.0

    def test_max_moment_matches_hetenyi(self) -> None:
        ei, k, h, beta, length = self._case()
        res = solve_laterally_loaded_pile(length, ei, linear_subgrade_py(k),
                                          lateral_load=h, n_elements=300)
        # Hetenyi end-loaded semi-infinite beam: Mmax = 0.3224 H/beta.
        assert res.max_moment == pytest.approx(0.3224 * h / beta, rel=2e-2)

    def test_deflection_decays_with_depth(self) -> None:
        ei, k, h, beta, length = self._case()
        res = solve_laterally_loaded_pile(length, ei, linear_subgrade_py(k),
                                          lateral_load=h, n_elements=200)
        assert abs(res.deflection[-1]) < 0.05 * abs(res.head_deflection)


class TestMatlockClay:
    def test_stiffer_clay_less_deflection(self) -> None:
        ei = 8.0e4
        length = 15.0
        soft = solve_laterally_loaded_pile(
            length, ei, matlock_clay_py(0.5, su=25.0, gamma_eff=9.0),
            lateral_load=150.0)
        stiff = solve_laterally_loaded_pile(
            length, ei, matlock_clay_py(0.5, su=100.0, gamma_eff=9.0),
            lateral_load=150.0)
        assert stiff.head_deflection < soft.head_deflection
        assert stiff.head_deflection > 0.0

    def test_more_load_more_deflection(self) -> None:
        ei = 8.0e4
        curve = matlock_clay_py(0.5, su=50.0, gamma_eff=9.0)
        light = solve_laterally_loaded_pile(15.0, ei, curve, lateral_load=50.0)
        heavy = solve_laterally_loaded_pile(15.0, ei, curve, lateral_load=200.0)
        assert heavy.head_deflection > light.head_deflection
        # Nonlinear softening: deflection grows faster than the load.
        assert (heavy.head_deflection / light.head_deflection) > 4.0

    def test_applied_moment_increases_deflection(self) -> None:
        ei = 8.0e4
        curve = matlock_clay_py(0.5, su=50.0, gamma_eff=9.0)
        no_m = solve_laterally_loaded_pile(15.0, ei, curve, lateral_load=100.0)
        with_m = solve_laterally_loaded_pile(15.0, ei, curve,
                                             lateral_load=100.0, moment=150.0)
        assert with_m.head_deflection > no_m.head_deflection
