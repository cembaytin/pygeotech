"""Unit tests for pile axial capacity and groups."""

import math

import numpy as np
import pytest

from pygeotech.foundations import (
    alpha_api,
    group_efficiency_converse_labarre,
    pile_capacity_alpha,
    pile_capacity_beta,
)


class TestAlphaMethod:
    def test_uniform_clay_hand_calc(self) -> None:
        # D=0.5, L=15, su=75 constant, alpha=0.5.
        depth = np.linspace(0.0, 15.0, 61)
        su = np.full_like(depth, 75.0)
        res = pile_capacity_alpha(0.5, depth, su, alpha=0.5)
        exp_shaft = 0.5 * 75.0 * (math.pi * 0.5) * 15.0
        exp_base = 9.0 * 75.0 * (math.pi * 0.5 ** 2 / 4.0)
        assert res.shaft == pytest.approx(exp_shaft, rel=1e-3)
        assert res.base == pytest.approx(exp_base, rel=1e-9)
        assert res.ultimate == pytest.approx(exp_shaft + exp_base, rel=1e-3)
        assert res.allowable == pytest.approx(res.ultimate / 3.0)

    def test_api_alpha_capped(self) -> None:
        # Soft clay: psi < 1 gives alpha up to 1 (capped).
        assert alpha_api(20.0, 100.0) <= 1.0
        assert alpha_api(200.0, 50.0) <= 1.0
        # Very soft (su << sigma') -> alpha capped at 1.
        assert alpha_api(5.0, 100.0) == 1.0

    def test_stronger_clay_more_capacity(self) -> None:
        depth = np.linspace(0.0, 12.0, 40)
        weak = pile_capacity_alpha(0.4, depth, np.full_like(depth, 40.0),
                                   alpha=0.6)
        strong = pile_capacity_alpha(0.4, depth, np.full_like(depth, 90.0),
                                     alpha=0.6)
        assert strong.ultimate > weak.ultimate


class TestBetaMethod:
    def test_linear_stress_profile(self) -> None:
        depth = np.linspace(0.0, 10.0, 41)
        sve = 9.0 * depth       # gamma' ~ 9 kN/m3
        res = pile_capacity_beta(0.5, depth, sve, beta=0.3, nq=20.0)
        # Shaft = beta * integral(sve) * perimeter.
        exp_shaft = 0.3 * np.trapz(sve, depth) * (math.pi * 0.5)
        assert res.shaft == pytest.approx(exp_shaft, rel=1e-6)
        assert res.base > 0.0

    def test_beta_from_k_delta(self) -> None:
        depth = np.linspace(0.0, 10.0, 41)
        sve = 9.0 * depth
        res = pile_capacity_beta(0.5, depth, sve, k_earth=0.8, delta=25.0,
                                 nq=20.0)
        beta = 0.8 * math.tan(math.radians(25.0))
        exp_shaft = beta * np.trapz(sve, depth) * (math.pi * 0.5)
        assert res.shaft == pytest.approx(exp_shaft, rel=1e-6)

    def test_base_limit(self) -> None:
        depth = np.linspace(0.0, 10.0, 41)
        sve = 9.0 * depth
        capped = pile_capacity_beta(0.5, depth, sve, beta=0.3, nq=40.0,
                                    limit_unit_base=1000.0)
        area = math.pi * 0.5 ** 2 / 4.0
        assert capped.base == pytest.approx(1000.0 * area)

    def test_missing_beta_inputs_raise(self) -> None:
        depth = np.linspace(0.0, 5.0, 10)
        with pytest.raises(ValueError):
            pile_capacity_beta(0.5, depth, 9.0 * depth)


class TestGroupEfficiency:
    def test_single_pile_unity(self) -> None:
        assert group_efficiency_converse_labarre(1, 1, 1.5, 0.5) == 1.0

    def test_efficiency_between_0_and_1(self) -> None:
        eg = group_efficiency_converse_labarre(3, 3, 1.5, 0.5)
        assert 0.0 < eg < 1.0

    def test_wider_spacing_more_efficient(self) -> None:
        close = group_efficiency_converse_labarre(3, 3, 1.0, 0.5)
        wide = group_efficiency_converse_labarre(3, 3, 3.0, 0.5)
        assert wide > close
