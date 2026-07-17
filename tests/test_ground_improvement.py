"""Unit tests for the ground_improvement submodule."""

import math

import pytest

from pygeotech.ground_improvement import (
    area_replacement_ratio,
    combined_degree_of_consolidation,
    drain_influence_diameter,
    hansbo_factor,
    priebe_improvement_factor,
    radial_degree_of_consolidation,
    radial_time_factor,
    settlement_improvement_equilibrium,
)


class TestVerticalDrains:
    def test_influence_diameter_patterns(self) -> None:
        assert drain_influence_diameter(1.5, "triangular") == pytest.approx(1.575)
        assert drain_influence_diameter(1.5, "square") == pytest.approx(1.695)

    def test_hansbo_no_smear(self) -> None:
        # s = 1 -> F(n) = ln(n) - 0.75.
        assert hansbo_factor(20.0) == pytest.approx(math.log(20.0) - 0.75)

    def test_hansbo_smear_increases_factor(self) -> None:
        no_smear = hansbo_factor(20.0, smear_ratio=1.0, kh_ks=1.0)
        smear = hansbo_factor(20.0, smear_ratio=3.0, kh_ks=5.0)
        assert smear > no_smear      # smear slows consolidation

    def test_radial_degree_monotonic(self) -> None:
        # ch = 2 m^2/year, de = 1.6 m; times in years.
        fn = hansbo_factor(20.0)
        u1 = radial_degree_of_consolidation(radial_time_factor(2.0, 0.1, 1.6), fn)
        u2 = radial_degree_of_consolidation(radial_time_factor(2.0, 0.5, 1.6), fn)
        assert 0.0 < u1 < u2 < 1.0

    def test_combined_exceeds_components(self) -> None:
        comb = combined_degree_of_consolidation(0.5, 0.6)
        assert comb == pytest.approx(1 - 0.5 * 0.4)
        assert comb > 0.6

    def test_smear_ratio_validation(self) -> None:
        with pytest.raises(ValueError):
            hansbo_factor(2.0, smear_ratio=3.0)


class TestStoneColumns:
    def test_area_replacement_ratio(self) -> None:
        a_s = area_replacement_ratio(0.8, 2.0, "square")
        assert a_s == pytest.approx((math.pi * 0.8 ** 2 / 4.0) / 4.0)

    def test_equilibrium_improvement(self) -> None:
        beta = settlement_improvement_equilibrium(0.2, 4.0)
        assert beta == pytest.approx(1 + 0.2 * 3.0)

    def test_priebe_reference(self) -> None:
        assert priebe_improvement_factor(0.2, 40.0) == pytest.approx(2.18,
                                                                     abs=0.01)

    def test_priebe_more_area_more_improvement(self) -> None:
        assert (priebe_improvement_factor(0.3, 40.0)
                > priebe_improvement_factor(0.1, 40.0))

    def test_priebe_stronger_column(self) -> None:
        assert (priebe_improvement_factor(0.2, 45.0)
                > priebe_improvement_factor(0.2, 35.0))
