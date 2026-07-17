"""Unit tests for the consolidation submodule."""

import pytest

from pygeotech.consolidation import (
    Consolidation1D,
    average_degree_of_consolidation,
    primary_consolidation_settlement,
    secondary_compression_settlement,
    time_factor_from_degree,
)


class TestTerzaghiSeries:
    def test_standard_time_factors(self) -> None:
        # Tabulated: U = 50% -> Tv ~ 0.197 ; U = 90% -> Tv ~ 0.848.
        assert average_degree_of_consolidation(0.197) == pytest.approx(0.5,
                                                                       abs=2e-3)
        assert average_degree_of_consolidation(0.848) == pytest.approx(0.9,
                                                                       abs=2e-3)

    def test_time_factor_from_degree_matches_series(self) -> None:
        for u in (0.3, 0.5, 0.6, 0.75, 0.9):
            tv = time_factor_from_degree(u)
            assert average_degree_of_consolidation(tv) == pytest.approx(
                u, abs=1e-2)

    def test_zero_and_monotonic(self) -> None:
        assert average_degree_of_consolidation(0.0) == 0.0
        assert (average_degree_of_consolidation(0.1)
                < average_degree_of_consolidation(0.5))


class TestConsolidation1D:
    def test_drainage_path_double(self) -> None:
        clay = Consolidation1D(cv=2.0, layer_thickness=4.0, drainage="double")
        assert clay.drainage_path == 2.0
        assert clay.time_factor(1.0) == pytest.approx(0.5)

    def test_drainage_path_single(self) -> None:
        clay = Consolidation1D(cv=2.0, layer_thickness=4.0, drainage="single")
        assert clay.drainage_path == 4.0

    def test_time_for_degree_roundtrip(self) -> None:
        clay = Consolidation1D(cv=1.5, layer_thickness=6.0)
        t = clay.time_for_degree(0.90)
        assert clay.average_degree(t) == pytest.approx(0.90, abs=1e-2)

    def test_isochrone_endpoints_drained(self) -> None:
        clay = Consolidation1D(cv=1.0, layer_thickness=2.0, drainage="double")
        z, ratio = clay.isochrone(0.1)
        # Fully drained boundaries: ue/u0 -> 0 at both faces.
        assert ratio[0] == pytest.approx(0.0, abs=1e-6)
        assert ratio[-1] == pytest.approx(0.0, abs=1e-6)
        assert ratio.max() > 0.5


class TestSettlement:
    def test_normally_consolidated(self) -> None:
        s = primary_consolidation_settlement(
            thickness=3.0, void_ratio=0.8, sigma0=100.0,
            delta_sigma=100.0, cc=0.25)
        # 0.25*3/1.8*log10(2) = 0.1254 m
        assert s == pytest.approx(0.1254, abs=1e-3)

    def test_overconsolidated_below_preconsolidation(self) -> None:
        s = primary_consolidation_settlement(
            thickness=3.0, void_ratio=0.8, sigma0=100.0, delta_sigma=50.0,
            cc=0.25, cr=0.05, sigma_p=200.0)
        # stays below sigma_p -> uses Cr only
        assert s == pytest.approx(
            0.05 * 3 / 1.8 * __import__("math").log10(150 / 100), abs=1e-4)

    def test_overconsolidated_crossing_preconsolidation(self) -> None:
        import math
        s = primary_consolidation_settlement(
            thickness=3.0, void_ratio=0.8, sigma0=100.0, delta_sigma=200.0,
            cc=0.25, cr=0.05, sigma_p=150.0)
        expected = 3 / 1.8 * (0.05 * math.log10(150 / 100)
                              + 0.25 * math.log10(300 / 150))
        assert s == pytest.approx(expected, abs=1e-4)

    def test_oc_requires_cr(self) -> None:
        with pytest.raises(ValueError):
            primary_consolidation_settlement(
                3.0, 0.8, 100.0, 200.0, cc=0.25, sigma_p=150.0)

    def test_secondary_compression(self) -> None:
        import math
        s = secondary_compression_settlement(
            thickness=3.0, void_ratio_p=0.7, c_alpha=0.02, t1=1.0, t2=10.0)
        assert s == pytest.approx(0.02 * 3 / 1.7 * math.log10(10.0), abs=1e-5)
