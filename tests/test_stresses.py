"""Unit tests for the stresses submodule (geostatic + induced).

Reference values are hand-calculated or taken from standard tables
(Newmark influence chart, Boussinesq/Westergaard point-load factors).
"""

import math

import numpy as np
import pytest

from pygeotech.stresses import (
    SoilLayer,
    SoilProfile,
    boussinesq_circle_center,
    boussinesq_point,
    boussinesq_rectangle,
    induced_stress_area,
    influence_factor_rectangle,
    westergaard_point,
)


class TestGeostatic:
    def _profile(self) -> SoilProfile:
        return SoilProfile(
            [
                SoilLayer(2.0, gamma=16.0, name="sand (moist)"),
                SoilLayer(4.0, gamma=18.0, gamma_sat=19.0, name="sand (sat)"),
                SoilLayer(6.0, gamma=17.0, gamma_sat=18.0, name="clay"),
            ],
            water_table_depth=2.0,
        )

    def test_hand_calculation_at_6m(self) -> None:
        p = self._profile()
        # sigma_v = 16*2 + 19*4 = 108 ; u = 9.81*4 = 39.24 ; sigma' = 68.76
        assert p.total_stress(6.0) == pytest.approx(108.0, abs=1e-6)
        assert p.pore_pressure(6.0) == pytest.approx(39.24, abs=1e-6)
        assert p.effective_stress(6.0) == pytest.approx(68.76, abs=1e-6)

    def test_at_water_table(self) -> None:
        p = self._profile()
        assert p.total_stress(2.0) == pytest.approx(32.0, abs=1e-6)
        assert p.pore_pressure(2.0) == pytest.approx(0.0, abs=1e-6)

    def test_dry_profile_has_zero_pore_pressure(self) -> None:
        p = SoilProfile([SoilLayer(5.0, gamma=18.0)])
        assert p.pore_pressure(5.0) == 0.0
        assert p.effective_stress(5.0) == pytest.approx(90.0)

    def test_surcharge_shifts_total_and_effective(self) -> None:
        p = SoilProfile([SoilLayer(3.0, gamma=18.0)], surcharge=25.0)
        assert p.total_stress(3.0) == pytest.approx(25.0 + 54.0)
        assert p.effective_stress(3.0) == pytest.approx(79.0)

    def test_capillary_zone_gives_suction_and_saturation(self) -> None:
        # WT at 3 m, 1 m capillary fringe -> saturated from 2 m down.
        p = SoilProfile(
            [SoilLayer(6.0, gamma=17.0, gamma_sat=20.0)],
            water_table_depth=3.0, capillary_rise=1.0,
        )
        # At z = 2.5 m (in fringe): u negative, unit weight saturated.
        assert p.pore_pressure(2.5) == pytest.approx(9.81 * (2.5 - 3.0))
        # sigma_v(2.5) = 17*2 (dry) + 20*0.5 (sat fringe) = 44
        assert p.total_stress(2.5) == pytest.approx(44.0, abs=1e-6)

    def test_profile_arrays_monotonic_depth(self) -> None:
        depth, sigma, u, eff = self._profile().profile(dz=0.25)
        assert np.all(np.diff(depth) >= 0)
        assert len(depth) == len(sigma) == len(u) == len(eff)
        assert np.allclose(eff, sigma - u)

    def test_negative_depth_raises(self) -> None:
        with pytest.raises(ValueError):
            self._profile().total_stress(-1.0)


class TestBoussinesqPoint:
    def test_under_load(self) -> None:
        # Delta sigma = 3Q / (2 pi z^2) at r = 0.
        val = boussinesq_point(100.0, 0.0, 2.0)
        assert float(val) == pytest.approx(3 * 100 / (2 * math.pi * 4), rel=1e-9)

    def test_decays_with_radius(self) -> None:
        near = boussinesq_point(100.0, 0.0, 2.0)
        far = boussinesq_point(100.0, 3.0, 2.0)
        assert float(near) > float(far)

    def test_vectorised(self) -> None:
        r = np.array([0.0, 1.0, 2.0])
        out = boussinesq_point(100.0, r, 2.0)
        assert out.shape == r.shape
        assert np.all(np.diff(out) < 0)


class TestWestergaardPoint:
    def test_center_factor_nu0(self) -> None:
        # At r = 0, nu = 0 the influence factor is 1/pi.
        val = westergaard_point(100.0, 0.0, 2.0, nu=0.0)
        assert float(val) == pytest.approx(100.0 / 4.0 / math.pi, rel=1e-9)

    def test_less_than_boussinesq(self) -> None:
        w = float(westergaard_point(100.0, 0.0, 2.0, nu=0.0))
        b = float(boussinesq_point(100.0, 0.0, 2.0))
        assert w < b

    def test_invalid_nu_raises(self) -> None:
        with pytest.raises(ValueError):
            westergaard_point(100.0, 0.0, 2.0, nu=0.5)


class TestRectangular:
    def test_newmark_reference_factor(self) -> None:
        # Classic tabulated value I(1, 1) = 0.1752.
        assert float(influence_factor_rectangle(1.0, 1.0)) == pytest.approx(
            0.1752, abs=5e-4
        )

    def test_symmetry_in_m_n(self) -> None:
        assert float(influence_factor_rectangle(0.6, 1.3)) == pytest.approx(
            float(influence_factor_rectangle(1.3, 0.6))
        )

    def test_center_equals_four_corners(self) -> None:
        q, b, l, z = 100.0, 2.0, 3.0, 1.5
        center = float(boussinesq_rectangle(q, b, l, z))
        corner_i = influence_factor_rectangle(b / 2 / z, l / 2 / z)
        assert center == pytest.approx(4.0 * q * float(corner_i), rel=1e-9)

    def test_corner_matches_single_influence(self) -> None:
        q, b, l, z = 100.0, 2.0, 3.0, 1.5
        corner = float(boussinesq_rectangle(q, b, l, z, x=0.0, y=0.0))
        expected = q * float(influence_factor_rectangle(b / z, l / z))
        assert corner == pytest.approx(expected, rel=1e-9)

    def test_numeric_matches_analytic_center(self) -> None:
        q, b, l, z = 120.0, 3.0, 4.0, 2.0
        analytic = float(boussinesq_rectangle(q, b, l, z))
        numeric = induced_stress_area(q, b, l, z, method="boussinesq",
                                      n_cells=120)
        assert numeric == pytest.approx(analytic, rel=2e-3)


class TestCircle:
    def test_center_formula(self) -> None:
        q, a, z = 100.0, 1.5, 2.0
        expected = q * (1 - (1 + (a / z) ** 2) ** -1.5)
        assert float(boussinesq_circle_center(q, a, z)) == pytest.approx(
            expected, rel=1e-9
        )

    def test_shallow_limit_approaches_q(self) -> None:
        # Very close to the surface the stress approaches the pressure.
        assert float(boussinesq_circle_center(100.0, 1.5, 0.01)) == \
            pytest.approx(100.0, abs=1e-2)
