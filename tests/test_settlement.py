"""Unit tests for the settlement submodule."""

import numpy as np
import pytest

from pygeotech.settlement import (
    burland_burbidge_settlement,
    elastic_settlement,
    schmertmann_settlement,
    steinbrenner_influence,
)


class TestElastic:
    def test_steinbrenner_halfspace_square(self) -> None:
        # Classic corner value for a square on a half-space: Is -> 0.561.
        assert steinbrenner_influence(1.0, 1e6, 0.5) == pytest.approx(
            0.561, abs=2e-3)

    def test_is_increases_with_layer_depth(self) -> None:
        shallow = steinbrenner_influence(1.0, 1.0, 0.3)
        deep = steinbrenner_influence(1.0, 5.0, 0.3)
        assert deep > shallow

    def test_settlement_scales_with_pressure(self) -> None:
        s1 = elastic_settlement(100.0, 2.0, 2.0, youngs_modulus=20000.0)
        s2 = elastic_settlement(200.0, 2.0, 2.0, youngs_modulus=20000.0)
        assert s2 == pytest.approx(2.0 * s1)

    def test_center_greater_than_corner(self) -> None:
        common = dict(pressure=150.0, width=2.0, length=3.0,
                      youngs_modulus=15000.0, poisson_ratio=0.3)
        center = elastic_settlement(position="center", **common)
        corner = elastic_settlement(position="corner", **common)
        assert center > corner

    def test_rigid_less_than_flexible(self) -> None:
        common = dict(pressure=150.0, width=2.0, youngs_modulus=15000.0)
        flex = elastic_settlement(position="center", rigid=False, **common)
        rigid = elastic_settlement(position="center", rigid=True, **common)
        assert rigid == pytest.approx(0.93 * flex)

    def test_softer_soil_settles_more(self) -> None:
        soft = elastic_settlement(150.0, 2.0, youngs_modulus=5000.0)
        stiff = elastic_settlement(150.0, 2.0, youngs_modulus=50000.0)
        assert soft > stiff


class TestSchmertmann:
    def _profile(self):
        depth = np.linspace(0.0, 8.0, 40)
        qc = np.full_like(depth, 8000.0)     # kPa, uniform sand
        return depth, qc

    def test_positive_and_load_scaling(self) -> None:
        depth, qc = self._profile()
        s1 = schmertmann_settlement(100.0, 50.0, 2.0, depth, qc)
        s2 = schmertmann_settlement(200.0, 50.0, 2.0, depth, qc)
        assert s1 > 0.0
        assert s2 > s1

    def test_stiffer_soil_settles_less(self) -> None:
        depth, qc = self._profile()
        soft = schmertmann_settlement(150.0, 50.0, 2.0, depth, qc)
        stiff = schmertmann_settlement(150.0, 50.0, 2.0, depth, 3.0 * qc)
        assert stiff < soft

    def test_creep_increases_settlement(self) -> None:
        depth, qc = self._profile()
        short = schmertmann_settlement(150.0, 50.0, 2.0, depth, qc,
                                       time_years=0.1)
        long = schmertmann_settlement(150.0, 50.0, 2.0, depth, qc,
                                      time_years=10.0)
        assert long > short

    def test_plane_strain_deeper_influence(self) -> None:
        depth, qc = self._profile()
        square = schmertmann_settlement(150.0, 50.0, 2.0, depth, qc)
        strip = schmertmann_settlement(150.0, 50.0, 2.0, depth, qc,
                                       length=40.0)
        assert strip > square      # strip mobilises a deeper soil column


class TestBurlandBurbidge:
    def test_reference_value(self) -> None:
        # B=2, N60=20, q'=150, square -> ~6.3 mm.
        s = burland_burbidge_settlement(150.0, 2.0, 20.0)
        assert s == pytest.approx(6.29, abs=0.2)

    def test_denser_sand_settles_less(self) -> None:
        loose = burland_burbidge_settlement(150.0, 2.0, 10.0)
        dense = burland_burbidge_settlement(150.0, 2.0, 40.0)
        assert dense < loose

    def test_overconsolidated_settles_less(self) -> None:
        nc = burland_burbidge_settlement(150.0, 2.0, 20.0)
        oc = burland_burbidge_settlement(150.0, 2.0, 20.0,
                                         preconsolidation=200.0)
        assert oc < nc      # q' < sp -> Ic/3 branch
