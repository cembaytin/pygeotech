"""Unit tests for the standards (design-code) adapters."""

import math

import pytest

from pygeotech.core import DesignStandard, PyGeotechError
from pygeotech.standards import (
    design_action,
    design_bearing_resistance,
    design_shear_strength,
    factor_set_for,
    factored_resistance,
    strength_i_load,
)


class TestEurocode7:
    def test_da2_factors_resistance(self) -> None:
        fs = factor_set_for(DesignStandard.EUROCODE7_DA2)
        # DA2: material factors 1.0, resistance 1.4.
        phi_d, c_d, cu_d = design_shear_strength(fs, friction_angle=30.0,
                                                 cohesion=10.0)
        assert phi_d == pytest.approx(30.0)     # M1 -> unchanged
        assert c_d == pytest.approx(10.0)
        assert design_bearing_resistance(fs, 1000.0) == pytest.approx(
            1000.0 / 1.4)

    def test_da3_factors_materials(self) -> None:
        fs = factor_set_for(DesignStandard.EUROCODE7_DA3)
        phi_d, c_d, _ = design_shear_strength(fs, friction_angle=30.0,
                                              cohesion=20.0)
        # tan(phi_d) = tan(30)/1.25.
        assert phi_d == pytest.approx(
            math.degrees(math.atan(math.tan(math.radians(30.0)) / 1.25)))
        assert c_d == pytest.approx(20.0 / 1.25)
        # R3 -> resistance unfactored.
        assert design_bearing_resistance(fs, 500.0) == pytest.approx(500.0)

    def test_design_action(self) -> None:
        fs = factor_set_for(DesignStandard.EUROCODE7_DA2)
        # A1: 1.35 G + 1.5 Q.
        assert design_action(fs, 100.0, 40.0) == pytest.approx(
            1.35 * 100.0 + 1.5 * 40.0)

    def test_design_lowers_strength(self) -> None:
        fs = factor_set_for(DesignStandard.EUROCODE7_DA3)
        phi_d, _, _ = design_shear_strength(fs, friction_angle=35.0)
        assert phi_d < 35.0

    def test_non_ec7_standard_raises(self) -> None:
        with pytest.raises(PyGeotechError):
            factor_set_for(DesignStandard.AASHTO_LRFD)


class TestAASHTO:
    def test_factored_resistance(self) -> None:
        assert factored_resistance(1000.0, 0.45) == pytest.approx(450.0)

    def test_resistance_factor_bounds(self) -> None:
        with pytest.raises(ValueError):
            factored_resistance(1000.0, 1.5)

    def test_strength_i_maximum(self) -> None:
        # 1.25 DC + 1.5 DW + 1.75 LL.
        assert strength_i_load(200.0, 50.0, 80.0) == pytest.approx(
            1.25 * 200 + 1.5 * 50 + 1.75 * 80)

    def test_strength_i_minimum_favourable(self) -> None:
        assert strength_i_load(200.0, 50.0, minimum=True) == pytest.approx(
            0.90 * 200 + 0.65 * 50)

    def test_lrfd_design_check(self) -> None:
        # Factored resistance must exceed factored load for an adequate design.
        r_factored = factored_resistance(2000.0, 0.45)
        q_factored = strength_i_load(dead_load=400.0, live_load=200.0)
        assert r_factored > q_factored
