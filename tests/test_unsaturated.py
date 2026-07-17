"""Unit tests for the unsaturated submodule."""

import math

import numpy as np
import pytest

from pygeotech.unsaturated import (
    fredlund_xing_water_content,
    relative_permeability_vg,
    unsaturated_shear_strength,
    unsaturated_shear_strength_vanapalli,
    van_genuchten_saturation,
    van_genuchten_water_content,
)


class TestSWCC:
    def test_van_genuchten_saturation_limits(self) -> None:
        # Se = 1 at zero suction, -> 0 at very high suction.
        assert van_genuchten_saturation(0.0, alpha=0.1, n=2.0) == pytest.approx(1.0)
        assert van_genuchten_saturation(1e6, alpha=0.1, n=2.0) < 1e-3

    def test_van_genuchten_monotonic(self) -> None:
        psi = np.array([1.0, 10.0, 100.0, 1000.0])
        se = van_genuchten_saturation(psi, alpha=0.05, n=1.8)
        assert np.all(np.diff(se) < 0)

    def test_water_content_bounds(self) -> None:
        theta = van_genuchten_water_content(
            np.array([0.0, 50.0, 1e5]), theta_s=0.45, theta_r=0.05,
            alpha=0.1, n=2.0)
        assert theta[0] == pytest.approx(0.45)
        assert theta[-1] == pytest.approx(0.05, abs=1e-3)

    def test_fredlund_xing_near_saturated(self) -> None:
        theta = fredlund_xing_water_content(0.1, theta_s=0.5, a=50.0, n=2.0,
                                            m=1.0)
        assert theta == pytest.approx(0.5, rel=0.05)

    def test_relative_permeability_limits(self) -> None:
        kr0 = relative_permeability_vg(0.0, alpha=0.1, n=2.0)
        kr_dry = relative_permeability_vg(1e5, alpha=0.1, n=2.0)
        assert kr0 == pytest.approx(1.0, abs=1e-6)
        assert kr_dry < 1e-4


class TestUnsaturatedStrength:
    def test_phi_b_contribution(self) -> None:
        s = unsaturated_shear_strength(100.0, 50.0, cohesion=10.0,
                                       friction_angle=30.0, phi_b=15.0)
        expected = (10.0 + 100.0 * math.tan(math.radians(30.0))
                    + 50.0 * math.tan(math.radians(15.0)))
        assert s == pytest.approx(expected)

    def test_suction_adds_strength(self) -> None:
        dry = unsaturated_shear_strength(100.0, 0.0, 10.0, 30.0, 15.0)
        wet = unsaturated_shear_strength(100.0, 80.0, 10.0, 30.0, 15.0)
        assert wet > dry

    def test_vanapalli_reduces_to_saturated(self) -> None:
        # Se = 0 (fully dry per this coefficient) -> no suction term.
        s0 = unsaturated_shear_strength_vanapalli(100.0, 80.0, 0.0, 10.0, 30.0)
        assert s0 == pytest.approx(10.0 + 100.0 * math.tan(math.radians(30.0)))

    def test_vanapalli_saturated_full_suction(self) -> None:
        # Se = 1 -> tan(phi^b) = tan(phi').
        s = unsaturated_shear_strength_vanapalli(100.0, 50.0, 1.0, 10.0, 30.0)
        tan_phi = math.tan(math.radians(30.0))
        assert s == pytest.approx(10.0 + 100.0 * tan_phi + 50.0 * tan_phi)
