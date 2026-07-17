"""Unit tests for the characterization submodule (SPT + CPT)."""

import math

import numpy as np
import pytest

from pygeotech.characterization import (
    CPTLog,
    SPTLog,
    corrected_n1_60,
    corrected_qt,
    correct_n60,
    friction_angle_cpt,
    friction_angle_spt,
    overburden_factor,
    relative_density_spt,
    soil_behaviour_type,
    soil_behaviour_type_index,
    undrained_strength_cpt,
    undrained_strength_spt,
)
from pygeotech.stresses import SoilLayer, SoilProfile


class TestSPT:
    def test_n60_energy_correction(self) -> None:
        # 45% donut hammer -> N60 = N * 45/60.
        assert correct_n60(20.0, energy_ratio=45.0) == pytest.approx(15.0)

    def test_overburden_factor_reference(self) -> None:
        # At sigma' = pa, CN = 1.
        assert overburden_factor(101.325) == pytest.approx(1.0, abs=1e-9)
        # CN is capped.
        assert overburden_factor(10.0, cap=1.7) == 1.7

    def test_n1_60(self) -> None:
        n1 = corrected_n1_60(20.0, sigma_v_eff=101.325)
        assert n1 == pytest.approx(20.0, abs=1e-6)

    def test_hatanaka_friction_angle(self) -> None:
        # (N1)60 = 20 -> sqrt(400) + 20 = 40 deg.
        assert friction_angle_spt(n1_60=20.0) == pytest.approx(40.0)

    def test_wolff_friction_angle(self) -> None:
        val = friction_angle_spt(n60=30.0, method="wolff")
        assert val == pytest.approx(27.1 + 0.3 * 30 - 5.4e-4 * 900, abs=1e-6)

    def test_relative_density_cap(self) -> None:
        assert relative_density_spt(60.0) == pytest.approx(1.0)
        assert relative_density_spt(15.0) == pytest.approx(0.5, abs=1e-9)

    def test_undrained_strength(self) -> None:
        assert undrained_strength_spt(10.0, factor=4.5) == pytest.approx(45.0)

    def test_sptlog_with_profile_integration(self) -> None:
        profile = SoilProfile(
            [SoilLayer(20.0, gamma=18.0, gamma_sat=20.0)],
            water_table_depth=2.0)
        log = SPTLog(depth=[3.0, 6.0, 9.0], n_field=[10, 18, 25],
                     profile=profile, energy_ratio=60.0)
        n60 = log.n60()
        n1 = log.n1_60()
        assert np.allclose(n60, [10, 18, 25])       # ER = 60 -> unchanged
        # (N1)60 must decrease with depth (CN falls as sigma' grows).
        cn = n1 / n60
        assert cn[0] > cn[1] > cn[2]
        assert np.all(log.friction_angle() > 20.0)

    def test_sptlog_requires_stress_source(self) -> None:
        with pytest.raises(ValueError):
            SPTLog(depth=[1.0], n_field=[5.0])


class TestCPT:
    def test_corrected_qt(self) -> None:
        # qt = qc + (1-a) u2.
        assert corrected_qt(5000.0, u2=300.0, area_ratio=0.8) == \
            pytest.approx(5000.0 + 0.2 * 300.0)

    def test_soil_behaviour_type_bands(self) -> None:
        assert soil_behaviour_type(1.2)[0] == 7
        assert soil_behaviour_type(1.9)[0] == 6
        assert soil_behaviour_type(2.3)[0] == 5
        assert soil_behaviour_type(2.8)[0] == 4
        assert soil_behaviour_type(3.2)[0] == 3
        assert soil_behaviour_type(4.0)[0] == 2

    def test_ic_sand_vs_clay(self) -> None:
        # Stiff sand: high qt, low fs -> low Ic (sand).
        ic_sand, _, _ = soil_behaviour_type_index(
            qt=12000.0, fs=60.0, sigma_v0=100.0, sigma_v0_eff=60.0)
        # Soft clay: low qt, higher friction ratio -> high Ic (clay).
        ic_clay, _, _ = soil_behaviour_type_index(
            qt=800.0, fs=40.0, sigma_v0=100.0, sigma_v0_eff=60.0)
        assert ic_sand < 2.6 < ic_clay

    def test_ic_iteration_converges(self) -> None:
        ic, qtn, n = soil_behaviour_type_index(
            qt=5000.0, fs=50.0, sigma_v0=100.0, sigma_v0_eff=60.0)
        assert 0.0 < n <= 1.0
        assert 1.0 < ic < 3.0
        assert qtn > 0.0

    def test_qt_below_overburden_raises(self) -> None:
        with pytest.raises(ValueError):
            soil_behaviour_type_index(qt=50.0, fs=5.0, sigma_v0=100.0,
                                      sigma_v0_eff=60.0)

    def test_undrained_strength_cpt(self) -> None:
        su = undrained_strength_cpt(qt=1000.0, sigma_v0=100.0, nkt=15.0)
        assert su == pytest.approx((1000.0 - 100.0) / 15.0)

    def test_friction_angle_cpt_range(self) -> None:
        phi = friction_angle_cpt(qt=10000.0, sigma_v0_eff=100.0)
        assert 30.0 < phi < 45.0

    def test_cptlog_process_shapes_and_estimates(self) -> None:
        depth = np.linspace(1.0, 10.0, 10)
        qc = np.full(10, 8000.0)          # kPa, sand-like
        fs = np.full(10, 60.0)
        log = CPTLog(depth, qc, fs, water_table_depth=2.0)
        res = log.process()
        assert res.ic.shape == depth.shape
        assert np.all(res.unit_weight > 10.0)     # sensible gamma estimate
        assert np.all(np.isfinite(res.ic))
        # Total stress must increase with depth.
        assert np.all(np.diff(res.sigma_v0) > 0)

    def test_cptlog_constant_unit_weight(self) -> None:
        depth = np.array([2.0, 4.0, 6.0])
        log = CPTLog(depth, qc=np.full(3, 5000.0), fs=np.full(3, 50.0),
                     unit_weight=19.0, water_table_depth=0.0)
        res = log.process()
        assert np.allclose(res.unit_weight, 19.0)
        # sigma_v0 at 6 m = 19*6 = 114 kPa.
        assert res.sigma_v0[-1] == pytest.approx(114.0, rel=1e-6)
