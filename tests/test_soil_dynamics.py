"""Unit tests for the soil_dynamics submodule."""

import math

import numpy as np
import pytest

from pygeotech.soil_dynamics import (
    crr_from_cpt,
    crr_from_spt,
    cyclic_stress_ratio,
    liquefaction_factor_of_safety,
    magnitude_scaling_factor,
    max_shear_modulus,
    peak_amplification,
    site_natural_frequency,
    site_natural_period,
    stress_reduction_coefficient,
    transfer_function_amplitude,
)


class TestLiquefaction:
    def test_msf_unity_at_m75(self) -> None:
        assert magnitude_scaling_factor(7.5) == pytest.approx(1.0, abs=1e-2)

    def test_msf_decreases_with_magnitude(self) -> None:
        assert magnitude_scaling_factor(8.0) < magnitude_scaling_factor(6.0)

    def test_crr_spt_reference_values(self) -> None:
        # Idriss-Boulanger clean-sand CRR at (N1)60cs = 10 ~ 0.118.
        assert crr_from_spt(10.0) == pytest.approx(0.118, abs=0.01)
        assert crr_from_spt(25.0) > crr_from_spt(10.0)

    def test_crr_cpt_increases(self) -> None:
        assert crr_from_cpt(150.0) > crr_from_cpt(80.0)

    def test_rd_decreases_with_depth(self) -> None:
        assert (stress_reduction_coefficient(15.0, 7.5)
                < stress_reduction_coefficient(3.0, 7.5))

    def test_csr_scales_with_amax(self) -> None:
        c1 = cyclic_stress_ratio(0.2 * 9.81, 100, 60, 6, 7.5)
        c2 = cyclic_stress_ratio(0.4 * 9.81, 100, 60, 6, 7.5)
        assert c2 == pytest.approx(2.0 * c1, rel=1e-9)

    def test_loose_sand_liquefies_dense_does_not(self) -> None:
        common = dict(fines_content=5.0, a_max=0.35 * 9.81, magnitude=7.5,
                      sigma_v0=115.0, sigma_v0_eff=70.0, depth=6.0)
        loose = liquefaction_factor_of_safety(n1_60=8.0, **common)
        dense = liquefaction_factor_of_safety(n1_60=35.0, **common)
        assert loose.liquefiable
        assert not dense.liquefiable
        assert dense.factor_of_safety > loose.factor_of_safety

    def test_fines_increase_resistance(self) -> None:
        common = dict(a_max=0.3 * 9.81, magnitude=7.5, sigma_v0=110.0,
                      sigma_v0_eff=70.0, depth=6.0)
        clean = liquefaction_factor_of_safety(n1_60=12.0, fines_content=2.0,
                                              **common)
        silty = liquefaction_factor_of_safety(n1_60=12.0, fines_content=25.0,
                                              **common)
        assert silty.factor_of_safety > clean.factor_of_safety


class TestSiteResponse:
    def test_natural_period(self) -> None:
        assert site_natural_period(30.0, 300.0) == pytest.approx(0.4)

    def test_natural_frequency_matches_period(self) -> None:
        f = site_natural_frequency(30.0, 300.0)
        assert f == pytest.approx(1.0 / site_natural_period(30.0, 300.0))

    def test_transfer_peaks_at_natural_frequency(self) -> None:
        h, vs, xi = 30.0, 300.0, 0.05
        f0 = site_natural_frequency(h, vs)
        amp_res = transfer_function_amplitude(f0, h, vs, xi)
        amp_off = transfer_function_amplitude(f0 * 1.5, h, vs, xi)
        assert amp_res > amp_off
        # Peak amplitude ~ 1/(xi*pi/2).
        assert float(amp_res) == pytest.approx(peak_amplification(xi),
                                               rel=0.1)

    def test_gmax(self) -> None:
        # rho = 2.0 Mg/m3, Vs = 250 m/s -> Gmax = 125000 kPa.
        assert max_shear_modulus(2.0, 250.0) == pytest.approx(125000.0)
