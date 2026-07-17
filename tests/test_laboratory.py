"""Unit tests for laboratory-test interpretation."""

import numpy as np
import pytest

from pygeotech.characterization import (
    compression_index,
    cv_casagrande,
    cv_taylor,
    preconsolidation_bilinear,
    proctor_optimum,
    zero_air_voids_curve,
)


class TestConsolidationTest:
    def test_cv_casagrande(self) -> None:
        # cv = 0.197 H^2 / t50.
        assert cv_casagrande(t50=10.0, drainage_path=0.01) == pytest.approx(
            0.197 * 0.01 ** 2 / 10.0)

    def test_cv_taylor(self) -> None:
        assert cv_taylor(t90=25.0, drainage_path=0.01) == pytest.approx(
            0.848 * 0.01 ** 2 / 25.0)

    def test_cv_positive_required(self) -> None:
        with pytest.raises(ValueError):
            cv_casagrande(-1.0, 0.01)

    def test_compression_index_recovers_slope(self) -> None:
        # Build a virgin line e = e0 - Cc*log10(sigma).
        sigma = np.array([100.0, 200.0, 400.0, 800.0])
        cc_true = 0.30
        e = 1.2 - cc_true * np.log10(sigma)
        assert compression_index(sigma, e) == pytest.approx(cc_true, abs=1e-9)

    def test_preconsolidation_bilinear(self) -> None:
        # Recompression Cr=0.03 up to sp=200, virgin Cc=0.30 beyond.
        sigma = np.array([25, 50, 100, 200, 400, 800, 1600], dtype=float)
        cr, cc, sp = 0.03, 0.30, 200.0
        e = np.empty_like(sigma)
        e0 = 1.0
        for i, s in enumerate(sigma):
            if s <= sp:
                e[i] = e0 - cr * np.log10(s / 25.0)
            else:
                e_sp = e0 - cr * np.log10(sp / 25.0)
                e[i] = e_sp - cc * np.log10(s / sp)
        est = preconsolidation_bilinear(sigma, e, n_recompression=4,
                                       n_virgin=4)
        assert est == pytest.approx(200.0, rel=0.05)


class TestCompaction:
    def test_proctor_optimum_peak(self) -> None:
        # Symmetric parabola peaking at w=0.15, gd_max=18.5.
        w = np.array([0.09, 0.12, 0.15, 0.18, 0.21])
        gd = 18.5 - 200.0 * (w - 0.15) ** 2
        w_opt, gd_max = proctor_optimum(w, gd)
        assert w_opt == pytest.approx(0.15, abs=1e-3)
        assert gd_max == pytest.approx(18.5, abs=1e-2)

    def test_proctor_needs_peak(self) -> None:
        w = np.array([0.1, 0.15, 0.2])
        gd = np.array([16.0, 17.0, 18.0])   # monotonic, no peak
        with pytest.raises(ValueError):
            proctor_optimum(w, gd)

    def test_zero_air_voids_decreasing(self) -> None:
        w = np.array([0.10, 0.15, 0.20, 0.25])
        zav = zero_air_voids_curve(w, specific_gravity=2.70)
        assert np.all(np.diff(zav) < 0)
        # At w=0.15, Gs=2.7: gd_zav = 2.7*9.81/(1+0.15*2.7).
        assert zav[1] == pytest.approx(2.70 * 9.81 / (1 + 0.15 * 2.70),
                                      rel=1e-9)
