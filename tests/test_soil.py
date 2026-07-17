"""Unit tests for the phase-relations solver (values cross-checked
against standard textbook solutions, e.g. Das, *Principles of
Geotechnical Engineering*)."""

import pytest

from pygeotech.phase_relations import InconsistentInputError, Soil


class TestForwardDerivations:
    def test_from_w_gs_e(self) -> None:
        s = Soil(w=0.20, gs=2.70, e=0.80)
        assert s.sr == pytest.approx(0.675, rel=1e-3)
        assert s.n == pytest.approx(0.4444, rel=1e-3)
        assert s.gamma_d == pytest.approx(14.715, rel=1e-3)
        assert s.gamma == pytest.approx(17.658, rel=1e-3)
        assert s.gamma_sat == pytest.approx(19.0753, rel=1e-3)
        assert s.gamma_sub == pytest.approx(9.2653, rel=1e-3)
        assert s.air_content == pytest.approx(0.4444 * 0.325, rel=1e-3)

    def test_from_gamma_w_gs(self) -> None:
        s = Soil(gamma=18.0, w=0.15, gs=2.65)
        assert s.gamma_d == pytest.approx(15.652, rel=1e-3)
        assert s.e == pytest.approx(0.6611, rel=1e-3)
        assert s.sr == pytest.approx(0.6013, rel=1e-3)

    def test_saturated_soil(self) -> None:
        s = Soil(sr=1.0, w=0.25, gs=2.70)
        assert s.e == pytest.approx(0.675, rel=1e-3)
        # For Sr = 1 the bulk and saturated unit weights coincide.
        assert s.gamma == pytest.approx(s.gamma_sat, rel=1e-6)

    def test_porosity_route(self) -> None:
        s = Soil(n=0.4444, gs=2.70, sr=0.675)
        assert s.e == pytest.approx(0.80, rel=1e-3)
        assert s.w == pytest.approx(0.20, rel=1e-3)

    def test_gamma_sat_inverse(self) -> None:
        # e recovered from gamma_sat: e = (Gs*gw - gsat)/(gsat - gw).
        s = Soil(gamma_sat=19.0753, gs=2.70)
        assert s.e == pytest.approx(0.80, rel=1e-3)


class TestValidationAndConsistency:
    def test_redundant_consistent_inputs_pass(self) -> None:
        s = Soil(w=0.20, gs=2.70, e=0.80, gamma_d=14.72)
        assert s.gamma == pytest.approx(17.658, rel=1e-2)

    def test_inconsistent_e_n(self) -> None:
        with pytest.raises(InconsistentInputError):
            Soil(e=0.80, n=0.60)

    def test_oversaturated_state_rejected(self) -> None:
        # w*Gs/e = 0.5*2.7/0.8 = 1.69 > 1 -> physically impossible.
        with pytest.raises(InconsistentInputError):
            Soil(w=0.50, gs=2.70, e=0.80)

    def test_negative_water_content_rejected(self) -> None:
        with pytest.raises(InconsistentInputError):
            Soil(w=-0.1, gs=2.70)

    def test_underdetermined_inputs_stay_none(self) -> None:
        s = Soil(w=0.20)
        assert s.e is None
        assert s.gamma is None
        summary = s.summary()
        assert summary["w"] == pytest.approx(0.20)
        assert summary["sr"] is None


class TestReporting:
    def test_report_and_log(self) -> None:
        s = Soil(w=0.20, gs=2.70, e=0.80)
        assert len(s.derivation_log) >= 5
        text = s.report()
        assert "derived" in text and "input" in text
        assert "Sr = w Gs / e" in text
