"""Unit tests for the USCS (ASTM D2487) classification engine."""

import pytest

from pygeotech.phase_relations import classify_uscs


class TestFineGrained:
    def test_lean_clay(self) -> None:
        res = classify_uscs(60.0, 90.0, liquid_limit=40.0,
                            plasticity_index=22.0)
        assert res.symbol == "CL"

    def test_silt(self) -> None:
        res = classify_uscs(85.0, 100.0, liquid_limit=30.0,
                            plasticity_index=4.0)
        # PI = 4 < A-line = 7.3 -> ML.
        assert res.symbol == "ML"

    def test_cl_ml_zone(self) -> None:
        res = classify_uscs(70.0, 100.0, liquid_limit=25.0,
                            plasticity_index=5.0)
        assert res.symbol == "CL-ML"

    def test_fat_clay_and_elastic_silt(self) -> None:
        ch = classify_uscs(95.0, 100.0, liquid_limit=60.0,
                           plasticity_index=35.0)
        mh = classify_uscs(95.0, 100.0, liquid_limit=60.0,
                           plasticity_index=20.0)
        assert ch.symbol == "CH"   # A-line at LL=60 -> 29.2
        assert mh.symbol == "MH"

    def test_organic(self) -> None:
        ol = classify_uscs(80.0, 100.0, liquid_limit=45.0,
                           plasticity_index=15.0, organic=True)
        oh = classify_uscs(80.0, 100.0, liquid_limit=70.0,
                           plasticity_index=25.0, organic=True)
        assert ol.symbol == "OL"
        assert oh.symbol == "OH"

    def test_pi_from_pl(self) -> None:
        res = classify_uscs(60.0, 95.0, liquid_limit=40.0,
                            plastic_limit=18.0)  # PI = 22
        assert res.symbol == "CL"

    def test_sandy_modifier(self) -> None:
        res = classify_uscs(60.0, 90.0, liquid_limit=35.0,
                            plasticity_index=20.0)
        # 40% coarse (30% sand > 10% gravel) -> "Sandy lean clay".
        assert res.group_name.lower() == "sandy lean clay"

    def test_u_line_warning(self) -> None:
        res = classify_uscs(90.0, 100.0, liquid_limit=30.0,
                            plasticity_index=25.0)
        assert any("U-line" in note for note in res.notes)


class TestCoarseGrained:
    def test_well_graded_gravel(self) -> None:
        res = classify_uscs(2.0, 40.0, d10=0.3, d30=3.0, d60=15.0)
        # Cu = 50 >= 4, Cc = 2.0 in [1, 3] -> GW; sand = 38% -> with sand.
        assert res.symbol == "GW"
        assert "with sand" in res.group_name

    def test_poorly_graded_sand(self) -> None:
        res = classify_uscs(3.0, 98.0, cu=3.0, cc=1.2)
        assert res.symbol == "SP"

    def test_silty_sand(self) -> None:
        res = classify_uscs(20.0, 95.0, liquid_limit=25.0,
                            plasticity_index=3.0)
        assert res.symbol == "SM"

    def test_clayey_gravel(self) -> None:
        res = classify_uscs(18.0, 45.0, liquid_limit=35.0,
                            plasticity_index=18.0)
        assert res.symbol == "GC"

    def test_dual_symbol_sp_sm(self) -> None:
        res = classify_uscs(8.0, 95.0, d10=0.1, d30=0.2, d60=0.4,
                            nonplastic=True)
        # Cu = 4 < 6 -> P; NP fines -> M.
        assert res.symbol == "SP-SM"

    def test_dual_symbol_sw_sc(self) -> None:
        res = classify_uscs(10.0, 96.0, cu=8.0, cc=1.5,
                            liquid_limit=30.0, plasticity_index=15.0)
        assert res.symbol == "SW-SC"

    def test_borderline_cl_ml_fines_take_clayey_branch(self) -> None:
        res = classify_uscs(30.0, 90.0, liquid_limit=25.0,
                            plasticity_index=5.0)
        assert res.symbol == "SC-SM"

    def test_missing_gradation_raises(self) -> None:
        with pytest.raises(ValueError, match="Gradation"):
            classify_uscs(2.0, 50.0)

    def test_missing_atterberg_raises(self) -> None:
        with pytest.raises(ValueError, match="Atterberg"):
            classify_uscs(30.0, 90.0)

    def test_invalid_sieve_data_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed"):
            classify_uscs(60.0, 40.0, liquid_limit=40.0,
                          plasticity_index=20.0)
