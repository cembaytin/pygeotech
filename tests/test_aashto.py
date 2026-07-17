"""Unit tests for AASHTO classification."""

import pytest

from pygeotech.phase_relations import classify_aashto, group_index


class TestAASHTO:
    def test_a1a_gravel(self) -> None:
        res = classify_aashto(50, 30, 8, liquid_limit=0, plasticity_index=0)
        assert res.group == "A-1-a"
        assert res.group_index == 0

    def test_a3_fine_sand(self) -> None:
        res = classify_aashto(100, 80, 5, liquid_limit=0, plasticity_index=0)
        assert res.group == "A-3"

    def test_a2_6(self) -> None:
        # <=35% fines, LL<=40, PI>=11 -> A-2-6 with partial GI.
        res = classify_aashto(80, 60, 30, liquid_limit=38, plasticity_index=14)
        assert res.group == "A-2-6"

    def test_a6_clay(self) -> None:
        res = classify_aashto(100, 95, 60, liquid_limit=38, plasticity_index=18)
        assert res.group == "A-6"
        assert res.group_index > 0

    def test_a7_5_vs_a7_6(self) -> None:
        # PI <= LL-30 -> A-7-5 ; PI > LL-30 -> A-7-6.
        a75 = classify_aashto(100, 95, 60, liquid_limit=60, plasticity_index=25)
        a76 = classify_aashto(100, 95, 60, liquid_limit=45, plasticity_index=20)
        assert a75.group == "A-7-5"
        assert a76.group == "A-7-6"

    def test_group_index_reference(self) -> None:
        # Textbook: F200=60, LL=45, PI=20 -> GI ~ 9.
        gi = group_index(60.0, 45.0, 20.0)
        assert gi == pytest.approx(9, abs=1)

    def test_group_index_non_negative(self) -> None:
        assert group_index(20.0, 20.0, 5.0) == 0
