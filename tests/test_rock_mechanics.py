"""Unit tests for the rock_mechanics submodule."""

import pytest

from pygeotech.rock_mechanics import (
    gsi_from_rmr,
    hoek_brown_parameters,
    hoek_brown_strength,
    q_system,
    rock_mass_rating,
)


class TestRMR:
    def test_reference_sum_and_class(self) -> None:
        # 12 + 17 + 15 + 25 + 15 = 84 -> Class I.
        res = rock_mass_rating(120, 80, 0.8, 25, 15)
        assert res.rmr == 84
        assert res.rock_class == "I"

    def test_orientation_adjustment(self) -> None:
        base = rock_mass_rating(120, 80, 0.8, 25, 15).rmr
        adj = rock_mass_rating(120, 80, 0.8, 25, 15,
                               orientation_adjustment=-12).rmr
        assert adj == base - 12

    def test_class_boundaries(self) -> None:
        # Weak rock mass -> poor class.
        res = rock_mass_rating(3, 20, 0.05, 10, 7)
        assert res.rock_class in ("IV", "V")


class TestQSystem:
    def test_product(self) -> None:
        q = q_system(rqd=80, jn=9, jr=2, ja=1, jw=1, srf=2.5)
        assert q == pytest.approx((80 / 9) * (2 / 1) * (1 / 2.5))

    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            q_system(80, 0, 2, 1, 1, 2.5)


class TestGSI:
    def test_from_rmr(self) -> None:
        assert gsi_from_rmr(60) == 55.0

    def test_low_rmr_raises(self) -> None:
        with pytest.raises(ValueError):
            gsi_from_rmr(20)


class TestHoekBrown:
    def test_parameter_reference_values(self) -> None:
        p = hoek_brown_parameters(gsi=50, mi=10, sigma_ci=80)
        assert p.mb == pytest.approx(1.677, abs=1e-3)
        assert p.s == pytest.approx(0.0039, abs=2e-4)
        assert p.a == pytest.approx(0.506, abs=1e-3)

    def test_disturbance_weakens_mass(self) -> None:
        undis = hoek_brown_parameters(50, 10, 80, disturbance=0.0)
        dist = hoek_brown_parameters(50, 10, 80, disturbance=1.0)
        assert dist.mb < undis.mb
        assert dist.s < undis.s

    def test_uniaxial_and_tensile_strength(self) -> None:
        p = hoek_brown_parameters(gsi=50, mi=10, sigma_ci=80)
        assert p.uniaxial_compressive_strength() == pytest.approx(
            80 * p.s ** p.a)
        assert p.tensile_strength() < 0.0

    def test_strength_increases_with_confinement(self) -> None:
        p = hoek_brown_parameters(gsi=60, mi=12, sigma_ci=100)
        assert p.major_principal_stress(5.0) > p.major_principal_stress(0.0)

    def test_free_function_matches_method(self) -> None:
        p = hoek_brown_parameters(gsi=55, mi=8, sigma_ci=60)
        assert hoek_brown_strength(3.0, 60, p.mb, p.s, p.a) == pytest.approx(
            float(p.major_principal_stress(3.0)))
