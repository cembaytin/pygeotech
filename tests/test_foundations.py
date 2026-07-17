"""Unit tests for the foundations submodule."""

import math

import pytest

from pygeotech.foundations import (
    ShallowFoundation,
    bearing_capacity_factors,
)


class TestFactors:
    def test_vesic_phi30_reference(self) -> None:
        nc, nq, ng = bearing_capacity_factors(30.0, "vesic")
        assert nc == pytest.approx(30.14, abs=0.05)
        assert nq == pytest.approx(18.40, abs=0.05)
        assert ng == pytest.approx(22.40, abs=0.05)

    def test_phi_zero(self) -> None:
        nc, nq, ng = bearing_capacity_factors(0.0, "vesic")
        assert (nc, nq, ng) == pytest.approx((5.14, 1.0, 0.0))

    def test_hansen_meyerhof_share_nc_nq(self) -> None:
        h = bearing_capacity_factors(32.0, "hansen")
        m = bearing_capacity_factors(32.0, "meyerhof")
        assert h[0] == pytest.approx(m[0])   # Nc
        assert h[1] == pytest.approx(m[1])   # Nq
        assert h[2] != pytest.approx(m[2])   # Ngamma differs

    def test_invalid_method(self) -> None:
        with pytest.raises(ValueError):
            bearing_capacity_factors(30.0, "bogus")


class TestShallowFoundation:
    def test_strip_clay_terzaghi_form(self) -> None:
        # phi = 0 clay, no shape/depth factors: qu = c*Nc + q.
        f = ShallowFoundation(
            width=2.0, length=None, depth=1.0, gamma=18.0,
            cohesion=50.0, phi=0.0, method="vesic",
            apply_shape=False, apply_depth=False)
        res = f.capacity()
        assert res.q_ultimate == pytest.approx(50.0 * 5.14 + 18.0 * 1.0,
                                              abs=1e-6)
        assert res.q_net_ultimate == pytest.approx(res.q_ultimate - 18.0)

    def test_allowable_uses_fs(self) -> None:
        f = ShallowFoundation(width=2.0, depth=1.0, phi=30.0, cohesion=10.0,
                              factor_of_safety=3.0)
        res = f.capacity()
        assert res.q_allowable_gross == pytest.approx(res.q_ultimate / 3.0)
        assert res.q_allowable_net == pytest.approx(res.q_net_ultimate / 3.0)

    def test_water_table_reduces_capacity(self) -> None:
        base = dict(width=2.0, depth=1.0, gamma=18.0, gamma_sat=20.0,
                    cohesion=0.0, phi=32.0)
        dry = ShallowFoundation(water_table_depth=math.inf, **base).capacity()
        wet = ShallowFoundation(water_table_depth=0.0, **base).capacity()
        assert wet.q_ultimate < dry.q_ultimate

    def test_allowable_load_scales_with_area(self) -> None:
        f = ShallowFoundation(width=2.0, length=3.0, depth=1.0, phi=30.0,
                              cohesion=10.0)
        load = f.allowable_load()
        assert load == pytest.approx(f.capacity().q_allowable_net * 6.0)
