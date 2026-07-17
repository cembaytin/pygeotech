"""Tests for the constitutive submodule (Modified Cam-Clay).

Validated against critical-state theory: every triaxial test must end on
the critical-state line q = M p'."""

import math

import pytest

from pygeotech.constitutive import CamClayParameters, triaxial_test


def _params():
    return CamClayParameters(M=0.95, lam=0.16, kappa=0.03, nu=0.3, e0=1.5)


class TestCriticalState:
    def test_undrained_ends_on_csl(self) -> None:
        # The undrained effective stress path terminates exactly on the CSL.
        p = _params()
        for ocr in (1.0, 2.0, 6.0):
            res = triaxial_test(p, p0=200.0, ocr=ocr, drained=False)
            assert res.q[-1] / res.p_eff[-1] == pytest.approx(p.M, abs=0.02)

    def test_drained_approaches_csl(self) -> None:
        p = _params()
        res = triaxial_test(p, p0=200.0, ocr=1.0, drained=True,
                            total_axial_strain=0.6)
        assert res.q[-1] / res.p_eff[-1] == pytest.approx(p.M, abs=0.02)

    def test_friction_angle(self) -> None:
        # M = 0.95 -> phi_cs approx 24.2 deg.
        p = _params()
        assert p.friction_angle() == pytest.approx(24.2, abs=0.5)


class TestUndrainedBehaviour:
    def test_nc_positive_pore_pressure(self) -> None:
        # Normally consolidated clay contracts -> positive excess PWP.
        res = triaxial_test(_params(), p0=200.0, ocr=1.0, drained=False)
        assert res.pore_pressure[-1] > 0.0

    def test_oc_negative_pore_pressure(self) -> None:
        # Heavily overconsolidated clay dilates -> negative excess PWP.
        res = triaxial_test(_params(), p0=200.0, ocr=6.0, drained=False)
        assert res.pore_pressure[-1] < 0.0

    def test_undrained_strength_increases_with_ocr(self) -> None:
        nc = triaxial_test(_params(), p0=200.0, ocr=1.0, drained=False)
        oc = triaxial_test(_params(), p0=200.0, ocr=6.0, drained=False)
        assert oc.q[-1] / 2.0 > nc.q[-1] / 2.0        # su = q_f / 2

    def test_undrained_constant_volume(self) -> None:
        res = triaxial_test(_params(), p0=200.0, ocr=2.0, drained=False)
        assert res.volumetric_strain[-1] == pytest.approx(0.0, abs=1e-9)


class TestDrainedBehaviour:
    def test_nc_contracts(self) -> None:
        # Drained NC clay compresses (positive volumetric strain).
        res = triaxial_test(_params(), p0=200.0, ocr=1.0, drained=True)
        assert res.volumetric_strain[-1] > 0.0
        assert res.pore_pressure[-1] == 0.0

    def test_stress_and_strain_monotone_start(self) -> None:
        res = triaxial_test(_params(), p0=200.0, ocr=1.0, drained=True)
        assert res.q[1] > res.q[0]
        assert res.void_ratio[-1] < res.void_ratio[0]   # densifies
