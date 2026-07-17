"""Modified Cam-Clay constitutive model with a triaxial stress-point driver.

Modified Cam-Clay (Roscoe & Burland, 1968) is an elastoplastic critical-
state model. In the mean-effective / deviatoric stress plane :math:`(p',
q)` the yield surface is the ellipse

.. math:: f(p', q, p'_c) = q^2 + M^2\\,p'\\,(p' - p'_c) = 0 ,

where :math:`M` is the critical-state slope and :math:`p'_c` the
preconsolidation pressure (isotropic hardening). Elasticity is pressure
dependent, :math:`K = (1+e)p'/\\kappa`, with shear modulus from Poisson's
ratio. Associated flow and volumetric hardening,

.. math:: \\dot{p}'_c = p'_c\\,\\frac{1+e}{\\lambda-\\kappa}\\,
    \\dot{\\varepsilon}_v^{\\,p},

drive the state towards the critical-state line :math:`q = M p'`, on which
the model shears at constant volume. The :func:`triaxial_test` driver
integrates the model under drained or undrained triaxial conditions and
is validated against the critical-state end point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = ["CamClayParameters", "TriaxialResult", "triaxial_test"]


@dataclass(frozen=True)
class CamClayParameters:
    """Modified Cam-Clay material parameters.

    Parameters
    ----------
    M
        Critical-state stress ratio :math:`M = q/p'` at failure.
    lam
        Slope :math:`\\lambda` of the normal compression line in
        :math:`e`-:math:`\\ln p'`.
    kappa
        Slope :math:`\\kappa` of the swelling line.
    nu
        Poisson's ratio.
    e0
        Initial void ratio at the start of the test.
    """

    M: float
    lam: float
    kappa: float
    nu: float
    e0: float

    def friction_angle(self) -> float:
        """Critical-state friction angle [deg] from ``M`` (compression)."""
        return math.degrees(math.asin(3.0 * self.M / (6.0 + self.M)))


@dataclass(frozen=True)
class TriaxialResult:
    """Simulated triaxial-test paths (all arrays aligned)."""

    axial_strain: np.ndarray
    q: np.ndarray                 # deviatoric stress [kPa]
    p_eff: np.ndarray             # mean effective stress [kPa]
    volumetric_strain: np.ndarray
    pore_pressure: np.ndarray     # excess pore pressure [kPa] (undrained)
    void_ratio: np.ndarray
    drained: bool

    @property
    def deviator_at_failure(self) -> float:
        """Deviatoric stress at the end of the test [kPa]."""
        return float(self.q[-1])


def _elastic_moduli(p_eff: float, e: float,
                    params: CamClayParameters) -> Tuple[float, float]:
    """Pressure-dependent bulk and shear moduli ``(K, G)``."""
    k_bulk = (1.0 + e) * p_eff / params.kappa
    g_shear = 3.0 * k_bulk * (1.0 - 2.0 * params.nu) / (2.0 * (1.0 + params.nu))
    return k_bulk, g_shear


def _yield(p_eff: float, q: float, pc: float, m: float) -> float:
    """Modified Cam-Clay yield function value."""
    return q ** 2 + m ** 2 * p_eff * (p_eff - pc)


def triaxial_test(
    params: CamClayParameters,
    p0: float,
    ocr: float = 1.0,
    drained: bool = True,
    total_axial_strain: float = 0.20,
    n_steps: int = 4000,
) -> TriaxialResult:
    """Simulate a strain-controlled triaxial compression test.

    Parameters
    ----------
    params
        :class:`CamClayParameters`.
    p0
        Initial mean effective (= isotropic consolidation) stress [kPa].
    ocr
        Overconsolidation ratio; the initial preconsolidation pressure is
        ``p'_c = ocr * p0``.
    drained
        ``True`` for a drained (CD) test, ``False`` for undrained (CU).
    total_axial_strain
        Total axial strain applied (compression positive).
    n_steps
        Number of strain increments.

    Returns
    -------
    TriaxialResult
    """
    m = params.M
    pc = ocr * p0
    p_eff = p0
    q = 0.0
    e = params.e0
    eps_v = 0.0
    total_mean = p0                     # total mean stress (cell + axial/3)

    d_eps_a = total_axial_strain / n_steps
    ea = np.linspace(0.0, total_axial_strain, n_steps + 1)
    q_hist = np.zeros(n_steps + 1)
    p_hist = np.zeros(n_steps + 1)
    ev_hist = np.zeros(n_steps + 1)
    u_hist = np.zeros(n_steps + 1)
    e_hist = np.zeros(n_steps + 1)
    p_hist[0], e_hist[0] = p_eff, e

    for i in range(1, n_steps + 1):
        k_bulk, g_shear = _elastic_moduli(p_eff, e, params)

        # Plastic gradient / hardening at the current state.
        df_dp = m ** 2 * (2.0 * p_eff - pc)
        df_dq = 2.0 * q
        hard = (m ** 4 * p_eff * pc * (1.0 + e) / (params.lam - params.kappa)
                * (2.0 * p_eff - pc))

        on_yield = _yield(p_eff, q, pc, m) >= -1e-6 and df_dp * 0 == 0
        # Elastoplastic tangent in (p', q) space; falls back to elastic.
        if on_yield:
            a = np.array([df_dp, df_dq])
            d_e = np.diag([k_bulk, 3.0 * g_shear])
            d_a = d_e @ a
            denom = a @ d_e @ a + hard
            d_ep = d_e - np.outer(d_a, d_a) / denom if denom > 1e-9 else d_e
        else:
            d_ep = np.diag([k_bulk, 3.0 * g_shear])

        if drained:
            # Solve radial strain so that d(sigma3') = dp' - dq/3 = 0.
            p_row = d_ep[0, 0] - d_ep[1, 0] / 3.0
            q_row = (d_ep[0, 1] - d_ep[1, 1] / 3.0) * (2.0 / 3.0)
            denom_r = 2.0 * p_row - q_row
            r = -(p_row + q_row) * d_eps_a / denom_r if abs(denom_r) > 1e-12 \
                else 0.0
            d_ev = d_eps_a + 2.0 * r
            d_es = (2.0 / 3.0) * (d_eps_a - r)
        else:
            d_ev = 0.0                  # undrained: no volume change
            d_es = d_eps_a

        d_p = d_ep[0, 0] * d_ev + d_ep[0, 1] * d_es
        d_q = d_ep[1, 0] * d_ev + d_ep[1, 1] * d_es
        p_eff += d_p
        q += d_q
        p_eff = max(p_eff, 1e-3)

        # Hardening update from the plastic volumetric strain.
        if on_yield and (params.lam - params.kappa) > 0:
            d_ev_e = d_p / k_bulk
            d_ev_p = d_ev - d_ev_e
            pc += pc * (1.0 + e) / (params.lam - params.kappa) * d_ev_p
            pc = max(pc, 1e-3)
            # Drift correction: pull q back onto the yield surface.
            arg = m ** 2 * p_eff * (pc - p_eff)
            if arg > 0:
                q = math.copysign(math.sqrt(arg), q) if q != 0 else math.sqrt(arg)

        eps_v += d_ev
        e -= (1.0 + e) * d_ev
        if drained:
            u = 0.0
        else:
            total_mean += d_q / 3.0     # cell pressure constant -> dp_tot=dq/3
            u = total_mean - p_eff
        q_hist[i], p_hist[i], ev_hist[i], u_hist[i], e_hist[i] = \
            q, p_eff, eps_v, u, e

    return TriaxialResult(ea, q_hist, p_hist, ev_hist, u_hist, e_hist, drained)
