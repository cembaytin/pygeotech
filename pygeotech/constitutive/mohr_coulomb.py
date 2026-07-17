"""Elastoplastic Mohr-Coulomb constitutive model (triaxial stress-point driver).

The Mohr-Coulomb model is linear-elastic, perfectly plastic: below the
failure line it is elastic, on it the stress state is fixed. In triaxial
compression the failure line in the :math:`(p', q)` plane is

.. math:: q_f = M_\\phi\\,p' + k_c, \\qquad
    M_\\phi = \\frac{6\\sin\\phi}{3-\\sin\\phi}, \\quad
    k_c = \\frac{6\\,c\\cos\\phi}{3-\\sin\\phi},

so a cohesionless soil fails at :math:`q_f/p'_f = M_\\phi`. A non-associated
flow rule uses the dilation angle :math:`\\psi` through
:math:`M_\\psi = 6\\sin\\psi/(3-\\sin\\psi)`, so drained shearing dilates
(:math:`\\dot\\varepsilon_v < 0`) at a rate set by :math:`\\psi`.

The :func:`mc_triaxial_test` driver integrates the model under drained or
undrained triaxial conditions and is validated against the analytical
failure ratio :math:`q_f/p'_f = M_\\phi`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pygeotech.constitutive.cam_clay import TriaxialResult

__all__ = ["MohrCoulombParameters", "mc_triaxial_test"]


@dataclass(frozen=True)
class MohrCoulombParameters:
    """Linear-elastic perfectly-plastic Mohr-Coulomb parameters.

    Parameters
    ----------
    youngs_modulus
        Young's modulus :math:`E` [kPa].
    poisson_ratio
        Poisson's ratio :math:`\\nu`.
    friction_angle
        Friction angle :math:`\\phi` [deg].
    cohesion
        Cohesion :math:`c` [kPa].
    dilation_angle
        Dilation angle :math:`\\psi` [deg] (0 to :math:`\\phi`).
    """

    youngs_modulus: float
    poisson_ratio: float
    friction_angle: float
    cohesion: float = 0.0
    dilation_angle: float = 0.0

    def M_phi(self) -> float:
        """Critical stress ratio :math:`M_\\phi = 6\\sin\\phi/(3-\\sin\\phi)`."""
        s = math.sin(math.radians(self.friction_angle))
        return 6.0 * s / (3.0 - s)

    def M_psi(self) -> float:
        """Dilatancy ratio :math:`M_\\psi = 6\\sin\\psi/(3-\\sin\\psi)`."""
        s = math.sin(math.radians(self.dilation_angle))
        return 6.0 * s / (3.0 - s)

    def _cohesion_intercept(self) -> float:
        phi = math.radians(self.friction_angle)
        return 6.0 * self.cohesion * math.cos(phi) / (3.0 - math.sin(phi))


def mc_triaxial_test(
    params: MohrCoulombParameters,
    p0: float,
    drained: bool = True,
    total_axial_strain: float = 0.05,
    n_steps: int = 2000,
) -> TriaxialResult:
    """Simulate a strain-controlled triaxial test with the Mohr-Coulomb model.

    Parameters
    ----------
    params
        :class:`MohrCoulombParameters`.
    p0
        Initial isotropic effective stress [kPa].
    drained
        ``True`` for drained (CD), ``False`` for undrained (CU).
    total_axial_strain
        Total axial strain (compression positive).
    n_steps
        Number of increments.

    Returns
    -------
    TriaxialResult
    """
    e_mod, nu = params.youngs_modulus, params.poisson_ratio
    k_bulk = e_mod / (3.0 * (1.0 - 2.0 * nu))
    g_shear = e_mod / (2.0 * (1.0 + nu))
    m_phi, m_psi = params.M_phi(), params.M_psi()
    k_c = params._cohesion_intercept()

    p_eff, q, eps_v, total_mean = p0, 0.0, 0.0, p0
    d_eps_a = total_axial_strain / n_steps

    ea = np.linspace(0.0, total_axial_strain, n_steps + 1)
    q_h = np.zeros(n_steps + 1)
    p_h = np.zeros(n_steps + 1)
    ev_h = np.zeros(n_steps + 1)
    u_h = np.zeros(n_steps + 1)
    p_h[0] = p0

    a_f = np.array([-m_phi, 1.0])           # yield gradient (dp', dq)
    a_g = np.array([-m_psi, 1.0])           # plastic-potential gradient
    d_e = np.diag([k_bulk, 3.0 * g_shear])

    for i in range(1, n_steps + 1):
        yielding = (q - m_phi * p_eff - k_c) >= -1e-6 and q > 0
        if yielding:
            denom = a_f @ d_e @ a_g          # H = 0 (perfectly plastic)
            d_ep = d_e - np.outer(d_e @ a_g, a_f @ d_e) / denom
        else:
            d_ep = d_e

        if drained:
            p_row = d_ep[0, 0] - d_ep[1, 0] / 3.0
            q_row = (d_ep[0, 1] - d_ep[1, 1] / 3.0) * (2.0 / 3.0)
            denom_r = 2.0 * p_row - q_row
            r = (-(p_row + q_row) * d_eps_a / denom_r
                 if abs(denom_r) > 1e-12 else 0.0)
            d_ev, d_es = d_eps_a + 2.0 * r, (2.0 / 3.0) * (d_eps_a - r)
        else:
            d_ev, d_es = 0.0, d_eps_a

        d_p = d_ep[0, 0] * d_ev + d_ep[0, 1] * d_es
        d_q = d_ep[1, 0] * d_ev + d_ep[1, 1] * d_es
        p_eff = max(p_eff + d_p, 1e-3)
        q += d_q
        # Drift correction back onto the failure line.
        if yielding and q > m_phi * p_eff + k_c:
            q = m_phi * p_eff + k_c
        eps_v += d_ev
        if drained:
            u = 0.0
        else:
            total_mean += d_q / 3.0
            u = total_mean - p_eff
        q_h[i], p_h[i], ev_h[i], u_h[i] = q, p_eff, eps_v, u

    return TriaxialResult(ea, q_h, p_h, ev_h, u_h, np.full(n_steps + 1, np.nan),
                          drained)
