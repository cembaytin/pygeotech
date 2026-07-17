"""Mohr-Coulomb shear strength and stress-path utilities.

Failure criterion
-----------------
.. math:: \\tau_f = c' + \\sigma'_n \\tan\\phi'.

The envelope can be fitted either from direct-shear data
:math:`(\\sigma_n, \\tau_f)` by linear regression, or from a set of
triaxial tests via the modified failure line in ``p-q`` space (MIT
convention),

.. math::

    p = \\tfrac{1}{2}(\\sigma_1 + \\sigma_3), \\qquad
    q = \\tfrac{1}{2}(\\sigma_1 - \\sigma_3),

whose best-fit line :math:`q = a + p\\tan\\alpha` maps to the strength
parameters through

.. math:: \\sin\\phi' = \\tan\\alpha, \\qquad c' = \\frac{a}{\\cos\\phi'}.

For effective-stress paths the pore pressure :math:`u` is subtracted
from the normal (mean) stress only: :math:`p' = p - u`, :math:`q' = q`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

__all__ = [
    "MohrCoulomb",
    "stress_path_pq",
    "principal_stresses_at_failure",
]


@dataclass(frozen=True)
class MohrCoulomb:
    """A Mohr-Coulomb failure envelope ``tau = c + sigma_n tan(phi)``.

    Attributes
    ----------
    cohesion
        Cohesion intercept :math:`c` [kPa].
    friction_angle
        Friction angle :math:`\\phi` [degrees].
    """

    cohesion: float
    friction_angle: float

    @property
    def tan_phi(self) -> float:
        """Tangent of the friction angle."""
        return math.tan(math.radians(self.friction_angle))

    def shear_strength(self, sigma_n: float) -> float:
        """Shear strength :math:`\\tau_f` at a normal stress [kPa]."""
        return self.cohesion + sigma_n * self.tan_phi

    def kf_line(self) -> Tuple[float, float]:
        """Return the ``p-q`` failure-line parameters ``(a, tan_alpha)``.

        With :math:`\\tan\\alpha = \\sin\\phi` and :math:`a = c\\cos\\phi`.
        """
        phi = math.radians(self.friction_angle)
        return self.cohesion * math.cos(phi), math.sin(phi)

    @classmethod
    def fit_direct_shear(
        cls, sigma_n: Sequence[float], tau_f: Sequence[float]
    ) -> "MohrCoulomb":
        """Fit the envelope to direct-shear ``(sigma_n, tau_f)`` data.

        A straight line :math:`\\tau = c + \\sigma_n\\tan\\phi` is fitted by
        ordinary least squares.
        """
        sigma = np.asarray(sigma_n, dtype=float)
        tau = np.asarray(tau_f, dtype=float)
        if sigma.size < 2:
            raise ValueError("need at least two data points to fit.")
        slope, intercept = np.polyfit(sigma, tau, 1)
        return cls(cohesion=float(intercept),
                   friction_angle=math.degrees(math.atan(slope)))

    @classmethod
    def fit_triaxial(
        cls, sigma3: Sequence[float], sigma1: Sequence[float]
    ) -> "MohrCoulomb":
        """Fit the envelope to triaxial ``(sigma3, sigma1)`` failure data.

        The ``p-q`` failure line is regressed and converted to
        :math:`c, \\phi`.
        """
        s3 = np.asarray(sigma3, dtype=float)
        s1 = np.asarray(sigma1, dtype=float)
        if s3.size < 2:
            raise ValueError("need at least two tests to fit.")
        p = 0.5 * (s1 + s3)
        q = 0.5 * (s1 - s3)
        tan_alpha, a = np.polyfit(p, q, 1)
        if not -1.0 < tan_alpha < 1.0:
            raise ValueError("fitted slope sin(phi) outside (-1, 1); "
                             "check the input data.")
        phi = math.asin(tan_alpha)
        cohesion = a / math.cos(phi)
        return cls(cohesion=float(cohesion),
                   friction_angle=math.degrees(phi))

    def __str__(self) -> str:
        return (f"Mohr-Coulomb(c = {self.cohesion:.2f} kPa, "
                f"phi = {self.friction_angle:.2f} deg)")


def stress_path_pq(
    sigma1: Sequence[float],
    sigma3: Sequence[float],
    pore_pressure: Sequence[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a ``p-q`` stress path (MIT convention).

    Parameters
    ----------
    sigma1, sigma3
        Major and minor total principal stresses at each stage [kPa].
    pore_pressure
        Pore pressure at each stage [kPa]; if given, ``p_eff = p - u`` is
        also returned, otherwise it equals ``p``.

    Returns
    -------
    (p, q, p_eff)
        Total mean stress ``p``, deviatoric half-difference ``q`` and
        effective mean stress ``p_eff`` (``q_eff == q``).
    """
    s1 = np.asarray(sigma1, dtype=float)
    s3 = np.asarray(sigma3, dtype=float)
    p = 0.5 * (s1 + s3)
    q = 0.5 * (s1 - s3)
    if pore_pressure is None:
        return p, q, p.copy()
    u = np.asarray(pore_pressure, dtype=float)
    return p, q, p - u


def principal_stresses_at_failure(
    envelope: MohrCoulomb, sigma3: float
) -> float:
    """Major principal stress :math:`\\sigma_1` at failure for a given
    confining pressure :math:`\\sigma_3`.

    .. math::

        \\sigma_1 = \\sigma_3\\tan^2\\!\\left(45 + \\tfrac{\\phi}{2}\\right)
        + 2c\\tan\\!\\left(45 + \\tfrac{\\phi}{2}\\right).
    """
    n_phi = math.tan(math.radians(45.0 + envelope.friction_angle / 2.0)) ** 2
    return sigma3 * n_phi + 2.0 * envelope.cohesion * math.sqrt(n_phi)
