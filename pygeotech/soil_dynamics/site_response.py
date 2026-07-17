"""1-D seismic site response and dynamic soil properties.

For a uniform soil layer of thickness :math:`H` and shear-wave velocity
:math:`V_s` on rigid rock, the fundamental period is

.. math:: T_s = \\frac{4H}{V_s},

and the (viscously damped) transfer-function amplitude between the rock
and the surface is approximated by

.. math::

    |F(\\omega)| = \\frac{1}{\\sqrt{\\cos^2(\\omega H/V_s)
        + \\big(\\zeta\\,\\omega H/V_s\\big)^2}},

which peaks at the natural frequencies with an amplification of about
:math:`1/[\\zeta(2n-1)\\pi/2]`. The small-strain shear modulus follows
from :math:`G_{max} = \\rho V_s^2`.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np

__all__ = [
    "max_shear_modulus",
    "shear_wave_velocity",
    "site_natural_period",
    "site_natural_frequency",
    "transfer_function_amplitude",
    "peak_amplification",
]

Number = Union[float, np.ndarray]


def max_shear_modulus(density: float, vs: float) -> float:
    """Small-strain shear modulus :math:`G_{max} = \\rho V_s^2` [kPa].

    Parameters
    ----------
    density
        Mass density :math:`\\rho` [kg/m^3] (or Mg/m^3 for MPa output).
    vs
        Shear-wave velocity :math:`V_s` [m/s].

    Notes
    -----
    With ``density`` in kg/m^3 and ``vs`` in m/s the result is in Pa;
    divide by 1000 for kPa. With ``density`` in Mg/m^3 the result is kPa.
    """
    return density * vs ** 2


def shear_wave_velocity(gmax: float, density: float) -> float:
    """Shear-wave velocity from :math:`G_{max}` and density."""
    if density <= 0.0:
        raise ValueError("density must be positive.")
    return math.sqrt(gmax / density)


def site_natural_period(thickness: float, vs: float) -> float:
    """Fundamental site period :math:`T_s = 4H/V_s` [s]."""
    if vs <= 0.0:
        raise ValueError("vs must be positive.")
    return 4.0 * thickness / vs


def site_natural_frequency(thickness: float, vs: float, mode: int = 1) -> float:
    """Natural frequency of mode ``n`` [Hz]: :math:`(2n-1)V_s/(4H)`."""
    return (2 * mode - 1) * vs / (4.0 * thickness)


def transfer_function_amplitude(
    frequency: Number, thickness: float, vs: float, damping: float
) -> Number:
    """Rock-to-surface transfer-function amplitude of a uniform layer.

    Parameters
    ----------
    frequency
        Frequency [Hz] (scalar or array).
    thickness, vs
        Layer thickness [m] and shear-wave velocity [m/s].
    damping
        Damping ratio :math:`\\zeta` (e.g. 0.05).
    """
    omega = 2.0 * np.pi * np.asarray(frequency, dtype=float)
    ks = omega * thickness / vs
    return 1.0 / np.sqrt(np.cos(ks) ** 2 + (damping * ks) ** 2)


def peak_amplification(damping: float, mode: int = 1) -> float:
    """Approximate resonant amplification of mode ``n``.

    :math:`A_n \\approx 1/[\\zeta\\,(2n-1)\\,\\pi/2]`.
    """
    if damping <= 0.0:
        raise ValueError("damping must be positive.")
    return 1.0 / (damping * (2 * mode - 1) * math.pi / 2.0)
