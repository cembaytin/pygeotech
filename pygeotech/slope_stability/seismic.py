"""Seismic slope displacement by the Newmark sliding-block method.

A potential sliding mass is treated as a rigid block on an inclined plane
with a yield acceleration :math:`a_y` (the pseudo-static acceleration
that brings the factor of safety to unity). Whenever the base
acceleration exceeds :math:`a_y` the block slips; integrating the
relative acceleration twice over these intervals gives the permanent
(Newmark) displacement.

.. math::

    \\ddot{u}_{rel}(t) = a(t) - a_y \\quad\\text{while sliding},

with sliding starting when :math:`a(t) > a_y` and stopping when the
relative velocity returns to zero.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

__all__ = ["yield_acceleration", "newmark_displacement"]


def yield_acceleration(
    static_factor_of_safety: float,
    slope_angle: float,
    g: float = 9.81,
) -> float:
    """Yield acceleration :math:`a_y` of a sliding block [m/s^2].

    A common first-order estimate is

    .. math:: a_y = (F_s - 1)\\,g\\sin\\beta,

    with ``F_s`` the static factor of safety and :math:`\\beta` the
    slope/failure-plane inclination.
    """
    import math
    return max(0.0, (static_factor_of_safety - 1.0) * g
               * math.sin(math.radians(slope_angle)))


def newmark_displacement(
    acceleration: Sequence[float],
    dt: float,
    yield_accel: float,
) -> Tuple[float, np.ndarray]:
    """Permanent displacement of a Newmark sliding block.

    Parameters
    ----------
    acceleration
        Base acceleration time history :math:`a(t)` [m/s^2] (positive in
        the downslope-driving sense).
    dt
        Time step [s].
    yield_accel
        Yield acceleration :math:`a_y` [m/s^2] (>= 0).

    Returns
    -------
    (total_displacement, displacement_history)
        Final permanent displacement [m] and the cumulative displacement
        at each time step.

    Notes
    -----
    One-directional (downslope) sliding is assumed: the block accumulates
    velocity while ``a > a_y`` and decelerates afterwards, but never slides
    back upslope.
    """
    a = np.asarray(acceleration, dtype=float)
    if yield_accel < 0.0:
        raise ValueError("yield_accel must be non-negative.")
    velocity = 0.0
    displacement = 0.0
    history = np.empty(a.size, dtype=float)
    for i, ai in enumerate(a):
        if velocity > 0.0 or ai > yield_accel:
            velocity += (ai - yield_accel) * dt
            if velocity < 0.0:
                velocity = 0.0
            displacement += velocity * dt
        history[i] = displacement
    return displacement, history
