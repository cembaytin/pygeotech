"""Laterally loaded piles: a finite-difference p-y (Winkler) solver.

The pile is modelled as a beam on a nonlinear elastic (Winkler)
foundation,

.. math:: E I\\,\\frac{d^4 y}{dz^4} + p(y, z) = 0 ,

where :math:`p(y, z)` is the distributed soil reaction described by a
p-y curve. The equation is discretised with central finite differences
(two ghost nodes at each end) and the nonlinear springs are resolved by
secant iteration. Free-head boundary conditions apply a shear (lateral
load) and a moment at the top; the tip is free (zero shear and moment).

For a *linear* subgrade the solver reproduces Hetenyi's closed-form long-
beam solution, :math:`y_0 = 2 H \\beta / k` with
:math:`\\beta = (k/4EI)^{1/4}`, which is used to validate it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

from pygeotech.constants import GAMMA_W

__all__ = [
    "PyResult",
    "linear_subgrade_py",
    "matlock_clay_py",
    "api_sand_py",
    "api_sand_modulus",
    "solve_laterally_loaded_pile",
]

#: A p-y curve: ``p(depth, deflection) -> soil reaction`` [kN/m].
PyCurve = Callable[[float, float], float]


@dataclass(frozen=True)
class PyResult:
    """Depth profiles from a laterally loaded pile analysis."""

    depth: np.ndarray            # [m]
    deflection: np.ndarray       # y [m]
    moment: np.ndarray           # bending moment [kN.m]
    shear: np.ndarray            # [kN]
    soil_reaction: np.ndarray    # p [kN/m]

    @property
    def head_deflection(self) -> float:
        """Deflection at the pile head [m]."""
        return float(self.deflection[0])

    @property
    def max_moment(self) -> float:
        """Maximum absolute bending moment [kN.m]."""
        return float(np.max(np.abs(self.moment)))


def linear_subgrade_py(k_modulus: float) -> PyCurve:
    """Constant subgrade-reaction p-y curve, ``p = k * y`` [kN/m per m]."""
    def curve(depth: float, y: float) -> float:
        return k_modulus * abs(y)
    return curve


def matlock_clay_py(
    diameter: float,
    su: float,
    gamma_eff: float,
    epsilon50: float = 0.01,
    j_factor: float = 0.5,
) -> PyCurve:
    """Matlock (1970) soft-clay static p-y curve factory.

    Parameters
    ----------
    diameter
        Pile diameter :math:`D` [m].
    su
        Undrained shear strength [kPa].
    gamma_eff
        Effective unit weight [kN/m^3].
    epsilon50
        Strain at 50 % of the failure stress (0.005-0.02).
    j_factor
        Empirical factor :math:`J` (~0.5).
    """
    y50 = 2.5 * epsilon50 * diameter

    def curve(depth: float, y: float) -> float:
        pu_shallow = (3.0 + gamma_eff * depth / su
                      + j_factor * depth / diameter) * su * diameter
        pu = min(pu_shallow, 9.0 * su * diameter)
        ay = abs(y)
        if ay >= 8.0 * y50:
            return pu
        return 0.5 * pu * (ay / y50) ** (1.0 / 3.0)

    return curve


def api_sand_modulus(friction_angle: float) -> float:
    """Initial modulus of subgrade reaction :math:`k` [kN/m^3] for sand.

    Interpolated from the API RP2A chart for sand above the water table
    (:math:`\\phi'` = 25, 30, 35, 40 deg -> k = 5.4, 11, 22, 45 MN/m^3).
    """
    phi = np.array([25.0, 30.0, 35.0, 40.0])
    k_mn = np.array([5.4, 11.0, 22.0, 45.0])       # MN/m^3
    return float(np.interp(friction_angle, phi, k_mn)) * 1.0e3


def api_sand_py(
    diameter: float,
    friction_angle: float,
    gamma_eff: float,
    k_initial: Optional[float] = None,
) -> PyCurve:
    """API RP2A sand p-y curve factory (O'Neill & Murchison / Reese).

    Ultimate resistance is the lesser of the shallow-wedge and deep-flow
    values, :math:`p_u = \\min[(C_1 z + C_2 D)\\gamma' z,\\; C_3 D\\gamma' z]`,
    with :math:`C_1, C_2, C_3` from :math:`\\phi'`; the curve itself is
    :math:`p = A\\,p_u\\tanh\\!\\big(\\tfrac{k z}{A p_u} y\\big)`,
    :math:`A = \\max(3 - 0.8 z/D,\\, 0.9)`.

    Parameters
    ----------
    diameter
        Pile diameter :math:`D` [m].
    friction_angle
        Effective friction angle :math:`\\phi'` [deg].
    gamma_eff
        Effective unit weight [kN/m^3].
    k_initial
        Initial modulus of subgrade reaction [kN/m^3]; defaults to
        :func:`api_sand_modulus`.
    """
    phi = math.radians(friction_angle)
    alpha = phi / 2.0
    beta = math.pi / 4.0 + phi / 2.0
    k0 = 0.4
    ka = (1.0 - math.sin(phi)) / (1.0 + math.sin(phi))
    tan_b = math.tan(beta)
    c1 = (k0 * math.tan(phi) * math.sin(beta) / (math.tan(beta - phi) * math.cos(alpha))
          + tan_b ** 2 * math.tan(alpha) / math.tan(beta - phi)
          + k0 * tan_b * (math.tan(phi) * math.sin(beta) - math.tan(alpha)))
    c2 = tan_b / math.tan(beta - phi) - ka
    c3 = ka * (tan_b ** 8 - 1.0) + k0 * math.tan(phi) * tan_b ** 4
    k_mod = api_sand_modulus(friction_angle) if k_initial is None else k_initial

    def curve(depth: float, y: float) -> float:
        if depth <= 0.0:
            return 0.0
        pu = min((c1 * depth + c2 * diameter) * gamma_eff * depth,
                 c3 * diameter * gamma_eff * depth)
        if pu <= 0.0:
            return 0.0
        a = max(3.0 - 0.8 * depth / diameter, 0.9)
        return a * pu * math.tanh(k_mod * depth * abs(y) / (a * pu))

    return curve


def _secant_stiffness(
    py_curve: PyCurve, depth: float, y: float, y_floor: float
) -> float:
    """Secant soil-spring stiffness ``p/y`` [kN/m per m], floored in ``y``."""
    ay = max(abs(y), y_floor)
    return py_curve(depth, ay) / ay


def solve_laterally_loaded_pile(
    length: float,
    flexural_rigidity: float,
    py_curve: PyCurve,
    lateral_load: float = 0.0,
    moment: float = 0.0,
    n_elements: int = 100,
    max_iter: int = 200,
    tol: float = 1e-8,
    relaxation: float = 0.5,
) -> PyResult:
    """Solve for the response of a free-head laterally loaded pile.

    Parameters
    ----------
    length
        Embedded pile length :math:`L` [m].
    flexural_rigidity
        :math:`EI` [kN.m^2].
    py_curve
        Soil reaction ``p(depth, y)`` [kN/m] (see the factories above).
    lateral_load
        Horizontal load at the head :math:`H` [kN].
    moment
        Applied moment at the head :math:`M_0` [kN.m].
    n_elements
        Number of pile segments.
    max_iter, tol, relaxation
        Secant-iteration controls.

    Returns
    -------
    PyResult
    """
    ei = flexural_rigidity
    n = n_elements
    h = length / n
    z = np.linspace(0.0, length, n + 1)
    size = n + 5                      # nodes -2..n+2 with offset 2
    y_floor = 1e-6

    def assemble(k_soil: np.ndarray) -> np.ndarray:
        a = np.zeros((size, size))
        b = np.zeros(size)
        ei_h2, ei_h3, ei_h4 = ei / h ** 2, ei / h ** 3, ei / h ** 4

        # Row 0: top shear  EI y'''(0) = -H
        a[0, 0] = -ei / (2 * h ** 3)
        a[0, 1] = ei_h3
        a[0, 3] = -ei_h3
        a[0, 4] = ei / (2 * h ** 3)
        b[0] = lateral_load
        # Row 1: top moment  EI y''(0) = M0
        a[1, 1] = ei_h2
        a[1, 2] = -2 * ei_h2
        a[1, 3] = ei_h2
        b[1] = moment
        # Governing rows at nodes i = 0..n  (matrix rows 2..n+2)
        for i in range(n + 1):
            r = i + 2
            a[r, i] += ei_h4
            a[r, i + 1] += -4 * ei_h4
            a[r, i + 2] += 6 * ei_h4 + k_soil[i]
            a[r, i + 3] += -4 * ei_h4
            a[r, i + 4] += ei_h4
        # Row n+3: bottom moment  EI y''(L) = 0
        a[n + 3, n + 1] = ei_h2
        a[n + 3, n + 2] = -2 * ei_h2
        a[n + 3, n + 3] = ei_h2
        # Row n+4: bottom shear  EI y'''(L) = 0
        a[n + 4, n] = -ei / (2 * h ** 3)
        a[n + 4, n + 1] = ei_h3
        a[n + 4, n + 3] = -ei_h3
        a[n + 4, n + 4] = ei / (2 * h ** 3)
        return a, b

    # Secant iteration on the soil springs.
    y = np.zeros(n + 1)
    k_soil = np.array([_secant_stiffness(py_curve, z[i], y_floor, y_floor)
                       for i in range(n + 1)])
    for _ in range(max_iter):
        a, b = assemble(k_soil)
        u = np.linalg.solve(a, b)
        y_new = u[2:n + 3]
        y = (1.0 - relaxation) * y + relaxation * y_new
        k_new = np.array([_secant_stiffness(py_curve, z[i], y[i], y_floor)
                          for i in range(n + 1)])
        if np.max(np.abs(k_new - k_soil)) <= tol * (np.max(k_soil) + 1.0):
            k_soil = k_new
            break
        k_soil = k_new

    # Recover response quantities with the final displacement field.
    a, b = assemble(k_soil)
    u = np.linalg.solve(a, b)
    deflection = u[2:n + 3]
    # Bending moment M = EI y'' via the central second difference.
    moment_arr = np.array([
        ei * (u[i] - 2 * u[i + 1] + u[i + 2]) / h ** 2
        for i in range(n + 1)])
    # Shear V = dM/dz.
    shear_arr = np.gradient(moment_arr, z)
    reaction = k_soil * deflection
    return PyResult(z, deflection, moment_arr, shear_arr, reaction)
