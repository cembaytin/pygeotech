"""Bearing capacity of shallow foundations.

General bearing-capacity equation (Terzaghi form, with shape and depth
corrections after Hansen/Vesic):

.. math::

    q_u = c\\,N_c\\,s_c d_c + q\\,N_q\\,s_q d_q
          + \\tfrac{1}{2}\\gamma B\\,N_\\gamma\\,s_\\gamma d_\\gamma .

Bearing-capacity factors
------------------------
For the Meyerhof / Hansen / Vesic families,

.. math::

    N_q = e^{\\pi\\tan\\phi}\\tan^2\\!\\left(45 + \\tfrac{\\phi}{2}\\right),
    \\qquad
    N_c = (N_q - 1)\\cot\\phi ,

with the depth-of-mechanism term

.. math::

    N_\\gamma = \\begin{cases}
        (N_q - 1)\\tan(1.4\\phi) & \\text{(Meyerhof)}\\\\
        1.5\\,(N_q - 1)\\tan\\phi & \\text{(Hansen)}\\\\
        2\\,(N_q + 1)\\tan\\phi   & \\text{(Vesic).}
    \\end{cases}

Terzaghi uses :math:`N_q = e^{(3\\pi/2 - \\phi)\\tan\\phi} /
[2\\cos^2(45 + \\phi/2)]` and :math:`N_c = 5.70` at :math:`\\phi = 0`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

__all__ = [
    "bearing_capacity_factors",
    "BearingCapacityResult",
    "ShallowFoundation",
]

_METHODS = ("terzaghi", "meyerhof", "hansen", "vesic")


def bearing_capacity_factors(
    phi: float, method: str = "vesic"
) -> Tuple[float, float, float]:
    """Return the bearing-capacity factors ``(Nc, Nq, Ngamma)``.

    Parameters
    ----------
    phi
        Friction angle :math:`\\phi` [degrees], 0 to < 90.
    method
        One of ``"terzaghi"``, ``"meyerhof"``, ``"hansen"``, ``"vesic"``.

    Examples
    --------
    >>> nc, nq, ng = bearing_capacity_factors(30.0, "vesic")
    >>> round(nc, 2), round(nq, 2), round(ng, 2)
    (30.14, 18.4, 22.4)
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}.")
    if not 0.0 <= phi < 90.0:
        raise ValueError("phi must lie in [0, 90) degrees.")
    phi_r = math.radians(phi)
    tan_phi = math.tan(phi_r)

    if method == "terzaghi":
        if phi == 0.0:
            return 5.70, 1.0, 0.0
        nq = (math.exp((1.5 * math.pi - phi_r) * tan_phi)
              / (2.0 * math.cos(math.radians(45.0 + phi / 2.0)) ** 2))
        nc = (nq - 1.0) / tan_phi
        # Terzaghi's N_gamma via Vesic's widely used approximation.
        n_gamma = 2.0 * (nq + 1.0) * tan_phi
        return nc, nq, n_gamma

    nq = math.exp(math.pi * tan_phi) * math.tan(
        math.radians(45.0 + phi / 2.0)) ** 2
    nc = 5.14 if phi == 0.0 else (nq - 1.0) / tan_phi
    if method == "meyerhof":
        n_gamma = (nq - 1.0) * math.tan(1.4 * phi_r)
    elif method == "hansen":
        n_gamma = 1.5 * (nq - 1.0) * tan_phi
    else:  # vesic
        n_gamma = 2.0 * (nq + 1.0) * tan_phi
    return nc, nq, n_gamma


@dataclass(frozen=True)
class BearingCapacityResult:
    """Result of a shallow-foundation bearing-capacity calculation.

    All stresses in kPa.
    """

    q_ultimate: float
    q_net_ultimate: float
    q_allowable_net: float
    q_allowable_gross: float
    factor_of_safety: float
    nc: float
    nq: float
    n_gamma: float

    def __str__(self) -> str:
        return (f"q_ult = {self.q_ultimate:.1f} kPa | "
                f"q_net,ult = {self.q_net_ultimate:.1f} kPa | "
                f"q_all,net = {self.q_allowable_net:.1f} kPa "
                f"(FS = {self.factor_of_safety:g})")


class ShallowFoundation:
    """A shallow (strip / rectangular) foundation.

    Parameters
    ----------
    width
        Foundation width :math:`B` [m] (the shorter plan dimension).
    length
        Foundation length :math:`L` [m]; ``None`` for a continuous strip.
    depth
        Founding depth :math:`D_f` [m].
    gamma
        Moist unit weight of the soil [kN/m^3].
    cohesion
        Cohesion :math:`c` [kPa].
    phi
        Friction angle :math:`\\phi` [degrees].
    method
        Bearing-capacity factor set (see :func:`bearing_capacity_factors`).
    gamma_sat
        Saturated unit weight [kN/m^3]; defaults to ``gamma``.
    water_table_depth
        Depth to the water table from the ground surface [m]
        (``inf`` = dry, the default).
    factor_of_safety
        Global factor of safety applied to the ultimate capacity.
    apply_shape, apply_depth
        Whether to include Vesic/Hansen shape and depth correction
        factors (default ``True``).
    gamma_w
        Unit weight of water [kN/m^3].
    """

    def __init__(
        self,
        width: float,
        length: Optional[float] = None,
        depth: float = 0.0,
        gamma: float = 18.0,
        cohesion: float = 0.0,
        phi: float = 30.0,
        method: str = "vesic",
        gamma_sat: Optional[float] = None,
        water_table_depth: float = math.inf,
        factor_of_safety: float = 3.0,
        apply_shape: bool = True,
        apply_depth: bool = True,
        gamma_w: float = 9.81,
    ) -> None:
        if width <= 0.0:
            raise ValueError("width must be positive.")
        if factor_of_safety <= 0.0:
            raise ValueError("factor_of_safety must be positive.")
        self.width = width
        self.length = length
        self.depth = depth
        self.gamma = gamma
        self.gamma_sat = gamma_sat if gamma_sat is not None else gamma
        self.cohesion = cohesion
        self.phi = phi
        self.method = method
        self.water_table_depth = water_table_depth
        self.fs = factor_of_safety
        self.apply_shape = apply_shape
        self.apply_depth = apply_depth
        self.gamma_w = gamma_w

    # ------------------------------------------------------------------
    # Water-table-adjusted stresses
    # ------------------------------------------------------------------
    def _surcharge(self) -> float:
        """Effective overburden :math:`q` at founding depth [kPa]."""
        dw = self.water_table_depth
        gamma_sub = self.gamma_sat - self.gamma_w
        if dw >= self.depth:
            return self.gamma * self.depth
        return self.gamma * dw + gamma_sub * (self.depth - dw)

    def _gamma_effective(self) -> float:
        """Effective unit weight for the ``0.5*gamma*B*Ngamma`` term."""
        dw = self.water_table_depth
        gamma_sub = self.gamma_sat - self.gamma_w
        if dw >= self.depth + self.width:
            return self.gamma
        if dw <= self.depth:
            return gamma_sub
        # Water table within B below the base: linear interpolation.
        return gamma_sub + (dw - self.depth) / self.width * (
            self.gamma - gamma_sub)

    # ------------------------------------------------------------------
    # Correction factors
    # ------------------------------------------------------------------
    def _shape_factors(self, nc: float, nq: float) -> Tuple[float, float, float]:
        if not self.apply_shape or self.length is None:
            return 1.0, 1.0, 1.0
        b_over_l = self.width / self.length
        sc = 1.0 + (nq / nc) * b_over_l
        sq = 1.0 + b_over_l * math.tan(math.radians(self.phi))
        s_gamma = max(1.0 - 0.4 * b_over_l, 0.6)
        return sc, sq, s_gamma

    def _depth_factors(self) -> Tuple[float, float, float]:
        if not self.apply_depth or self.depth <= 0.0:
            return 1.0, 1.0, 1.0
        phi_r = math.radians(self.phi)
        k = self.depth / self.width
        if k > 1.0:
            k = math.atan(k)
        dc = 1.0 + 0.4 * k
        dq = 1.0 + 2.0 * math.tan(phi_r) * (1.0 - math.sin(phi_r)) ** 2 * k
        return dc, dq, 1.0

    # ------------------------------------------------------------------
    # Capacity
    # ------------------------------------------------------------------
    def capacity(self) -> BearingCapacityResult:
        """Compute ultimate, net and allowable bearing capacities."""
        nc, nq, n_gamma = bearing_capacity_factors(self.phi, self.method)
        sc, sq, s_gamma = self._shape_factors(nc, nq)
        dc, dq, d_gamma = self._depth_factors()
        q = self._surcharge()
        gamma_eff = self._gamma_effective()

        q_ult = (self.cohesion * nc * sc * dc
                 + q * nq * sq * dq
                 + 0.5 * gamma_eff * self.width * n_gamma * s_gamma * d_gamma)
        q_net = q_ult - q
        return BearingCapacityResult(
            q_ultimate=q_ult,
            q_net_ultimate=q_net,
            q_allowable_net=q_net / self.fs,
            q_allowable_gross=q_ult / self.fs,
            factor_of_safety=self.fs,
            nc=nc, nq=nq, n_gamma=n_gamma,
        )

    def allowable_load(self) -> float:
        """Allowable *net* column load [kN] from the net capacity."""
        area = self.width * (self.length if self.length is not None else 1.0)
        return self.capacity().q_allowable_net * area

    def __repr__(self) -> str:
        shape = "strip" if self.length is None else f"{self.width}x{self.length} m"
        return (f"ShallowFoundation({shape}, Df={self.depth} m, "
                f"c={self.cohesion} kPa, phi={self.phi} deg, "
                f"method={self.method!r})")
