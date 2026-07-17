"""Geostatic (in-situ) vertical stress profiles in layered soils.

Physical background
-------------------
For a horizontally layered deposit the *total* vertical stress at depth
``z`` is the weight of everything above a unit area,

.. math::

    \\sigma_v(z) = q_0 + \\int_0^{z} \\gamma(\\zeta)\\, d\\zeta ,

where :math:`q_0` is a uniform surface surcharge and the unit weight
:math:`\\gamma` switches from the *moist* value :math:`\\gamma` above the
zone of saturation to the *saturated* value :math:`\\gamma_{sat}` below
it.  The hydrostatic pore-water pressure referenced to the water table
at depth :math:`z_w` is

.. math::

    u(z) = \\gamma_w \\, (z - z_w) ,

which is positive below the water table and (in a capillary fringe of
height :math:`h_c` above it) negative down to :math:`z_w - h_c`.  Terzaghi's
effective-stress principle then gives

.. math::

    \\sigma'_v(z) = \\sigma_v(z) - u(z).

Sign convention: compression positive, depth ``z`` measured downward
from the ground surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from pygeotech.constants import GAMMA_W

__all__ = ["SoilLayer", "StressState", "SoilProfile"]


@dataclass
class SoilLayer:
    """A single homogeneous soil stratum.

    Parameters
    ----------
    thickness
        Layer thickness [m] (must be positive).
    gamma
        Moist (bulk) unit weight above the zone of saturation [kN/m^3].
    gamma_sat
        Saturated unit weight [kN/m^3]; defaults to ``gamma`` when the
        layer is never submerged.
    name
        Optional label used in tables and plots.
    """

    thickness: float
    gamma: float
    gamma_sat: Optional[float] = None
    name: str = ""

    def __post_init__(self) -> None:
        if self.thickness <= 0.0:
            raise ValueError("layer thickness must be positive.")
        if self.gamma <= 0.0:
            raise ValueError("unit weight gamma must be positive.")
        if self.gamma_sat is None:
            self.gamma_sat = self.gamma
        elif self.gamma_sat < self.gamma - 1e-9:
            raise ValueError(
                f"gamma_sat ({self.gamma_sat}) cannot be smaller than "
                f"gamma ({self.gamma}) kN/m^3."
            )


@dataclass(frozen=True)
class StressState:
    """Vertical stress state at one depth (all values in kPa)."""

    depth: float
    total_vertical: float
    pore_pressure: float
    effective_vertical: float


class SoilProfile:
    """A stack of :class:`SoilLayer` with a water table and surcharge.

    Parameters
    ----------
    layers
        Layers ordered from the ground surface downward. The deepest
        layer is treated as extending indefinitely, so stresses can be
        queried below the defined stack.
    water_table_depth
        Depth to the phreatic surface [m] (0 = at ground surface,
        ``inf`` = dry profile, the default).
    surcharge
        Uniform vertical surcharge applied at the surface [kPa].
    capillary_rise
        Height of the capillary fringe above the water table [m]. Within
        it the soil is taken as saturated (uses ``gamma_sat``) and the
        pore pressure is negative (suction). Default 0.
    gamma_w
        Unit weight of water [kN/m^3].

    Examples
    --------
    >>> profile = SoilProfile(
    ...     [SoilLayer(2.0, gamma=16.0),
    ...      SoilLayer(4.0, gamma=18.0, gamma_sat=19.0),
    ...      SoilLayer(6.0, gamma=17.0, gamma_sat=18.0)],
    ...     water_table_depth=2.0,
    ... )
    >>> round(profile.effective_stress(6.0), 2)
    68.76
    """

    def __init__(
        self,
        layers: Sequence[SoilLayer],
        water_table_depth: float = math.inf,
        surcharge: float = 0.0,
        capillary_rise: float = 0.0,
        gamma_w: float = GAMMA_W,
    ) -> None:
        if not layers:
            raise ValueError("a soil profile needs at least one layer.")
        if water_table_depth < 0.0:
            raise ValueError("water_table_depth cannot be negative.")
        if capillary_rise < 0.0:
            raise ValueError("capillary_rise cannot be negative.")
        self.layers: List[SoilLayer] = list(layers)
        self.water_table_depth: float = water_table_depth
        self.surcharge: float = surcharge
        self.capillary_rise: float = capillary_rise
        self.gamma_w: float = gamma_w
        # Cumulative depths of the layer interfaces (top of each layer).
        self._tops: List[float] = [0.0]
        for layer in self.layers:
            self._tops.append(self._tops[-1] + layer.thickness)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    @property
    def total_thickness(self) -> float:
        """Combined thickness of the defined layers [m]."""
        return self._tops[-1]

    @property
    def _saturation_depth(self) -> float:
        """Depth below which soil is saturated (top of capillary fringe)."""
        if math.isinf(self.water_table_depth):
            return math.inf
        return self.water_table_depth - self.capillary_rise

    def layer_at(self, z: float) -> SoilLayer:
        """Return the layer containing depth ``z`` (last layer if below)."""
        for i, layer in enumerate(self.layers):
            if z < self._tops[i + 1] or i == len(self.layers) - 1:
                return layer
        return self.layers[-1]

    # ------------------------------------------------------------------
    # Stress components
    # ------------------------------------------------------------------
    def total_stress(self, z: float) -> float:
        """Total vertical stress :math:`\\sigma_v` at depth ``z`` [kPa]."""
        if z < 0.0:
            raise ValueError("depth z cannot be negative.")
        z_sat = self._saturation_depth
        sigma = self.surcharge
        for i, layer in enumerate(self.layers):
            top = self._tops[i]
            if z <= top:
                break
            bottom = self._tops[i + 1] if i < len(self.layers) - 1 else math.inf
            seg_bot = min(z, bottom)
            sigma += self._segment_weight(top, seg_bot, z_sat, layer)
            if seg_bot >= z:
                break
        return sigma

    @staticmethod
    def _segment_weight(
        a: float, b: float, z_sat: float, layer: SoilLayer
    ) -> float:
        """Weight per unit area of a layer segment [a, b], split at z_sat."""
        gamma_sat = layer.gamma_sat if layer.gamma_sat is not None else layer.gamma
        if b <= z_sat:
            return layer.gamma * (b - a)
        if a >= z_sat:
            return gamma_sat * (b - a)
        return layer.gamma * (z_sat - a) + gamma_sat * (b - z_sat)

    def pore_pressure(self, z: float) -> float:
        """Hydrostatic pore-water pressure :math:`u` at depth ``z`` [kPa]."""
        if z < 0.0:
            raise ValueError("depth z cannot be negative.")
        zw = self.water_table_depth
        if math.isinf(zw):
            return 0.0
        if z >= self._saturation_depth:
            return self.gamma_w * (z - zw)
        return 0.0

    def effective_stress(self, z: float) -> float:
        """Effective vertical stress :math:`\\sigma'_v` at depth ``z`` [kPa]."""
        return self.total_stress(z) - self.pore_pressure(z)

    def state(self, z: float) -> StressState:
        """Full :class:`StressState` at depth ``z``."""
        sigma = self.total_stress(z)
        u = self.pore_pressure(z)
        return StressState(z, sigma, u, sigma - u)

    # ------------------------------------------------------------------
    # Profiles for plotting / export
    # ------------------------------------------------------------------
    def profile(
        self, zmax: Optional[float] = None, dz: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample the stress profile on a fine depth grid.

        Parameters
        ----------
        zmax
            Maximum depth [m]; defaults to the profile thickness.
        dz
            Sampling interval [m].

        Returns
        -------
        (depth, sigma_v, u, sigma_v_eff)
            Four aligned 1-D arrays. Break points (layer interfaces, the
            water table and the top of the capillary fringe) are inserted
            so the piecewise-linear curves render with sharp kinks.
        """
        if zmax is None:
            zmax = self.total_thickness
        grid = list(np.arange(0.0, zmax + 1e-9, dz))
        breaks = list(self._tops)
        if not math.isinf(self.water_table_depth):
            breaks.extend([self.water_table_depth, self._saturation_depth])
        for b in breaks:
            if 0.0 <= b <= zmax:
                grid.extend([b - 1e-6, b, b + 1e-6])
        depths = np.array(sorted(d for d in grid if 0.0 <= d <= zmax))
        sigma = np.array([self.total_stress(d) for d in depths])
        u = np.array([self.pore_pressure(d) for d in depths])
        return depths, sigma, u, sigma - u

    def summary_table(self) -> str:
        """Return a text table of stresses at every layer interface."""
        header = (
            f"{'depth [m]':>10s}{'sigma_v':>12s}"
            f"{'u':>10s}{'eff_sigma_v':>14s}"
        )
        lines = [header, "-" * 46]
        for top in self._tops:
            s = self.state(top)
            lines.append(
                f"{s.depth:>10.2f}{s.total_vertical:>12.2f}"
                f"{s.pore_pressure:>10.2f}{s.effective_vertical:>14.2f}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        wt = ("dry" if math.isinf(self.water_table_depth)
              else f"{self.water_table_depth:g} m")
        return (f"SoilProfile({len(self.layers)} layers, "
                f"H={self.total_thickness:g} m, WT={wt})")
