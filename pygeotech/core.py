"""Core layer shared across all pyGeotech domain modules.

This module holds the cross-cutting building blocks that let the library
scale to the full breadth of geotechnical engineering while staying
*design-code agnostic*: a general soil/material container, a base
exception, and the scaffolding for multi-code (Eurocode 7 / AASHTO LRFD /
...) design-factor adapters that wrap the mechanics cores.

The philosophy is a strict layering:

``core``  ->  domain mechanics (stresses, seepage, foundations, ...)
          ->  ``standards`` design-factor adapters
          ->  ``viz`` / ``io`` / ``reliability`` cross-cutting layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

__all__ = [
    "PyGeotechError",
    "DesignStandard",
    "SoilMaterial",
]


class PyGeotechError(Exception):
    """Base class for all pyGeotech-specific errors."""


class DesignStandard(Enum):
    """Supported design-code frameworks for the ``standards`` adapters.

    The mechanics cores return *characteristic* (unfactored) resistances;
    a :class:`DesignStandard` adapter then applies the appropriate partial
    or load-and-resistance factors. This keeps a single physics
    implementation usable under any code (the multi-code strategy).
    """

    CHARACTERISTIC = "characteristic"      # no factors (raw mechanics)
    EUROCODE7_DA1 = "ec7-da1"
    EUROCODE7_DA2 = "ec7-da2"
    EUROCODE7_DA3 = "ec7-da3"
    AASHTO_LRFD = "aashto-lrfd"
    ASD = "asd"                            # allowable stress (global FS)


@dataclass
class SoilMaterial:
    """A general soil/material property container used across modules.

    Every field is optional so the same object can carry whatever a given
    analysis needs (a bearing-capacity check needs ``gamma, phi, cohesion``;
    a seepage analysis needs ``k``; a consolidation analysis needs
    ``cc, cr, cv``). Unit conventions follow :mod:`pygeotech.constants`
    (m, kN, kPa, kN/m^3).

    Parameters
    ----------
    name
        Human-readable identifier.
    gamma, gamma_sat
        Moist and saturated unit weights [kN/m^3].
    phi, cohesion
        Effective friction angle [deg] and cohesion [kPa].
    friction_angle_cu, cohesion_undrained
        Undrained parameters (``phi_u``, ``s_u``) [deg, kPa].
    youngs_modulus, poisson_ratio
        Elastic constants :math:`E` [kPa] and :math:`\\nu` [-].
    permeability
        Isotropic hydraulic conductivity :math:`k` [m/s]; use
        ``permeability_x`` / ``permeability_y`` for anisotropy.
    cc, cr, cv, c_alpha, e0, preconsolidation
        Consolidation parameters (:math:`C_c, C_r, c_v, C_\\alpha, e_0,
        \\sigma'_p`).
    extra
        Free-form dictionary for any additional property.
    """

    name: str = "soil"
    gamma: Optional[float] = None
    gamma_sat: Optional[float] = None
    phi: Optional[float] = None
    cohesion: Optional[float] = None
    friction_angle_cu: Optional[float] = None
    cohesion_undrained: Optional[float] = None
    youngs_modulus: Optional[float] = None
    poisson_ratio: Optional[float] = None
    permeability: Optional[float] = None
    permeability_x: Optional[float] = None
    permeability_y: Optional[float] = None
    cc: Optional[float] = None
    cr: Optional[float] = None
    cv: Optional[float] = None
    c_alpha: Optional[float] = None
    e0: Optional[float] = None
    preconsolidation: Optional[float] = None
    extra: Dict[str, float] = field(default_factory=dict)

    def conductivity(self) -> "tuple[float, float]":
        """Return ``(kx, ky)``, falling back to isotropic ``permeability``.

        Raises
        ------
        PyGeotechError
            If no permeability information has been supplied.
        """
        kx = self.permeability_x if self.permeability_x is not None else self.permeability
        ky = self.permeability_y if self.permeability_y is not None else self.permeability
        if kx is None or ky is None:
            raise PyGeotechError(
                f"material {self.name!r} has no permeability defined.")
        return float(kx), float(ky)

    def require(self, *names: str) -> None:
        """Assert that the named properties are present, else raise.

        Examples
        --------
        >>> SoilMaterial(name="clay", gamma=18.0).require("gamma")
        """
        missing = [n for n in names if getattr(self, n, None) is None]
        if missing:
            raise PyGeotechError(
                f"material {self.name!r} is missing required properties: "
                f"{', '.join(missing)}.")
