"""Shared physical constants and unit conventions for pyGeotech.

All modules use a consistent set of units:

============  ===================================
Quantity      Unit
============  ===================================
Length        metre (m)
Force         kilonewton (kN)
Stress        kilopascal (kPa = kN/m^2)
Unit weight   kN/m^3
Angle         degree at the public API, radian internally
============  ===================================
"""

from __future__ import annotations

#: Unit weight of water at ~4 degC [kN/m^3].
GAMMA_W: float = 9.81

#: Standard gravitational acceleration [m/s^2].
G: float = 9.81

#: Atmospheric reference pressure [kPa].
P_ATM: float = 101.325

__all__ = ["GAMMA_W", "G", "P_ATM"]
