"""Stresses induced in a semi-infinite elastic half-space by surface loads.

Two classical solutions are provided for the vertical stress increment
:math:`\\Delta\\sigma_z`:

**Boussinesq (1885)** -- homogeneous, isotropic, elastic half-space.
For a vertical point load :math:`Q` at the surface,

.. math::

    \\Delta\\sigma_z = \\frac{3 Q z^3}{2\\pi R^5},
    \\qquad R = \\sqrt{r^2 + z^2}.

**Westergaard (1938)** -- an elastic medium reinforced by rigid, closely
spaced horizontal sheets (a model for thinly stratified/varved soils),

.. math::

    \\Delta\\sigma_z = \\frac{Q}{z^2}\\,
    \\frac{1}{2\\pi}\\,
    \\frac{\\eta}{\\left[\\eta^2 + (r/z)^2\\right]^{3/2}},
    \\qquad
    \\eta^2 = \\frac{1 - 2\\nu}{2 - 2\\nu}.

Westergaard predicts smaller stresses beneath the load than Boussinesq
and is often used as a conservative lower bound for settlement in
layered deposits (``nu = 0`` gives :math:`\\eta^2 = 1/2`).

Distributed loads are obtained by superposition: closed-form results are
given for a uniform rectangle (Newmark's influence factor) and a uniform
circle (under the centre), and any shape can be integrated numerically
with :func:`induced_stress_area`.
"""

from __future__ import annotations

from typing import Union

import numpy as np

__all__ = [
    "boussinesq_point",
    "westergaard_point",
    "influence_factor_rectangle",
    "boussinesq_rectangle",
    "boussinesq_circle_center",
    "induced_stress_area",
]

Number = Union[float, np.ndarray]


def boussinesq_point(load: float, r: Number, z: Number) -> Number:
    """Vertical stress under a surface point load (Boussinesq).

    Parameters
    ----------
    load
        Vertical point load :math:`Q` [kN].
    r
        Horizontal (radial) distance from the load [m].
    z
        Depth below the surface [m], must be > 0.

    Returns
    -------
    float or ndarray
        Vertical stress increment :math:`\\Delta\\sigma_z` [kPa].
    """
    r_arr = np.asarray(r, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    radius = np.sqrt(r_arr ** 2 + z_arr ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = 3.0 * load * z_arr ** 3 / (2.0 * np.pi * radius ** 5)
    return sigma


def westergaard_point(
    load: float, r: Number, z: Number, nu: float = 0.0
) -> Number:
    """Vertical stress under a surface point load (Westergaard).

    Parameters
    ----------
    load
        Vertical point load :math:`Q` [kN].
    r, z
        Radial distance and depth [m]; ``z`` > 0.
    nu
        Poisson's ratio of the soil (0 for the classic rigid-sheet
        idealisation, giving :math:`\\eta^2 = 1/2`). Must be < 0.5.

    Returns
    -------
    float or ndarray
        Vertical stress increment :math:`\\Delta\\sigma_z` [kPa].
    """
    if not 0.0 <= nu < 0.5:
        raise ValueError("Poisson's ratio nu must lie in [0, 0.5).")
    r_arr = np.asarray(r, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    eta2 = (1.0 - 2.0 * nu) / (2.0 - 2.0 * nu)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio2 = (r_arr / z_arr) ** 2
        infl = np.sqrt(eta2) / (2.0 * np.pi * np.power(eta2 + ratio2, 1.5))
        sigma = load / z_arr ** 2 * infl
    return sigma


def influence_factor_rectangle(m: Number, n: Number) -> Number:
    """Newmark's influence factor for a uniformly loaded rectangle.

    Vertical stress under a *corner* of a flexible rectangle (sides
    :math:`B, L`) carrying a uniform pressure :math:`q` is
    :math:`\\Delta\\sigma_z = q\\, I(m, n)` with :math:`m = B/z`,
    :math:`n = L/z` (``m`` and ``n`` are interchangeable):

    .. math::

        I = \\frac{1}{4\\pi}\\left[
        \\frac{2mn\\sqrt{m^2+n^2+1}}{m^2+n^2+1+m^2n^2}
        \\cdot\\frac{m^2+n^2+2}{m^2+n^2+1}
        + \\arctan\\!\\frac{2mn\\sqrt{m^2+n^2+1}}{m^2+n^2+1-m^2n^2}
        \\right].

    The ``arctan`` branch is handled with ``arctan2`` so that the factor
    stays correct when :math:`m^2 n^2 > m^2 + n^2 + 1`.

    Examples
    --------
    >>> round(float(influence_factor_rectangle(1.0, 1.0)), 4)
    0.1752
    """
    m_arr = np.abs(np.asarray(m, dtype=float))
    n_arr = np.abs(np.asarray(n, dtype=float))
    m2 = m_arr ** 2
    n2 = n_arr ** 2
    a = m2 + n2 + 1.0
    y = 2.0 * m_arr * n_arr * np.sqrt(a)
    term1 = (y / (a + m2 * n2)) * ((a + 1.0) / a)
    term2 = np.arctan2(y, a - m2 * n2)
    return (term1 + term2) / (4.0 * np.pi)


def _signed_corner(dx: Number, dy: Number, z: Number) -> Number:
    """Signed corner influence used in the integral-image superposition."""
    dx_arr = np.asarray(dx, dtype=float)
    dy_arr = np.asarray(dy, dtype=float)
    sign = np.sign(dx_arr) * np.sign(dy_arr)
    return sign * influence_factor_rectangle(
        np.abs(dx_arr) / z, np.abs(dy_arr) / z
    )


def boussinesq_rectangle(
    pressure: float,
    width: float,
    length: float,
    z: Number,
    x: Number = None,
    y: Number = None,
) -> Number:
    """Vertical stress under a uniformly loaded rectangle (Boussinesq).

    The rectangle occupies ``0 <= X <= width`` and ``0 <= Y <= length``
    at the surface. The stress is evaluated beneath the point ``(x, y)``
    at depth ``z`` by superposing four signed corner rectangles, which is
    valid whether the point lies inside or outside the loaded area.

    Parameters
    ----------
    pressure
        Uniform contact pressure :math:`q` [kPa].
    width, length
        Plan dimensions :math:`B, L` of the loaded area [m].
    z
        Depth [m], > 0.
    x, y
        Plan coordinates of the point [m]; default to the centre of the
        rectangle (``width/2``, ``length/2``).

    Returns
    -------
    float or ndarray
        Vertical stress increment :math:`\\Delta\\sigma_z` [kPa].
    """
    if x is None:
        x = width / 2.0
    if y is None:
        y = length / 2.0
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    influence = (
        _signed_corner(x_arr, y_arr, z)
        - _signed_corner(x_arr - width, y_arr, z)
        - _signed_corner(x_arr, y_arr - length, z)
        + _signed_corner(x_arr - width, y_arr - length, z)
    )
    return pressure * influence


def boussinesq_circle_center(
    pressure: float, radius: float, z: Number
) -> Number:
    """Vertical stress under the centre of a uniformly loaded circle.

    .. math::

        \\Delta\\sigma_z = q\\left[1 -
        \\frac{1}{\\left(1 + (a/z)^2\\right)^{3/2}}\\right],

    with circle radius :math:`a` and depth :math:`z`.
    """
    z_arr = np.asarray(z, dtype=float)
    return pressure * (1.0 - np.power(1.0 + (radius / z_arr) ** 2, -1.5))


def induced_stress_area(
    pressure: float,
    width: float,
    length: float,
    z: float,
    x: float = None,
    y: float = None,
    method: str = "boussinesq",
    nu: float = 0.0,
    n_cells: int = 80,
) -> float:
    """Vertical stress under a uniformly loaded rectangle by integration.

    The rectangle is discretised into ``n_cells x n_cells`` sub-areas,
    each treated as an equivalent point load ``pressure * dA``. This
    works for any point kernel and is the general route for the
    Westergaard solution and for arbitrary evaluation points.

    Parameters
    ----------
    pressure
        Uniform contact pressure :math:`q` [kPa].
    width, length
        Plan dimensions :math:`B, L` [m].
    z
        Depth [m], > 0.
    x, y
        Plan coordinates of the evaluation point [m]; default centre.
    method
        ``"boussinesq"`` or ``"westergaard"``.
    nu
        Poisson's ratio (Westergaard only).
    n_cells
        Number of integration cells per side (accuracy vs. cost).

    Returns
    -------
    float
        Vertical stress increment :math:`\\Delta\\sigma_z` [kPa].
    """
    if x is None:
        x = width / 2.0
    if y is None:
        y = length / 2.0
    xs = (np.arange(n_cells) + 0.5) / n_cells * width
    ys = (np.arange(n_cells) + 0.5) / n_cells * length
    grid_x, grid_y = np.meshgrid(xs, ys)
    r = np.sqrt((grid_x - x) ** 2 + (grid_y - y) ** 2)
    dA = (width / n_cells) * (length / n_cells)
    point_load = pressure * dA
    if method == "boussinesq":
        contributions = boussinesq_point(point_load, r, z)
    elif method == "westergaard":
        contributions = westergaard_point(point_load, r, z, nu=nu)
    else:
        raise ValueError("method must be 'boussinesq' or 'westergaard'.")
    return float(np.sum(contributions))
