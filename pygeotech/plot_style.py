"""Centralized academic plotting style for all pyGeotech figures.

Every plotting routine in the library draws inside :func:`academic_style`
so that figures are consistent, vector-ready (PDF, Type-42 embedded fonts)
and typeset with a serif (Times-like) font suitable for journal
submission (e.g. Elsevier/ASCE single- or double-column layouts).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, Optional

import matplotlib as mpl

#: rcParams applied to every pyGeotech figure.
ACADEMIC_RC: Dict[str, object] = {
    # Typography: Times New Roman with safe serif fallbacks; STIX math.
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "STIXGeneral",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",
    "font.size": 10.0,
    "axes.labelsize": 11.0,
    "axes.titlesize": 11.0,
    "legend.fontsize": 9.0,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    # Axes/tick appearance (journal style: inward ticks on all sides).
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": False,
    # Output quality: 600 dpi raster, tight bounding box, editable
    # (Type-42 / TrueType) fonts inside PDF and PS files.
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


@contextmanager
def academic_style(overrides: Optional[Dict[str, object]] = None) -> Iterator[None]:
    """Context manager that applies the pyGeotech academic style locally.

    Parameters
    ----------
    overrides
        Optional rcParams overriding :data:`ACADEMIC_RC` inside the context.

    Examples
    --------
    >>> from pygeotech.plot_style import academic_style
    >>> with academic_style():
    ...     pass  # build matplotlib figures here
    """
    rc = dict(ACADEMIC_RC)
    if overrides:
        rc.update(overrides)
    with mpl.rc_context(rc):
        yield


def use_academic_style() -> None:
    """Apply the academic style globally (mutates ``matplotlib.rcParams``)."""
    mpl.rcParams.update(ACADEMIC_RC)
