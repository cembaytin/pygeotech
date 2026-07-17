"""Finite-element engine for pyGeotech.

A from-scratch 2-D finite-element framework built on three-node linear
triangles. It solves steady-state confined seepage, transient consolidation and
plane linear elasticity on the same mesh and assembly machinery, with
sparse (``scipy.sparse``) assembly for the elasticity kernel.

Public API
----------
:class:`TriMesh`, :func:`rectangular_mesh`
    Mesh container and structured rectangular generator.
:class:`SeepageFEM`
    Steady 2-D confined-seepage solver.
:class:`ConsolidationFEM`
    Transient 2-D consolidation (diffusion) solver.
:class:`ElasticityFEM`
    Plane-strain / plane-stress linear-elasticity solver (sparse).
Plotting (lazy; needs matplotlib):
    :func:`plot_seepage`.
"""

from typing import List

from pygeotech.fem.consolidation import ConsolidationFEM
from pygeotech.fem.elasticity import ElasticityFEM
from pygeotech.fem.mesh import TriMesh, rectangular_mesh
from pygeotech.fem.seepage import SeepageFEM

__all__: List[str] = [
    "TriMesh",
    "rectangular_mesh",
    "SeepageFEM",
    "ConsolidationFEM",
    "ElasticityFEM",
    "plot_seepage",
    "plot_deformed_mesh",
]

_LAZY = {"plot_seepage", "plot_deformed_mesh"}


def __getattr__(name: str):
    if name in _LAZY:
        from pygeotech.fem import plotting

        return getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
