# Changelog

All notable changes to pyGeotech are documented here. The project follows
[Semantic Versioning](https://semver.org).

## [1.2.0] — 2026-07-16

Advanced limit-equilibrium slope stability.

### Added
- **slope_stability.advanced** — layered soils (:class:`LayeredSoil`,
  :class:`SlopeLayer`), non-circular slip surfaces
  (:class:`PolylineSurface`) and **Spencer's method** (force + moment
  equilibrium via the interslice-resultant formulation), alongside
  Fellenius / Bishop / Janbu. Validated: the layered engine reproduces the
  homogeneous solver exactly, and Spencer matches Bishop within 0.13% for
  a circular surface. New `plot_layered_slope` visualisation.

## [1.1.0] — 2026-07-16

Depth upgrades to the finite-element engine and laterally loaded piles.

### Added
- **fem.ElasticityFEM** — plane-strain / plane-stress linear-elasticity
  kernel on the triangle mesh with **sparse** (`scipy.sparse`) assembly,
  element stresses and von Mises output; validated against the uniaxial-
  bar closed form (uniform `sigma = P/A`, `delta = PL/AE`). New
  `plot_deformed_mesh` visualisation.
- **foundations.api_sand_py** — API RP2A sand p-y curves (Reese C1/C2/C3
  coefficients, `tanh` form) plus `api_sand_modulus`, extending the
  laterally loaded pile solver from clay to sand.
- `scipy>=1.7` added as a dependency.

## [1.0.0] — 2026-07-16

First stable release. Covers the breadth of geotechnical engineering with
validated, code-agnostic mechanics and a from-scratch finite-element
engine, plus multi-code (Eurocode 7 / AASHTO LRFD) design adapters.

### Added
- **standards** — Eurocode 7 partial-factor sets (DA1/DA2/DA3) and AASHTO
  LRFD load/resistance factors, applied over the mechanics cores.
- **fem** — transient 2-D consolidation kernel (validated against
  Terzaghi's 1-D solution) alongside the steady seepage kernel.
- Packaging: `py.typed`, PyPI metadata, ruff/mypy config, CI workflow.

### Module map (all implemented)
`core`, `constants`, `phase_relations` (USCS + AASHTO), `stresses`,
`consolidation`, `shear_strength`, `foundations` (shallow + deep + lateral
p-y), `slope_stability` (+ seismic), `characterization` (SPT/CPT/lab),
`settlement`, `soil_dynamics` (liquefaction + site response),
`rock_mechanics`, `ground_improvement`, `unsaturated`,
`retaining_structures`, `fem`, `standards`, `reliability`.

## [0.9.0] — Phase 4 complete: ground improvement, unsaturated soils.
## [0.8.0] — Phase 4: soil dynamics, rock mechanics.
## [0.7.0] — Phase 3 complete: lateral piles, downdrag, seismic slopes.
## [0.6.0] — Phase 3: slope stability, pile axial capacity.
## [0.5.0] — Phase 2 complete: settlement, lab tests, AASHTO.
## [0.4.0] — Phase 2: site characterization (SPT/CPT).
## [0.3.0] — Phase 1: core layer + custom 2-D FEM engine.
## [0.2.0] — Phase 0: six classical submodules + reliability.
