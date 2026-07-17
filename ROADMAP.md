# pyGeotech — Development Roadmap

**Mission:** a broad, general-purpose, community-usable library covering
**the whole of geotechnical engineering** — analytical methods *and* a
from-scratch numerical (finite-element) engine, usable under **multiple
design codes** (Eurocode 7, AASHTO LRFD, …). This is a multi-release
program delivered in tested increments, not a single drop.

Each submodule is decoupled (separation of concerns): a **computation
core** (pure functions / dataclasses, no plotting dependency), an
optional **plotting layer** on the shared `pygeotech.plot_style`, and —
where a design code applies — a thin **standards adapter** over the same
mechanics.

Design invariants across the whole library:

- Layered architecture: `core` → domain mechanics → `standards` adapters
  → `viz` / `io` / `reliability`.
- Consistent SI units (m, kN, kPa, kN/m³) declared in `pygeotech.constants`.
- Full type hints (Python 3.9 compatible), PEP 8, NumPy-style docstrings
  carrying the governing equations.
- **Trust through validation:** every method checked against published
  benchmarks / textbook examples (this is what makes a general library
  usable by everyone).
- Vectorised NumPy cores; every figure vector-ready (PDF), 600 dpi.

## Delivery phases

- **Phase 0 — foundations & analytical core** ✅ (v0.1–0.2): the six
  classical submodules + reliability.
- **Phase 1 — numerical spine** ◐ (v0.3, current): `core` layer +
  custom 2-D FEM engine; first kernel = confined seepage. *(done)*
- **Phase 2 — site & settlement pillars** ✅ (v0.4-0.5): `characterization`
  (SPT/CPTu + laboratory), `settlement` (elastic + Schmertmann +
  Burland-Burbidge), AASHTO classification. *(analytical unconfined-seepage
  helpers folded into a later FEM extension.)*
- **Phase 3 — stability & deep foundations** ✅ (v0.6-0.7):
  `slope_stability` (infinite + method of slices Fellenius/Bishop/Janbu +
  critical-circle search + pseudo-static seismic + Newmark displacement);
  deep foundations (pile axial alpha/beta, group efficiency, downdrag, and
  a finite-difference p-y solver for laterally loaded piles). *(Spencer/M-P
  and Broms remain optional refinements.)*
- **Phase 4 — dynamics, rock, improvement** ✅ (v0.8-0.9): `soil_dynamics`
  (liquefaction triggering + site response), `rock_mechanics`
  (RMR/Q/GSI + Hoek-Brown), `ground_improvement` (vertical drains + stone
  columns) and `unsaturated` (SWCC + suction strength).
- **Phase 5 — FEM breadth & codes** ✅ (v1.0): extended the FEM engine to
  transient consolidation; added `standards` (EC7 DA1/2/3 + AASHTO LRFD);
  packaging (`py.typed`, PyPI metadata, wheel/sdist), CI (ruff/mypy/pytest).
  *(Plane-elasticity FEM and `io`/AGS import remain as post-1.0 additions.)*

## Status

| # | Submodule | Status | Core deliverables |
|---|-----------|--------|-------------------|
| 1 | `phase_relations` | ✅ done | `Soil` phase solver · USCS (ASTM D2487) · plasticity chart |
| 2 | `stresses` | ✅ done | `SoilProfile` geostatic profiles · Boussinesq/Westergaard · pressure bulbs |
| 3 | `consolidation` | ✅ done | Terzaghi 1-D solver · settlement · isochrones / time–settlement / U–Tv |
| 4 | `shear_strength` | ✅ done | Mohr–Coulomb envelope fitting · p–q stress paths · Mohr circles |
| 5 | `foundations` | ✅ done | Bearing capacity (Terzaghi, Meyerhof, Hansen, Vesić) · shape/depth/WT |
| 6 | `retaining_structures` | ✅ done | Rankine/Coulomb earth pressures · gravity-wall stability |
| — | `reliability` | ✅ done | LHS / Monte-Carlo sampling · propagation · summary statistics |
| — | `core` | ✅ done | `SoilMaterial` container · `PyGeotechError` · `DesignStandard` scaffolding |
| — | `fem` | ◐ v0.3 | Custom 2-D linear-triangle engine · **seepage kernel done** (Laplace) · consolidation/elasticity next |
| — | `characterization` | ✅ v0.4-0.5 | SPT + CPTu (`SPTLog`/`CPTLog`) · laboratory (oedometer c_v Casagrande/Taylor, Cc/Cr, σ'p, Proctor) |
| 1b | AASHTO classification | ✅ v0.5 | AASHTO M145 groups A-1…A-7 + group index (alongside USCS) |
| — | `settlement` | ✅ v0.5 | elastic (Steinbrenner) · Schmertmann (CPT) · Burland–Burbidge (SPT) |
| — | `slope_stability` | ✅ v0.6-0.7 | infinite slope · method of slices (Fellenius/Bishop/Janbu) · critical-circle search · pseudo-static seismic · Newmark displacement |
| — | deep foundations | ✅ v0.6-0.7 | piles axial (α/β) · group efficiency · downdrag · **p-y FD solver** for lateral loading (Hetenyi-validated) |
| — | `soil_dynamics` | ✅ v0.8 | liquefaction triggering (Idriss-Boulanger SPT/CPT) · 1-D site response · Gmax |
| — | `rock_mechanics` | ✅ v0.8 | RMR89 · Barton Q · GSI · Hoek–Brown (rock-slope kinematics next) |
| — | `ground_improvement` | ✅ v0.9 | Hansbo radial consolidation (PVD, smear) · stone columns (equilibrium + Priebe) |
| — | `unsaturated` | ✅ v0.9 | SWCC (van Genuchten, Fredlund–Xing) · rel. permeability · unsat. shear strength |
| — | `fem` (consolidation) | ✅ v1.0 | transient diffusion kernel on the TriMesh engine (Terzaghi-validated) |
| — | `standards` | ✅ v1.0 | EC7 DA1/2/3 partial factors · AASHTO LRFD load/resistance factors |
| — | packaging | ✅ v1.0 | `py.typed` · PyPI metadata · wheel/sdist · CI (ruff/mypy/pytest) |

**v0.3.0** — Phase 1 numerical spine landed: `core` + custom FEM engine
with a validated confined-seepage kernel.

**v0.4.0** — Phase 2 started: `characterization` submodule (SPT & CPTu
interpretation) integrated with `stresses`.

**v0.5.0** — Phase 2 complete: laboratory-test interpretation, AASHTO
classification and the `settlement` submodule (elastic + Schmertmann +
Burland-Burbidge).

**v0.6.0** — Phase 3 (main pillars): `slope_stability` (infinite +
method of slices + critical-circle search) and deep-foundation pile axial
capacity (alpha/beta) with group efficiency.

**v0.7.0** — Phase 3 complete: laterally loaded pile p-y finite-difference
solver (validated against Hetenyi's closed form), pile downdrag, and slope
pseudo-static seismic loading + Newmark sliding-block displacement.

**v0.8.0** — Phase 4 (dynamics & rock): `soil_dynamics` (Idriss-Boulanger
liquefaction triggering + 1-D site response) and `rock_mechanics`
(RMR/Q/GSI + Hoek-Brown).

**v0.9.0** — Phase 4 complete: `ground_improvement` (Hansbo radial
consolidation + stone-column improvement factors) and `unsaturated` (SWCC,
relative permeability, unsaturated shear strength). Every submodule ships
publication-quality plots.

**v1.0.0** — First stable release. Phase 5 done: transient FEM
consolidation, EC7/AASHTO `standards` adapters, full packaging
(`pip install pygeotech`), CI. **266 unit tests + doctests** passing;
wheel + sdist build and pass `twine check`; verified by a clean-venv
install and smoke test.

## Submodule plans

### 3. `consolidation` (done) — future extensions
- Crank–Nicolson finite-difference option for layered `c_v` / arbitrary
  drainage and time-dependent loading.
- Casagrande (log-t) and Taylor (√t) fitting helpers to back out `c_v`
  from oedometer data.

### 4. `shear_strength` (done) — future extensions
- Cambridge `p'-q` (`q = σ1-σ3`) convention alongside the MIT one.
- Critical-state parameters (`M`, `Γ`, `λ`, `κ`) and NCL/CSL plotting.

### 5. `foundations` (done) — future extensions
- Load inclination, base and ground-slope factors; eccentric loading via
  the effective-area method.
- Settlement coupling to `stresses` (elastic) and `consolidation`.
- Deep foundations (pile axial capacity).

### 6. `retaining_structures` (done) — future extensions
- `CantileverWall` with stem/base proportioning and internal stresses.
- Log-spiral / trial-wedge passive resistance for large `δ`.

### Cross-cutting infrastructure
- `reliability` (done): LHS / Monte-Carlo sampling + propagation; next add
  FORM/SORM reliability index `β` and Sobol sensitivity indices.
- Packaging: `pip install pygeotech`, versioned releases (currently
  editable install via `PYTHONPATH`).
- CI: `pytest` (90 tests) is in place; add `ruff`/`flake8` + `mypy` and a
  GitHub Actions workflow; docs via Sphinx/MkDocs.
