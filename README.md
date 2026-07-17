# pyGeotech

A broad, general-purpose, open-source **geotechnical engineering** library
in Python — analytical methods *and* a from-scratch numerical
(finite-element) engine, usable under multiple design codes (Eurocode 7,
AASHTO LRFD, …). It covers the whole of geotechnics with reproducible,
scriptable, validated code instead of black-box software or spreadsheets.

## Installation

```bash
pip install pygeotech
```

Requires Python ≥ 3.9 (NumPy and Matplotlib are the only runtime
dependencies). From source:

```bash
git clone https://github.com/cembaytin/pygeotech
cd pygeotech
pip install -e ".[dev]"
```

## Architecture

Layered by design: `core` → domain mechanics → `standards` adapters →
`viz` / `io` / `reliability`.

| Module | Status | Scope |
|---|---|---|
| `core` | **implemented** | General `SoilMaterial` container, base errors, multi-code `DesignStandard` scaffolding |
| `phase_relations` | **implemented** | Weight-volume identities, USCS (ASTM D2487) classification, Casagrande plasticity chart |
| `stresses` | **implemented** | Layered geostatic profiles, Boussinesq / Westergaard induced stresses, pressure bulbs |
| `consolidation` | **implemented** | Terzaghi 1-D solver, primary/secondary settlement, isochrones & time-settlement curves |
| `shear_strength` | **implemented** | Mohr-Coulomb envelope fitting (direct-shear/triaxial), p-q stress paths, Mohr circles |
| `foundations` | **implemented** | Shallow bearing capacity + deep foundations (pile axial α/β, groups, downdrag, **p-y lateral FD solver**) |
| `slope_stability` | **implemented (v0.6-1.2)** | Method of slices (Fellenius/Bishop/Janbu/**Spencer**), **layered soils**, **non-circular surfaces**, critical-circle search, seismic (pseudo-static + Newmark) |
| `soil_dynamics` | **implemented (v0.8)** | Liquefaction triggering (Idriss-Boulanger SPT/CPT), 1-D site response, Gmax |
| `rock_mechanics` | **implemented (v0.8)** | RMR89, Barton Q, GSI, generalized Hoek-Brown criterion |
| `ground_improvement` | **implemented (v0.9)** | Vertical drains (Hansbo radial consolidation), stone columns (Priebe / equilibrium) |
| `unsaturated` | **implemented (v0.9)** | SWCC (van Genuchten, Fredlund-Xing), relative permeability, unsaturated shear strength |
| `retaining_structures` | **implemented** | Rankine/Coulomb earth pressures, gravity-wall stability |
| `fem` | **implemented (v0.3, v1.0)** | Custom 2-D linear-triangle engine; validated confined-seepage **and transient consolidation** kernels |
| `characterization` | **implemented (v0.4-0.5)** | SPT & CPTu interpretation (`SPTLog`/`CPTLog`) + laboratory tests (oedometer c_v, Cc/Cr, σ'p, Proctor) |
| `settlement` | **implemented (v0.5)** | Elastic (Steinbrenner), Schmertmann (CPT) and Burland-Burbidge (SPT) |
| `standards` | **implemented (v1.0)** | Eurocode 7 partial factors (DA1/2/3) and AASHTO LRFD load/resistance factors |
| `reliability` | **implemented** | Latin Hypercube / Monte-Carlo sampling and propagation for parametric studies |

Also: USCS **and AASHTO** soil classification in `phase_relations`.
Current release **v1.2.0** (stable) — the whole roadmap is implemented,
plus a plane-elasticity FEM kernel (sparse), API sand p-y curves and
advanced slope stability (layered, non-circular, Spencer);
**291 unit tests + doctests** passing, every submodule with
publication-quality plots.

## Quick start

```python
from pygeotech.phase_relations import Soil, classify_uscs

# Solve the full phase state from any consistent subset of inputs.
soil = Soil(w=0.20, gs=2.70, e=0.80)
print(soil.sr, soil.gamma_d, soil.gamma_sat)   # 0.675 14.715 19.075
print(soil.report())                           # includes derivation trace

# USCS classification (ASTM D2487).
res = classify_uscs(60.0, 90.0, liquid_limit=40.0, plasticity_index=22.0)
print(res)   # CL: Sandy lean clay
```

```python
from pygeotech.phase_relations.plotting import plot_plasticity_chart

plot_plasticity_chart(
    [(40.0, 22.0, "BH1-S1"), (60.0, 35.0, "BH3-S1")],
    save_path="plasticity_chart.pdf",   # vector, serif, 600 dpi
)
```

```python
from pygeotech.stresses import SoilLayer, SoilProfile, boussinesq_rectangle

# Geostatic stresses in a layered profile with a water table.
profile = SoilProfile(
    [SoilLayer(2.0, gamma=16.5),
     SoilLayer(4.0, gamma=18.0, gamma_sat=19.5),
     SoilLayer(6.0, gamma=17.0, gamma_sat=18.5)],
    water_table_depth=2.0, surcharge=10.0,
)
print(profile.effective_stress(6.0))            # 81.76 kPa

# Vertical stress under the centre of a 2x3 m footing at 2 m depth.
print(float(boussinesq_rectangle(150.0, 2.0, 3.0, 2.0)))   # 64.2 kPa
```

```python
from pygeotech.stresses.plotting import plot_geostatic_profile, plot_pressure_bulb

plot_geostatic_profile(profile, save_path="geostatic.pdf")
plot_pressure_bulb(150.0, width=2.0, method="boussinesq",
                   save_path="bulb.pdf")   # Δσz/q isobars
```

```python
# Consolidation: Terzaghi 1-D solver + settlement.
from pygeotech.consolidation import (Consolidation1D,
                                     primary_consolidation_settlement)

clay = Consolidation1D(cv=1.5, layer_thickness=6.0, drainage="double")
print(clay.time_for_degree(0.90))                       # 5.09 (yr)
print(primary_consolidation_settlement(6.0, 1.1, 80.0, 60.0,
                                       cc=0.35, cr=0.06, sigma_p=120.0))
```

```python
# Shear strength: fit a Mohr-Coulomb envelope from triaxial tests.
from pygeotech.shear_strength import MohrCoulomb

env = MohrCoulomb.fit_triaxial(sigma3=[50, 100, 150, 200],
                               sigma1=[180, 300, 415, 530])
print(env)                                              # c, phi

# Foundations: bearing capacity of a footing (with water table).
from pygeotech.foundations import ShallowFoundation

f = ShallowFoundation(width=2.5, length=2.5, depth=1.5, phi=32.0,
                      cohesion=5.0, method="vesic", water_table_depth=2.0)
print(f.capacity())                                     # q_ult, q_allow ...

# Retaining walls: Rankine thrust + gravity-wall stability.
from pygeotech.retaining_structures import active_thrust, GravityWall

P, arm = active_thrust(6.0, gamma=18.0, phi=30.0, surcharge=10.0)
wall = GravityWall(6.0, base_width=3.5, weight=420.0, weight_arm=1.8,
                   base_friction_angle=24.0)
print(wall.check(P, arm))                               # FS_sliding, ...
```

### Parametric / stochastic studies

The `reliability` helpers sweep any core function over Latin Hypercube or
Monte-Carlo samples — the typical driver for research-grade fragility and
sensitivity analyses:

```python
from pygeotech.reliability import latin_hypercube, propagate, summarize
from pygeotech.foundations import ShallowFoundation

def q_allow(phi, cohesion):
    return ShallowFoundation(width=2.0, depth=1.0, phi=phi,
                             cohesion=cohesion).capacity().q_allowable_net

samples = latin_hypercube(2000, {"phi": (28, 34), "cohesion": (0, 15)}, seed=0)
print(summarize(propagate(q_allow, samples)))   # mean, std, COV, P05, P95 ...
```

### Site characterization (in-situ tests)

Interpret SPT and CPTu soundings; the SPT log integrates directly with a
`SoilProfile` to get the effective stress for overburden correction:

```python
from pygeotech.characterization import SPTLog, CPTLog
from pygeotech.stresses import SoilLayer, SoilProfile

profile = SoilProfile([SoilLayer(15.0, gamma=18.0, gamma_sat=20.0)],
                      water_table_depth=2.0)
spt = SPTLog(depth=[3, 6, 9], n_field=[12, 19, 26], profile=profile)
spt.n1_60(); spt.friction_angle(); spt.relative_density()

# CPTu: Robertson Ic / soil-behaviour type; unit weight auto-estimated.
cpt = CPTLog(depth, qc, fs, u2=u2, water_table_depth=1.5, nkt=14)
res = cpt.process()      # res.ic, res.sbt_zone, res.undrained_strength, ...
```

### Finite-element engine (numerical)

A from-scratch 2-D FEM (linear triangles). The first kernel solves
steady-state confined seepage on homogeneous or heterogeneous, isotropic
or anisotropic domains:

```python
import numpy as np
from pygeotech.fem import SeepageFEM, rectangular_mesh

mesh = rectangular_mesh(width=20.0, height=8.0, nx=80, ny=32)
k = np.full(mesh.n_elements, 1e-4)              # base soil
# ... set a low-permeability clay lens on selected elements: k[mask] = 1e-6

fem = SeepageFEM(mesh, conductivity=k)
fem.set_head(mesh.nodes_where("x", 0.0), 8.0)   # upstream head
fem.set_head(mesh.nodes_where("x", 20.0), 2.0)  # downstream head
head = fem.solve()

Q = fem.boundary_flow(mesh.nodes_where("x", 0.0))   # seepage flow rate
v = fem.velocities()                                # Darcy velocity per element

from pygeotech.fem.plotting import plot_seepage
plot_seepage(fem, save_path="flow_net.pdf")         # equipotentials + flow
```

## Conventions

- Unit weights in kN/m³ (γ_w = 9.81 kN/m³ by default); `w` and `Sr` are
  decimal ratios; sieve data in percent (0–100).
- All figures use a centralized academic style
  (`pygeotech.plot_style`): serif/Times typography, inward ticks,
  600 dpi, Type-42 embedded fonts for editable vector PDFs.

## Design under a code (Eurocode 7 / AASHTO)

The mechanics return *characteristic* values; the `standards` adapters
apply the code factors:

```python
from pygeotech.core import DesignStandard
from pygeotech.standards import factor_set_for, design_shear_strength
from pygeotech.foundations import ShallowFoundation

fs = factor_set_for(DesignStandard.EUROCODE7_DA3)
phi_d, c_d, _ = design_shear_strength(fs, friction_angle=32, cohesion=5)
q_ult = ShallowFoundation(width=2.5, depth=1.5, phi=phi_d,
                          cohesion=c_d).capacity().q_ultimate
```

## Development

```bash
pip install -e ".[dev]"
pytest            # 266 unit tests + doctests
ruff check .
mypy pygeotech
```

## Building & publishing a release

The distribution builds and passes `twine check`:

```bash
python -m build                 # -> dist/pygeotech-<ver>.whl + .tar.gz
twine check dist/*
twine upload dist/*             # requires your PyPI account/token
```

Once uploaded, anyone can `pip install pygeotech`.

