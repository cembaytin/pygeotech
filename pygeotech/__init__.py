"""pyGeotech: modular, object-oriented soil mechanics toolkit.

Submodules
----------
core
    Shared building blocks: general ``SoilMaterial`` container, base
    exception and the ``DesignStandard`` scaffolding for multi-code
    (Eurocode 7 / AASHTO LRFD / ...) design-factor adapters.
fem
    From-scratch 2-D finite-element engine (linear triangles) solving
    steady-state confined seepage, transient consolidation and plane
    linear elasticity (sparse assembly).
standards
    Multi-code design adapters (Eurocode 7 partial factors, AASHTO LRFD)
    over the code-agnostic mechanics cores.
characterization
    In-situ test interpretation: SPT (N60, (N1)60, correlations) and
    CPTu (Robertson Ic / soil-behaviour type, strength and density);
    laboratory tests (oedometer c_v, Cc/Cr, preconsolidation, compaction).
settlement
    Immediate elastic settlement (Steinbrenner) and semi-empirical
    granular settlement (Schmertmann CPT, Burland-Burbidge SPT).
phase_relations
    Weight-volume (phase) relationships, index properties and
    USCS classification with publication-quality plotting.
stresses
    Geostatic vertical-stress profiles in layered soils and
    load-induced elastic stresses (Boussinesq, Westergaard) with
    pressure-bulb contours.
consolidation
    Terzaghi 1-D consolidation solver, settlement estimation and
    time-settlement / isochrone plots.
shear_strength
    Mohr-Coulomb envelope fitting (direct-shear / triaxial) and
    p-q / p'-q' stress paths.
constitutive
    Advanced constitutive modelling: Modified Cam-Clay critical-state
    model with a drained/undrained triaxial stress-point driver.
foundations
    Shallow-foundation bearing capacity (Terzaghi, Meyerhof, Hansen,
    Vesic) with corrections; deep foundations (pile axial capacity by the
    alpha / beta methods, group efficiency, downdrag) and a
    finite-difference p-y solver for laterally loaded piles.
slope_stability
    Infinite-slope and circular method-of-slices analysis (Fellenius,
    Bishop, Janbu) with a critical-circle search, pseudo-static seismic
    loading and Newmark sliding-block displacement.
soil_dynamics
    Liquefaction triggering (Idriss-Boulanger simplified procedure) and
    1-D site response (natural period, transfer function, Gmax).
rock_mechanics
    Rock-mass classification (RMR, Barton Q, GSI) and the generalized
    Hoek-Brown failure criterion.
ground_improvement
    Vertical drains (Hansbo radial consolidation) and stone columns
    (settlement improvement factors).
unsaturated
    Soil-water characteristic curves (van Genuchten, Fredlund-Xing),
    relative permeability and unsaturated shear strength.
retaining_structures
    Rankine / Coulomb earth pressures and gravity-wall stability.
reliability
    Latin Hypercube / Monte-Carlo helpers for parametric studies.
"""

from pygeotech.constants import GAMMA_W
from pygeotech.constitutive import (
    CamClayParameters,
    MohrCoulombParameters,
    mc_triaxial_test,
    triaxial_test,
)
from pygeotech.core import DesignStandard, PyGeotechError, SoilMaterial
from pygeotech.characterization import CPTLog, SPTLog
from pygeotech.consolidation import (
    Consolidation1D,
    primary_consolidation_settlement,
)
from pygeotech.fem import (
    ConsolidationFEM,
    ElasticityFEM,
    SeepageFEM,
    rectangular_mesh,
)
from pygeotech.foundations import (
    ShallowFoundation,
    bearing_capacity_factors,
    pile_capacity_alpha,
    pile_capacity_beta,
    solve_laterally_loaded_pile,
)
from pygeotech.ground_improvement import (
    priebe_improvement_factor,
    radial_degree_of_consolidation,
)
from pygeotech.phase_relations import (
    Soil,
    USCSResult,
    classify_aashto,
    classify_uscs,
)
from pygeotech.retaining_structures import (
    GravityWall,
    rankine_active_coefficient,
)
from pygeotech.settlement import (
    burland_burbidge_settlement,
    elastic_settlement,
    schmertmann_settlement,
)
from pygeotech.rock_mechanics import (
    hoek_brown_parameters,
    rock_mass_rating,
)
from pygeotech.shear_strength import MohrCoulomb, stress_path_pq
from pygeotech.standards import design_shear_strength, factor_set_for
from pygeotech.slope_stability import (
    infinite_slope_factor,
    newmark_displacement,
    slope_factor_of_safety,
)
from pygeotech.soil_dynamics import (
    liquefaction_factor_of_safety,
    site_natural_period,
)
from pygeotech.unsaturated import (
    unsaturated_shear_strength,
    van_genuchten_water_content,
)
from pygeotech.stresses import (
    SoilLayer,
    SoilProfile,
    StressState,
    boussinesq_point,
    boussinesq_rectangle,
    westergaard_point,
)

__version__ = "1.4.0"

__all__ = [
    "GAMMA_W",
    # core
    "SoilMaterial",
    "DesignStandard",
    "PyGeotechError",
    # characterization
    "SPTLog",
    "CPTLog",
    # phase_relations
    "Soil",
    "USCSResult",
    "classify_uscs",
    "classify_aashto",
    # settlement
    "elastic_settlement",
    "schmertmann_settlement",
    "burland_burbidge_settlement",
    # stresses
    "SoilLayer",
    "SoilProfile",
    "StressState",
    "boussinesq_point",
    "boussinesq_rectangle",
    "westergaard_point",
    # consolidation
    "Consolidation1D",
    "primary_consolidation_settlement",
    # shear_strength
    "MohrCoulomb",
    "stress_path_pq",
    # foundations
    "ShallowFoundation",
    "bearing_capacity_factors",
    "pile_capacity_alpha",
    "pile_capacity_beta",
    "solve_laterally_loaded_pile",
    # slope_stability
    "infinite_slope_factor",
    "slope_factor_of_safety",
    "newmark_displacement",
    # soil_dynamics
    "liquefaction_factor_of_safety",
    "site_natural_period",
    # rock_mechanics
    "rock_mass_rating",
    "hoek_brown_parameters",
    # ground_improvement
    "radial_degree_of_consolidation",
    "priebe_improvement_factor",
    # unsaturated
    "van_genuchten_water_content",
    "unsaturated_shear_strength",
    # retaining_structures
    "GravityWall",
    "rankine_active_coefficient",
    # fem
    "SeepageFEM",
    "ConsolidationFEM",
    "ElasticityFEM",
    "rectangular_mesh",
    # standards (multi-code design adapters)
    "factor_set_for",
    "design_shear_strength",
    # constitutive
    "CamClayParameters",
    "triaxial_test",
    "MohrCoulombParameters",
    "mc_triaxial_test",
    "__version__",
]
