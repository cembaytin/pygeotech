"""End-to-end demo touching every pyGeotech submodule.

Run from the repository root:

    python examples/demo_full_library.py
"""

import numpy as np

from pygeotech.consolidation import (
    Consolidation1D,
    primary_consolidation_settlement,
)
from pygeotech.consolidation.plotting import (
    plot_degree_vs_time_factor,
    plot_isochrones,
    plot_time_settlement,
)
from pygeotech.foundations import ShallowFoundation
from pygeotech.foundations.plotting import plot_bearing_capacity_factors
from pygeotech.reliability import latin_hypercube, propagate, summarize
from pygeotech.retaining_structures import GravityWall, active_thrust
from pygeotech.retaining_structures.plotting import plot_active_pressure_diagram
from pygeotech.shear_strength import MohrCoulomb, principal_stresses_at_failure
from pygeotech.shear_strength.plotting import plot_mohr_circles, plot_stress_path

OUT = "examples/output"


def consolidation_demo() -> None:
    print("== Consolidation ==")
    clay = Consolidation1D(cv=1.5, layer_thickness=6.0, drainage="double")
    sc = primary_consolidation_settlement(
        thickness=6.0, void_ratio=1.1, sigma0=80.0, delta_sigma=60.0,
        cc=0.35, cr=0.06, sigma_p=120.0)
    print(f"  primary settlement s_c = {sc * 1000:.1f} mm")
    print(f"  t(U=90%) = {clay.time_for_degree(0.90):.2f} yr")
    plot_isochrones(save_path=f"{OUT}/consol_isochrones.png")
    plot_degree_vs_time_factor(save_path=f"{OUT}/consol_U_Tv.png")
    plot_time_settlement(clay, sc * 1000.0,
                         save_path=f"{OUT}/consol_time_settlement.png")
    print(f"  figures -> {OUT}/consol_*.png\n")


def shear_demo() -> None:
    print("== Shear strength ==")
    env_true = MohrCoulomb(cohesion=18.0, friction_angle=27.0)
    sigma3 = np.array([50.0, 100.0, 150.0, 200.0])
    sigma1 = np.array([principal_stresses_at_failure(env_true, s) for s in sigma3])
    fitted = MohrCoulomb.fit_triaxial(sigma3, sigma1)
    print(f"  fitted envelope: {fitted}")
    plot_mohr_circles(sigma3, sigma1, fitted,
                      save_path=f"{OUT}/shear_mohr.png")

    # A CU-type effective stress path (increasing sigma1, constant sigma3).
    s1 = np.linspace(100.0, 260.0, 9)
    s3 = np.full_like(s1, 100.0)
    u = np.linspace(0.0, 55.0, 9)
    from pygeotech.shear_strength import stress_path_pq
    p, q, p_eff = stress_path_pq(s1, s3, u)
    plot_stress_path(p, q, p_eff, envelope=fitted,
                     save_path=f"{OUT}/shear_stress_path.png")
    print(f"  figures -> {OUT}/shear_*.png\n")


def foundation_demo() -> None:
    print("== Foundations ==")
    footing = ShallowFoundation(width=2.5, length=2.5, depth=1.5, gamma=18.0,
                                gamma_sat=20.0, cohesion=5.0, phi=32.0,
                                method="vesic", water_table_depth=2.0)
    res = footing.capacity()
    print(f"  {footing}")
    print(f"  {res}")
    print(f"  allowable column load = {footing.allowable_load():.0f} kN")
    plot_bearing_capacity_factors(save_path=f"{OUT}/foundation_factors.png")
    print(f"  figure -> {OUT}/foundation_factors.png\n")


def retaining_demo() -> None:
    print("== Retaining structures ==")
    thrust, line = active_thrust(6.0, gamma=18.0, phi=30.0, surcharge=10.0)
    print(f"  active thrust = {thrust:.1f} kN/m at {line:.2f} m above base")
    wall = GravityWall(height=6.0, base_width=3.5, weight=420.0,
                       weight_arm=1.8, base_friction_angle=24.0)
    print(f"  {wall.check(thrust, line)}")
    plot_active_pressure_diagram(6.0, 18.0, 30.0, surcharge=10.0,
                                 water_table_depth=3.0,
                                 save_path=f"{OUT}/retaining_pressure.png")
    print(f"  figure -> {OUT}/retaining_pressure.png\n")


def reliability_demo() -> None:
    print("== Reliability (parametric sweep) ==")

    def q_all(phi: float, cohesion: float) -> float:
        return ShallowFoundation(width=2.0, depth=1.0, phi=phi,
                                 cohesion=cohesion).capacity().q_allowable_net

    samples = latin_hypercube(2000, {"phi": (28.0, 34.0),
                                     "cohesion": (0.0, 15.0)}, seed=0)
    out = propagate(q_all, samples)
    stats = summarize(out)
    print(f"  q_allow,net over LHS(phi, c): mean={stats['mean']:.1f} kPa, "
          f"COV={stats['cov']:.2f}, "
          f"P05={stats['p05']:.1f}, P95={stats['p95']:.1f}\n")


def main() -> None:
    consolidation_demo()
    shear_demo()
    foundation_demo()
    retaining_demo()
    reliability_demo()


if __name__ == "__main__":
    main()
