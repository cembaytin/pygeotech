"""Shallow-foundation design calculator built on pyGeotech.

A self-contained command-line tool that combines several pyGeotech modules
(foundations + settlement + stresses + standards) into one design check.

Run it (from anywhere, once pygeotech is installed):

    python3 examples/foundation_design_tool.py
    python3 examples/foundation_design_tool.py --width 3 --length 4 --phi 34 \
        --load 3500 --code EC7-DA2

It prints an engineering report and saves a pressure-bulb figure.
"""

from __future__ import annotations

import argparse
import math

from pygeotech.core import DesignStandard
from pygeotech.foundations import ShallowFoundation
from pygeotech.settlement import elastic_settlement
from pygeotech.standards import (
    design_action,
    design_bearing_resistance,
    design_shear_strength,
    factor_set_for,
)

_CODES = {
    "EC7-DA1": DesignStandard.EUROCODE7_DA1,
    "EC7-DA2": DesignStandard.EUROCODE7_DA2,
    "EC7-DA3": DesignStandard.EUROCODE7_DA3,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="pyGeotech shallow-foundation "
                                            "design calculator")
    p.add_argument("--width", type=float, default=2.5, help="footing B [m]")
    p.add_argument("--length", type=float, default=2.5, help="footing L [m]")
    p.add_argument("--depth", type=float, default=1.5,
                   help="embedment Df [m]")
    p.add_argument("--gamma", type=float, default=18.0,
                   help="unit weight [kN/m^3]")
    p.add_argument("--cohesion", type=float, default=5.0, help="c' [kPa]")
    p.add_argument("--phi", type=float, default=32.0, help="phi' [deg]")
    p.add_argument("--load", type=float, default=3000.0,
                   help="vertical service load [kN]")
    p.add_argument("--modulus", type=float, default=25000.0,
                   help="Young's modulus Es [kPa]")
    p.add_argument("--water-table", type=float, default=math.inf,
                   help="water-table depth [m] (default: dry)")
    p.add_argument("--method", default="vesic",
                   choices=["terzaghi", "meyerhof", "hansen", "vesic"])
    p.add_argument("--code", default="EC7-DA2", choices=list(_CODES),
                   help="design code check")
    p.add_argument("--fig", default="examples/output/design_pressure_bulb.png")
    return p.parse_args()


def rule(title: str) -> None:
    print("\n" + title)
    print("-" * 66)


def main() -> None:
    a = parse_args()
    area = a.width * a.length
    q_applied = a.load / area

    foundation = ShallowFoundation(
        width=a.width, length=a.length, depth=a.depth, gamma=a.gamma,
        cohesion=a.cohesion, phi=a.phi, method=a.method,
        water_table_depth=a.water_table)
    cap = foundation.capacity()

    settlement_m = elastic_settlement(
        q_applied, a.width, a.length, youngs_modulus=a.modulus,
        poisson_ratio=0.3, position="center")

    # ---------------------------------------------------------------- report
    print("=" * 66)
    print("  pyGeotech - SHALLOW FOUNDATION DESIGN CHECK")
    print("=" * 66)

    rule("INPUTS")
    print(f"  Footing            : {a.width:g} x {a.length:g} m, "
          f"Df = {a.depth:g} m")
    print(f"  Soil               : gamma = {a.gamma:g} kN/m3, "
          f"c' = {a.cohesion:g} kPa, phi' = {a.phi:g} deg")
    print(f"  Service load       : {a.load:g} kN  ->  applied pressure "
          f"q = {q_applied:.1f} kPa")

    rule(f"BEARING CAPACITY  ({a.method.capitalize()})")
    print(f"  Factors            : Nc = {cap.nc:.2f}, Nq = {cap.nq:.2f}, "
          f"Ngamma = {cap.n_gamma:.2f}")
    print(f"  Ultimate           : q_ult     = {cap.q_ultimate:8.1f} kPa")
    print(f"  Net ultimate       : q_net,ult = {cap.q_net_ultimate:8.1f} kPa")
    print(f"  Allowable (net)    : q_all,net = {cap.q_allowable_net:8.1f} kPa "
          f"(FS = {cap.factor_of_safety:g})")
    fs_asd = "OK" if q_applied <= cap.q_allowable_net else "NOT OK"
    print(f"  ASD check          : q = {q_applied:.1f} <= q_all,net "
          f"{cap.q_allowable_net:.1f} kPa  ->  {fs_asd}")

    rule("SETTLEMENT  (immediate, elastic)")
    print(f"  Es = {a.modulus:g} kPa, nu = 0.30, footing centre")
    print(f"  Elastic settlement : Se = {settlement_m * 1000:.1f} mm")
    limit = 25.0
    print(f"  Serviceability     : {settlement_m * 1000:.1f} mm vs {limit:g} mm "
          f"limit  ->  {'OK' if settlement_m * 1000 <= limit else 'CHECK'}")

    rule(f"DESIGN-CODE CHECK  ({a.code})")
    fs = factor_set_for(_CODES[a.code])
    phi_d, c_d, _ = design_shear_strength(fs, friction_angle=a.phi,
                                          cohesion=a.cohesion)
    # Full EC7: recompute the resistance with design (factored) strengths,
    # then apply the resistance factor gamma_Rv.
    cap_d = ShallowFoundation(
        width=a.width, length=a.length, depth=a.depth, gamma=a.gamma,
        cohesion=c_d, phi=phi_d, method=a.method,
        water_table_depth=a.water_table).capacity()
    r_d = design_bearing_resistance(fs, cap_d.q_net_ultimate)
    e_d = design_action(fs, permanent=q_applied)
    util = e_d / r_d if r_d > 0 else math.inf
    print(f"  Partial factors    : {fs.name} "
          f"(gamma_phi={fs.gamma_phi:g}, gamma_Rv={fs.gamma_rv:g})")
    print(f"  Design strengths   : phi_d = {phi_d:.1f} deg, c_d = {c_d:.1f} kPa")
    print(f"  Design resistance  : R_d = {r_d:8.1f} kPa")
    print(f"  Design action      : E_d = {e_d:8.1f} kPa")
    print(f"  Utilisation        : E_d / R_d = {util:.2f}  ->  "
          f"{'OK' if util <= 1.0 else 'NOT OK'}")

    # ---------------------------------------------------------------- figure
    try:
        from pygeotech.stresses.plotting import plot_pressure_bulb
        plot_pressure_bulb(q_applied, width=a.width, length=a.length,
                           method="boussinesq", save_path=a.fig)
        rule("OUTPUT")
        print(f"  Pressure-bulb figure written to: {a.fig}")
    except Exception as exc:                       # pragma: no cover
        print(f"  (figure skipped: {exc})")

    print("\n" + "=" * 66)
    verdict = ("ADEQUATE" if (q_applied <= cap.q_allowable_net
                              and util <= 1.0) else "REVISE DESIGN")
    print(f"  OVERALL VERDICT: {verdict}")
    print("=" * 66)


if __name__ == "__main__":
    main()
