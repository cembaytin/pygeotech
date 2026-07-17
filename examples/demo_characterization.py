"""MVP demo for the pyGeotech characterization submodule (SPT + CPT).

Run from the repository root:

    python examples/demo_characterization.py
"""

import numpy as np

from pygeotech.characterization import CPTLog, SPTLog
from pygeotech.characterization.plotting import (
    plot_cpt_profile,
    plot_spt_profile,
)
from pygeotech.stresses import SoilLayer, SoilProfile


def spt_demo() -> None:
    print("== SPT ==")
    profile = SoilProfile(
        [SoilLayer(15.0, gamma=18.0, gamma_sat=20.0)],
        water_table_depth=2.0)
    depth = [1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0]
    n_field = [8, 12, 15, 19, 22, 26, 30, 34]
    log = SPTLog(depth=depth, n_field=n_field, profile=profile,
                 energy_ratio=60.0)
    phi = log.friction_angle()
    dr = log.relative_density()
    print(f"  depth 6.0 m -> N60={log.n60()[3]:.0f}, "
          f"(N1)60={log.n1_60()[3]:.1f}, phi={phi[3]:.1f} deg, "
          f"Dr={dr[3] * 100:.0f}%")
    plot_spt_profile(log, save_path="examples/output/spt_profile.png")
    print("  figure -> examples/output/spt_profile.png\n")


def cpt_demo() -> None:
    print("== CPT ==")
    # Synthetic sounding: sand (0-4 m), clay (4-8 m), dense sand (8-12 m).
    depth = np.linspace(0.5, 12.0, 120)
    qc = np.piecewise(
        depth,
        [depth < 4.0, (depth >= 4.0) & (depth < 8.0), depth >= 8.0],
        [lambda z: 6000 + 400 * z,
         lambda z: 900 + 60 * (z - 4.0),
         lambda z: 14000 + 300 * (z - 8.0)])
    fs = np.piecewise(
        depth,
        [depth < 4.0, (depth >= 4.0) & (depth < 8.0), depth >= 8.0],
        [40.0, 55.0, 90.0])
    log = CPTLog(depth, qc, fs, water_table_depth=1.5, nkt=14.0)
    res = log.process()
    # Report the mid-clay point.
    j = int(np.argmin(np.abs(depth - 6.0)))
    print(f"  depth 6.0 m -> Ic={res.ic[j]:.2f} (SBT zone {res.sbt_zone[j]}), "
          f"su={res.undrained_strength[j]:.0f} kPa, "
          f"gamma~{res.unit_weight[j]:.1f} kN/m^3")
    plot_cpt_profile(log, res, save_path="examples/output/cpt_profile.png")
    print("  figure -> examples/output/cpt_profile.png")


def main() -> None:
    spt_demo()
    cpt_demo()


if __name__ == "__main__":
    main()
