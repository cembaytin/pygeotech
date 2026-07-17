"""MVP demo for pygeotech.stresses.

Run from the repository root:

    python examples/demo_stresses.py
"""

from pygeotech.stresses import SoilLayer, SoilProfile, boussinesq_rectangle
from pygeotech.stresses.plotting import (
    plot_geostatic_profile,
    plot_pressure_bulb,
)


def main() -> None:
    # ------------------------------------------------------------------
    # 1) Geostatic profile of a layered deposit with a shallow water table.
    # ------------------------------------------------------------------
    profile = SoilProfile(
        layers=[
            SoilLayer(2.0, gamma=16.5, name="Fill / moist sand"),
            SoilLayer(4.0, gamma=18.0, gamma_sat=19.5, name="Medium sand"),
            SoilLayer(6.0, gamma=17.0, gamma_sat=18.5, name="Soft clay"),
        ],
        water_table_depth=2.0,
        surcharge=10.0,
    )
    print(profile)
    print(profile.summary_table())
    print()

    plot_geostatic_profile(profile, save_path="examples/output/geostatic.pdf")
    plot_geostatic_profile(profile, save_path="examples/output/geostatic.png")
    print("Geostatic diagram written to examples/output/geostatic.{pdf,png}")

    # ------------------------------------------------------------------
    # 2) Vertical stress increment under a rectangular footing.
    # ------------------------------------------------------------------
    q, b, l = 150.0, 2.0, 3.0
    for depth in (1.0, 2.0, 4.0):
        ds = float(boussinesq_rectangle(q, b, l, depth))
        print(f"Delta sigma_z at {depth:.0f} m below footing centre "
              f"= {ds:6.2f} kPa  ({ds / q:.3f} q)")
    print()

    # ------------------------------------------------------------------
    # 3) Pressure bulbs: Boussinesq (strip) and Westergaard comparison.
    # ------------------------------------------------------------------
    plot_pressure_bulb(q, width=2.0, method="boussinesq",
                       save_path="examples/output/bulb_boussinesq.png")
    plot_pressure_bulb(q, width=2.0, method="westergaard", nu=0.0,
                       save_path="examples/output/bulb_westergaard.png")
    print("Pressure bulbs written to examples/output/bulb_*.png")


if __name__ == "__main__":
    main()
