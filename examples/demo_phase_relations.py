"""MVP demo for pygeotech.phase_relations.

Run from the repository root:

    python examples/demo_phase_relations.py
"""

from pygeotech.phase_relations import Soil, classify_uscs
from pygeotech.phase_relations.plotting import plot_plasticity_chart


def main() -> None:
    # ------------------------------------------------------------------
    # 1) Phase relations: solve the full state from three knowns.
    # ------------------------------------------------------------------
    soil = Soil(w=0.20, gs=2.70, e=0.80)
    print(soil.report())
    print()

    # Alternative input route: field unit weight + water content.
    field_soil = Soil(gamma=18.0, w=0.15, gs=2.65)
    print(f"Field sample -> {field_soil!r}")
    print()

    # ------------------------------------------------------------------
    # 2) USCS classification of three laboratory samples.
    # ------------------------------------------------------------------
    samples = {
        "BH1-S1": classify_uscs(60.0, 90.0, liquid_limit=40.0,
                                plasticity_index=22.0),
        "BH1-S2": classify_uscs(8.0, 95.0, d10=0.1, d30=0.2, d60=0.4,
                                nonplastic=True),
        "BH2-S1": classify_uscs(95.0, 100.0, liquid_limit=60.0,
                                plasticity_index=20.0),
    }
    print("USCS classification:")
    for sample_id, result in samples.items():
        print(f"  {sample_id}: {result}")
    print()

    # ------------------------------------------------------------------
    # 3) Publication-ready Casagrande plasticity chart (vector PDF).
    # ------------------------------------------------------------------
    chart_points = [
        (40.0, 22.0, "BH1-S1 (CL)"),
        (60.0, 20.0, "BH2-S1 (MH)"),
        (60.0, 35.0, "BH3-S1 (CH)"),
        (25.0, 5.0, "BH3-S2 (CL-ML)"),
    ]
    for path in ("examples/output/plasticity_chart.pdf",
                 "examples/output/plasticity_chart.png"):
        plot_plasticity_chart(chart_points, save_path=path)
        print(f"Plasticity chart written to {path}")


if __name__ == "__main__":
    main()
