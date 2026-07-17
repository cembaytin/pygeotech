"""Interactive pile-design GUI built on pyGeotech.

A small Tkinter desktop application that uses pyGeotech as its calculation
engine: enter the pile, soil and load data, press *Hesapla / Calculate*,
and see the axial capacity plus the laterally loaded pile response
(deflection and bending-moment profiles) drawn live.

Run it (from anywhere, once pygeotech is installed):

    python3 examples/pile_design_gui.py

The engineering all lives in pyGeotech; this file is only the interface.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Dict, Tuple

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from pygeotech.foundations import (
    PileCapacityResult,
    PyResult,
    matlock_clay_py,
    pile_capacity_alpha,
    solve_laterally_loaded_pile,
)


# --------------------------------------------------------------------------
# Calculation engine (pure, testable — no GUI).  Everything here is pyGeotech.
# --------------------------------------------------------------------------
def compute_pile_design(
    diameter: float,
    length: float,
    flexural_rigidity: float,
    undrained_strength: float,
    gamma_eff: float,
    alpha: float,
    epsilon50: float,
    lateral_load: float,
    moment: float,
    factor_of_safety: float,
    n_segments: int = 100,
) -> Tuple[PileCapacityResult, PyResult]:
    """Axial (alpha-method) and lateral (Matlock p-y) pile response."""
    depth = np.linspace(0.0, length, n_segments + 1)
    su_profile = np.full_like(depth, undrained_strength)
    axial = pile_capacity_alpha(diameter, depth, su_profile, alpha=alpha,
                                factor_of_safety=factor_of_safety)
    curve = matlock_clay_py(diameter, undrained_strength, gamma_eff,
                            epsilon50=epsilon50)
    lateral = solve_laterally_loaded_pile(length, flexural_rigidity, curve,
                                          lateral_load=lateral_load,
                                          moment=moment)
    return axial, lateral


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------
#: (key, label, default) for every input field.
_FIELDS = [
    ("diameter", "Çap, D [m]", "0.6"),
    ("length", "Boy, L [m]", "15"),
    ("flexural_rigidity", "Eğilme rijitliği, EI [kN·m²]", "190000"),
    ("undrained_strength", "Drenajsız dayanım, su [kPa]", "60"),
    ("gamma_eff", "Efektif birim ağırlık, γ' [kN/m³]", "9"),
    ("alpha", "Aderans faktörü, α [-]", "0.55"),
    ("epsilon50", "ε₅₀ [-]", "0.01"),
    ("lateral_load", "Yanal yük, H [kN]", "150"),
    ("moment", "Başlık momenti, M [kN·m]", "0"),
    ("factor_of_safety", "Güvenlik sayısı, FS [-]", "3"),
]


class PileDesignApp:
    """Tkinter front-end; delegates all mechanics to :func:`compute_pile_design`."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("pyGeotech — Kazık Tasarımı")
        self.entries: Dict[str, tk.Entry] = {}

        main = ttk.Frame(root, padding=10)
        main.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # ---- Left: inputs ------------------------------------------------
        form = ttk.LabelFrame(main, text="Girdiler", padding=10)
        form.grid(row=0, column=0, sticky="nw", padx=(0, 10))
        for i, (key, label, default) in enumerate(_FIELDS):
            ttk.Label(form, text=label).grid(row=i, column=0, sticky="w",
                                             pady=2)
            entry = ttk.Entry(form, width=12, justify="right")
            entry.insert(0, default)
            entry.grid(row=i, column=1, sticky="e", pady=2, padx=(8, 0))
            self.entries[key] = entry

        ttk.Button(form, text="Hesapla", command=self.calculate).grid(
            row=len(_FIELDS), column=0, columnspan=2, pady=(10, 6), sticky="ew")

        self.results = tk.Text(form, width=34, height=12, font=("Menlo", 10),
                               relief="solid", borderwidth=1)
        self.results.grid(row=len(_FIELDS) + 1, column=0, columnspan=2)

        # ---- Right: figure ----------------------------------------------
        self.figure = Figure(figsize=(6.2, 5.4), dpi=100)
        self.ax_defl = self.figure.add_subplot(1, 2, 1)
        self.ax_mom = self.figure.add_subplot(1, 2, 2)
        self.canvas = FigureCanvasTkAgg(self.figure, master=main)
        self.canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        self.calculate()          # show a result on startup

    # ------------------------------------------------------------------
    def _read_inputs(self) -> Dict[str, float]:
        return {key: float(entry.get()) for key, entry in self.entries.items()}

    def calculate(self) -> None:
        try:
            params = self._read_inputs()
            axial, lateral = compute_pile_design(**params)
        except ValueError as exc:
            messagebox.showerror("Girdi hatası", f"Sayısal değer bekleniyor.\n{exc}")
            return
        except Exception as exc:                       # noqa: BLE001
            messagebox.showerror("Hesap hatası", str(exc))
            return

        self._show_results(axial, lateral)
        self._draw(lateral)

    def _show_results(self, axial: PileCapacityResult,
                      lateral: PyResult) -> None:
        text = (
            "EKSENEL KAPASİTE (α-yöntemi)\n"
            f"  Çeper (Qs)   : {axial.shaft:8.0f} kN\n"
            f"  Uç (Qb)      : {axial.base:8.0f} kN\n"
            f"  Nihai (Qult) : {axial.ultimate:8.0f} kN\n"
            f"  İzin (Qall)  : {axial.allowable:8.0f} kN\n"
            f"                 (FS = {axial.factor_of_safety:g})\n\n"
            "YANAL DAVRANIŞ (Matlock p-y)\n"
            f"  Başlık öteleme : {lateral.head_deflection * 1000:7.1f} mm\n"
            f"  Maks. moment   : {lateral.max_moment:7.0f} kN·m\n"
        )
        self.results.delete("1.0", tk.END)
        self.results.insert("1.0", text)

    def _draw(self, lateral: PyResult) -> None:
        z = lateral.depth
        for ax in (self.ax_defl, self.ax_mom):
            ax.clear()
        self.ax_defl.plot(lateral.deflection * 1000, z, color="C0", lw=1.6)
        self.ax_defl.axvline(0, color="0.7", lw=0.6)
        self.ax_defl.set_xlabel("Öteleme (mm)")
        self.ax_defl.set_ylabel("Derinlik, z (m)")
        self.ax_defl.set_ylim(z.max(), 0.0)
        self.ax_defl.set_title("Yanal öteleme", fontsize=9)

        self.ax_mom.plot(lateral.moment, z, color="C3", lw=1.6)
        self.ax_mom.axvline(0, color="0.7", lw=0.6)
        self.ax_mom.set_xlabel("Moment (kN·m)")
        self.ax_mom.set_ylim(z.max(), 0.0)
        self.ax_mom.set_title("Eğilme momenti", fontsize=9)

        self.figure.tight_layout()
        self.canvas.draw()


def main() -> None:
    root = tk.Tk()
    PileDesignApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
