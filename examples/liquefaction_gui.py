"""Interactive liquefaction-triggering GUI built on pyGeotech.

A Tkinter desktop app that runs the Idriss-Boulanger simplified procedure
over a borehole: enter the earthquake (a_max, Mw), the site (water table,
unit weights) and an SPT profile (depth, N60, fines), press *Hesapla*, and
see the factor-of-safety-against-liquefaction profile with depth, plus the
Liquefaction Potential Index (LPI).

Run it (from anywhere, once pygeotech is installed):

    python3 examples/liquefaction_gui.py

The engineering lives in pyGeotech (soil_dynamics + characterization);
this file is only the interface.
"""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Dict, List, Tuple

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from pygeotech.characterization import corrected_n1_60
from pygeotech.constants import GAMMA_W
from pygeotech.soil_dynamics import liquefaction_factor_of_safety


# --------------------------------------------------------------------------
# Calculation engine (pure, testable — no GUI).  Everything here is pyGeotech.
# --------------------------------------------------------------------------
def compute_liquefaction(
    a_max_g: float,
    magnitude: float,
    water_table: float,
    gamma: float,
    gamma_sat: float,
    borehole: List[Tuple[float, float, float]],
    g: float = 9.81,
) -> Tuple[List[Dict[str, float]], float]:
    """Liquefaction FS profile and LPI for a borehole.

    Parameters
    ----------
    a_max_g
        Peak ground acceleration as a fraction of g.
    magnitude
        Earthquake moment magnitude Mw.
    water_table
        Water-table depth [m].
    gamma, gamma_sat
        Moist and saturated unit weights [kN/m^3].
    borehole
        Rows of ``(depth [m], N60, fines_content [%])``.

    Returns
    -------
    (rows, lpi)
        Per-depth results and the Liquefaction Potential Index.
    """
    a_max = a_max_g * g
    rows: List[Dict[str, float]] = []
    for depth, n60, fines in sorted(borehole, key=lambda r: r[0]):
        sigma_v0 = gamma * min(depth, water_table) + gamma_sat * max(
            0.0, depth - water_table)
        u = GAMMA_W * max(0.0, depth - water_table)
        sigma_v0_eff = max(sigma_v0 - u, 1.0)
        if depth <= water_table:
            rows.append(dict(depth=depth, csr=math.nan, crr_adj=math.nan,
                             fs=math.inf, liquefies=False))
            continue
        n1_60 = corrected_n1_60(n60, sigma_v0_eff)
        res = liquefaction_factor_of_safety(
            n1_60=n1_60, fines_content=fines, a_max=a_max,
            magnitude=magnitude, sigma_v0=sigma_v0,
            sigma_v0_eff=sigma_v0_eff, depth=depth)
        rows.append(dict(depth=depth, csr=res.csr,
                         crr_adj=res.factor_of_safety * res.csr,
                         fs=res.factor_of_safety, liquefies=res.liquefiable))

    # Liquefaction Potential Index (Iwasaki 1982), integrated 0-20 m.
    z = np.array([r["depth"] for r in rows])
    fs = np.array([min(r["fs"], 5.0) for r in rows])
    f_factor = np.where((z <= 20.0), np.maximum(0.0, 1.0 - fs), 0.0)
    weight = np.maximum(0.0, 10.0 - 0.5 * z)
    lpi = float(np.trapz(f_factor * weight, z)) if z.size > 1 else 0.0
    return rows, lpi


def lpi_severity(lpi: float) -> str:
    """Iwasaki severity class for an LPI value."""
    if lpi <= 0.0:
        return "yok (none)"
    if lpi < 5.0:
        return "dusuk (low)"
    if lpi < 15.0:
        return "yuksek (high)"
    return "cok yuksek (very high)"


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------
_FIELDS = [
    ("a_max_g", "Maks. ivme, a_max [g]", "0.35"),
    ("magnitude", "Deprem büyüklüğü, Mw", "7.5"),
    ("water_table", "Yeraltı suyu derinliği [m]", "2.0"),
    ("gamma", "Nemli birim ağırlık, γ [kN/m³]", "18"),
    ("gamma_sat", "Doygun birim ağırlık, γsat [kN/m³]", "20"),
]

_DEFAULT_BOREHOLE = (
    "# derinlik  N60  ince%\n"
    "2   8  15\n4   10 12\n6   9  20\n8   12 8\n"
    "10  15 25\n12  20 10\n14  25 5\n16  30 8\n"
)


class LiquefactionApp:
    """Tkinter front-end; all mechanics live in :func:`compute_liquefaction`."""

    def __init__(self, root: tk.Tk) -> None:
        root.title("pyGeotech — Sıvılaşma Analizi")
        self.entries: Dict[str, tk.Entry] = {}

        main = ttk.Frame(root, padding=10)
        main.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        form = ttk.LabelFrame(main, text="Girdiler", padding=10)
        form.grid(row=0, column=0, sticky="nw", padx=(0, 10))
        for i, (key, label, default) in enumerate(_FIELDS):
            ttk.Label(form, text=label).grid(row=i, column=0, sticky="w",
                                             pady=2)
            entry = ttk.Entry(form, width=10, justify="right")
            entry.insert(0, default)
            entry.grid(row=i, column=1, sticky="e", pady=2, padx=(8, 0))
            self.entries[key] = entry

        ttk.Label(form, text="Sondaj (derinlik  N60  ince%):").grid(
            row=len(_FIELDS), column=0, columnspan=2, sticky="w", pady=(8, 2))
        self.borehole = tk.Text(form, width=26, height=9, font=("Menlo", 10))
        self.borehole.insert("1.0", _DEFAULT_BOREHOLE)
        self.borehole.grid(row=len(_FIELDS) + 1, column=0, columnspan=2)

        ttk.Button(form, text="Hesapla", command=self.calculate).grid(
            row=len(_FIELDS) + 2, column=0, columnspan=2, pady=(8, 6),
            sticky="ew")
        self.summary = tk.Label(form, text="", justify="left",
                                font=("Menlo", 10), anchor="w")
        self.summary.grid(row=len(_FIELDS) + 3, column=0, columnspan=2,
                          sticky="w")

        self.figure = Figure(figsize=(6.4, 5.6), dpi=100)
        self.ax_ratio = self.figure.add_subplot(1, 2, 1)
        self.ax_fs = self.figure.add_subplot(1, 2, 2)
        self.canvas = FigureCanvasTkAgg(self.figure, master=main)
        self.canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        self.calculate()

    def _parse_borehole(self) -> List[Tuple[float, float, float]]:
        rows = []
        for line in self.borehole.get("1.0", tk.END).splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
        if not rows:
            raise ValueError("En az bir sondaj satırı girin.")
        return rows

    def calculate(self) -> None:
        try:
            params = {k: float(e.get()) for k, e in self.entries.items()}
            borehole = self._parse_borehole()
            rows, lpi = compute_liquefaction(borehole=borehole, **params)
        except ValueError as exc:
            messagebox.showerror("Girdi hatası", str(exc))
            return
        except Exception as exc:                       # noqa: BLE001
            messagebox.showerror("Hesap hatası", str(exc))
            return

        liq_depths = [r["depth"] for r in rows if r["liquefies"]]
        txt = f"LPI = {lpi:.1f}  ->  {lpi_severity(lpi)}\n"
        txt += ("Sıvılaşan derinlikler: "
                + (", ".join(f"{d:g}m" for d in liq_depths) if liq_depths
                   else "yok"))
        self.summary.config(text=txt)
        self._draw(rows)

    def _draw(self, rows: List[Dict[str, float]]) -> None:
        z = np.array([r["depth"] for r in rows])
        csr = np.array([r["csr"] for r in rows])
        crr = np.array([r["crr_adj"] for r in rows])
        fs = np.array([min(r["fs"], 3.0) for r in rows])
        liq = np.array([r["liquefies"] for r in rows])

        for ax in (self.ax_ratio, self.ax_fs):
            ax.clear()

        self.ax_ratio.plot(csr, z, "o-", color="C3", lw=1.4, ms=4,
                           label="CSR (talep)")
        self.ax_ratio.plot(crr, z, "s-", color="C2", lw=1.4, ms=4,
                           label="CRR·MSF·Kσ (direnç)")
        self.ax_ratio.set_xlabel("Çevrimsel oran")
        self.ax_ratio.set_ylabel("Derinlik, z (m)")
        self.ax_ratio.set_ylim(z.max() + 1, 0.0)
        self.ax_ratio.set_xlim(left=0.0)
        self.ax_ratio.legend(fontsize=7, loc="lower right")
        self.ax_ratio.set_title("Talep vs Direnç", fontsize=9)

        self.ax_fs.plot(fs, z, "o-", color="k", lw=1.4, ms=4)
        self.ax_fs.axvline(1.0, color="C3", lw=1.2, ls="--")
        self.ax_fs.fill_betweenx(z, 0, 1.0, color="C3", alpha=0.12)
        if liq.any():
            self.ax_fs.scatter(fs[liq], z[liq], color="C3", zorder=5, s=45,
                               label="sıvılaşır")
            self.ax_fs.legend(fontsize=7, loc="lower right")
        self.ax_fs.set_xlabel("Güvenlik sayısı, FS")
        self.ax_fs.set_ylim(z.max() + 1, 0.0)
        self.ax_fs.set_xlim(0.0, 3.0)
        self.ax_fs.set_title("FS profili", fontsize=9)

        self.figure.tight_layout()
        self.canvas.draw()


def main() -> None:
    root = tk.Tk()
    LiquefactionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
