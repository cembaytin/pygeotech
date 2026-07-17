"""pyGeotech Suite — a full tabbed geotechnical analysis interface.

One window, many tabs, each a different analysis, all powered by pyGeotech:

    Sınıflandırma · Gerilme profili · Sığ temel · Kazık ·
    Şev stabilitesi · Konsolidasyon · Sıvılaşma

Run it (from anywhere, once pygeotech is installed):

    python3 examples/geotech_suite.py

Every tab is a thin ``AnalysisTab`` subclass: it declares its input fields
and a ``run()`` method that calls pyGeotech and draws on the tab's figure.
All engineering lives in pyGeotech; this file is only the interface.
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
from pygeotech.consolidation import Consolidation1D
from pygeotech.foundations import (
    ShallowFoundation,
    matlock_clay_py,
    pile_capacity_alpha,
    solve_laterally_loaded_pile,
)
from pygeotech.phase_relations import classify_aashto, classify_uscs
from pygeotech.slope_stability import critical_circle, simple_slope_surface
from pygeotech.soil_dynamics import liquefaction_factor_of_safety
from pygeotech.stresses import SoilLayer, SoilProfile


# ==========================================================================
# Reusable tab base class
# ==========================================================================
class AnalysisTab(ttk.Frame):
    """A generic analysis tab: input form + results box + embedded figure.

    Subclasses set ``title`` and ``fields`` and implement ``run(params)``,
    which returns a summary string and draws on ``self.figure``.
    """

    title = "Analiz"
    #: (key, label, default[, kind]) — kind in {"num", "str", "combo:a,b", "text"}
    fields: List[tuple] = []

    def __init__(self, master: tk.Widget) -> None:
        super().__init__(master, padding=8)
        self.widgets: Dict[str, tuple] = {}

        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky="nw", padx=(0, 8))
        row = 0
        for spec in self.fields:
            key, label, default = spec[0], spec[1], spec[2]
            kind = spec[3] if len(spec) > 3 else "num"
            if kind == "text":
                ttk.Label(left, text=label).grid(row=row, column=0,
                                                 columnspan=2, sticky="w")
                widget = tk.Text(left, width=24, height=7, font=("Menlo", 9))
                widget.insert("1.0", default)
                widget.grid(row=row + 1, column=0, columnspan=2, pady=2)
                self.widgets[key] = ("text", widget)
                row += 2
                continue
            ttk.Label(left, text=label).grid(row=row, column=0, sticky="w",
                                             pady=1)
            if kind.startswith("combo"):
                var = tk.StringVar(value=default)
                opts = kind.split(":", 1)[1].split(",")
                ttk.Combobox(left, textvariable=var, values=opts, width=10,
                             state="readonly").grid(row=row, column=1,
                                                    sticky="e", padx=(6, 0))
                self.widgets[key] = ("combo", var)
            else:
                entry = ttk.Entry(left, width=12, justify="right")
                entry.insert(0, default)
                entry.grid(row=row, column=1, sticky="e", padx=(6, 0))
                self.widgets[key] = (kind, entry)
            row += 1

        ttk.Button(left, text="Hesapla", command=self.calculate).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(8, 4))
        self.results = tk.Text(left, width=36, height=10, font=("Menlo", 9),
                               relief="solid", borderwidth=1)
        self.results.grid(row=row + 1, column=0, columnspan=2)

        self.figure = Figure(figsize=(5.8, 4.9), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._computed = False

    # -- helpers -------------------------------------------------------
    def read(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for key, (kind, widget) in self.widgets.items():
            if kind == "text":
                out[key] = widget.get("1.0", tk.END)
            elif kind in ("combo", "str"):
                out[key] = widget.get()
            else:
                out[key] = float(widget.get())
        return out

    def calculate(self) -> None:
        try:
            self.figure.clf()
            text = self.run(self.read())
        except Exception as exc:                       # noqa: BLE001
            messagebox.showerror(f"{self.title} — hata", str(exc))
            return
        self.results.delete("1.0", tk.END)
        self.results.insert("1.0", text)
        self.figure.tight_layout()
        self.canvas.draw()
        self._computed = True

    def run(self, p: Dict[str, object]) -> str:        # pragma: no cover
        raise NotImplementedError


# ==========================================================================
# Individual analyses
# ==========================================================================
class ClassificationTab(AnalysisTab):
    title = "Sınıflandırma"
    fields = [
        ("ll", "Likit limit, LL [%]", "42"),
        ("pi", "Plastisite indeksi, PI [%]", "18"),
        ("no4", "No.4 elekten geçen [%]", "95"),
        ("no10", "No.10 elekten geçen [%]", "90"),
        ("no40", "No.40 elekten geçen [%]", "75"),
        ("no200", "No.200 elekten geçen [%]", "60"),
    ]

    def run(self, p) -> str:
        uscs = classify_uscs(p["no200"], p["no4"], liquid_limit=p["ll"],
                             plasticity_index=p["pi"])
        aashto = classify_aashto(p["no10"], p["no40"], p["no200"],
                                 liquid_limit=p["ll"], plasticity_index=p["pi"])
        ax = self.figure.add_subplot(111)
        ll = np.linspace(0, 100, 200)
        a_line = np.where(ll > 25.5, 0.73 * (ll - 20), 4.0)
        u_line = np.where(ll > 16, 0.9 * (ll - 8), 7.0)
        ax.plot(ll, np.clip(a_line, 0, None), "k-", lw=1.2, label="A-çizgisi")
        ax.plot(ll, np.clip(u_line, 0, None), "k--", lw=0.8, label="U-çizgisi")
        ax.axvline(50, color="0.6", lw=0.6, ls=":")
        ax.scatter(p["ll"], p["pi"], s=80, color="C3", zorder=5,
                   edgecolor="k")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 60)
        ax.set_xlabel("Likit limit, LL (%)")
        ax.set_ylabel("Plastisite indeksi, PI (%)")
        ax.legend(fontsize=8)
        ax.set_title("Plastisite kartı", fontsize=9)
        return (f"USCS   : {uscs.symbol}\n         {uscs.group_name}\n\n"
                f"AASHTO : {aashto.group} (GI={aashto.group_index})\n"
                f"         {aashto.description}")


class StressTab(AnalysisTab):
    title = "Gerilme Profili"
    fields = [
        ("layers", "Tabakalar (kalınlık γ γsat):",
         "2  16.5\n4  18  19.5\n6  17  18.5", "text"),
        ("wt", "Yeraltı suyu derinliği [m]", "2.0"),
        ("surcharge", "Sürşarj, q [kPa]", "10"),
    ]

    def run(self, p) -> str:
        layers = []
        for line in str(p["layers"]).splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            t, g = float(parts[0]), float(parts[1])
            gsat = float(parts[2]) if len(parts) > 2 else None
            layers.append(SoilLayer(t, g, gsat))
        profile = SoilProfile(layers, water_table_depth=p["wt"],
                              surcharge=p["surcharge"])
        depth, sv, u, eff = profile.profile(dz=0.1)
        ax = self.figure.add_subplot(111)
        ax.plot(sv, depth, "C3", lw=1.6, label=r"$\sigma_v$")
        ax.plot(u, depth, "C0--", lw=1.4, label="$u$")
        ax.plot(eff, depth, "k", lw=1.8, label=r"$\sigma'_v$")
        ax.axhline(p["wt"], color="C0", lw=0.8, ls="-.")
        ax.set_xlabel("Gerilme (kPa)")
        ax.set_ylabel("Derinlik (m)")
        ax.set_ylim(depth.max(), 0)
        ax.legend(fontsize=8)
        z_base = depth[-1]
        s = profile.state(z_base)
        return (f"Taban (z={z_base:g} m):\n"
                f"  σv  = {s.total_vertical:7.1f} kPa\n"
                f"  u   = {s.pore_pressure:7.1f} kPa\n"
                f"  σ'v = {s.effective_vertical:7.1f} kPa")


class ShallowFoundationTab(AnalysisTab):
    title = "Sığ Temel"
    fields = [
        ("width", "Genişlik, B [m]", "2.5"),
        ("length", "Uzunluk, L [m]", "2.5"),
        ("depth", "Derinlik, Df [m]", "1.5"),
        ("gamma", "γ [kN/m³]", "18"),
        ("cohesion", "c [kPa]", "5"),
        ("phi", "φ [°]", "32"),
        ("load", "Servis yükü [kN]", "3000"),
        ("method", "Yöntem", "vesic", "combo:terzaghi,meyerhof,hansen,vesic"),
    ]

    def run(self, p) -> str:
        f = ShallowFoundation(width=p["width"], length=p["length"],
                              depth=p["depth"], gamma=p["gamma"],
                              cohesion=p["cohesion"], phi=p["phi"],
                              method=p["method"])
        cap = f.capacity()
        q = p["load"] / (p["width"] * p["length"])
        ax = self.figure.add_subplot(111)
        vals = [q, cap.q_allowable_net, cap.q_ultimate]
        labels = ["q uygulanan", "q izin (net)", "q nihai"]
        colors = ["C0", "C2", "C3"]
        ax.bar(labels, vals, color=colors)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Basınç (kPa)")
        ax.set_title(f"{p['method'].capitalize()} taşıma gücü", fontsize=9)
        ok = "YETERLİ" if q <= cap.q_allowable_net else "YETERSİZ"
        return (f"Nc={cap.nc:.1f} Nq={cap.nq:.1f} Nγ={cap.n_gamma:.1f}\n"
                f"q_ult    = {cap.q_ultimate:7.0f} kPa\n"
                f"q_izin   = {cap.q_allowable_net:7.0f} kPa (FS=3)\n"
                f"q_uygul. = {q:7.0f} kPa\n"
                f"SONUÇ    : {ok}")


class PileTab(AnalysisTab):
    title = "Kazık"
    fields = [
        ("diameter", "Çap, D [m]", "0.6"),
        ("length", "Boy, L [m]", "15"),
        ("ei", "EI [kN·m²]", "190000"),
        ("su", "su [kPa]", "60"),
        ("gamma_eff", "γ' [kN/m³]", "9"),
        ("alpha", "α [-]", "0.55"),
        ("load_h", "Yanal yük, H [kN]", "150"),
        ("moment", "Moment, M [kN·m]", "0"),
    ]

    def run(self, p) -> str:
        depth = np.linspace(0, p["length"], 101)
        su_prof = np.full_like(depth, p["su"])
        axial = pile_capacity_alpha(p["diameter"], depth, su_prof,
                                    alpha=p["alpha"])
        curve = matlock_clay_py(p["diameter"], p["su"], p["gamma_eff"])
        lat = solve_laterally_loaded_pile(p["length"], p["ei"], curve,
                                          lateral_load=p["load_h"],
                                          moment=p["moment"])
        a1 = self.figure.add_subplot(1, 2, 1)
        a2 = self.figure.add_subplot(1, 2, 2)
        a1.plot(lat.deflection * 1000, lat.depth, "C0", lw=1.5)
        a1.axvline(0, color="0.7", lw=0.6)
        a1.set_xlabel("Öteleme (mm)")
        a1.set_ylabel("z (m)")
        a1.set_ylim(p["length"], 0)
        a1.set_title("Öteleme", fontsize=9)
        a2.plot(lat.moment, lat.depth, "C3", lw=1.5)
        a2.axvline(0, color="0.7", lw=0.6)
        a2.set_xlabel("Moment (kN·m)")
        a2.set_ylim(p["length"], 0)
        a2.set_title("Moment", fontsize=9)
        return (f"EKSENEL (α):\n  Qult = {axial.ultimate:6.0f} kN\n"
                f"  Qizin= {axial.allowable:6.0f} kN\n\n"
                f"YANAL (p-y):\n"
                f"  başlık öteleme = {lat.head_deflection * 1000:.1f} mm\n"
                f"  maks moment    = {lat.max_moment:.0f} kN·m")


class SlopeTab(AnalysisTab):
    title = "Şev Stabilitesi"
    fields = [
        ("height", "Yükseklik, H [m]", "10"),
        ("angle", "Şev açısı, β [°]", "30"),
        ("gamma", "γ [kN/m³]", "19"),
        ("cohesion", "c [kPa]", "15"),
        ("phi", "φ [°]", "25"),
        ("ru", "Boşluk basınç oranı, ru", "0.0"),
        ("method", "Yöntem", "bishop", "combo:fellenius,bishop,janbu"),
    ]

    def run(self, p) -> str:
        ground = simple_slope_surface(p["height"], p["angle"])
        fos, circle = critical_circle(
            ground, gamma=p["gamma"], cohesion=p["cohesion"],
            friction_angle=p["phi"], height=p["height"],
            slope_angle=p["angle"], ru=p["ru"], method=p["method"],
            n_centers=8, n_radii=8, n_slices=30)
        ax = self.figure.add_subplot(111)
        x = np.linspace(circle.xc - 1.3 * circle.radius,
                        circle.xc + 1.3 * circle.radius, 300)
        ax.plot(x, ground(x), "k", lw=1.6)
        xa = np.linspace(circle.xc - circle.radius,
                         circle.xc + circle.radius, 300)
        ya = circle.yc - np.sqrt(np.maximum(
            circle.radius ** 2 - (xa - circle.xc) ** 2, 0))
        below = ya < ground(xa) - 1e-9
        ax.plot(xa[below], ya[below], "C3", lw=1.8)
        ax.plot(circle.xc, circle.yc, "+", color="C3", ms=9)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        status = "GÜVENLİ" if fos >= 1.5 else ("SINIRDA" if fos >= 1.0
                                               else "GÖÇER")
        return (f"Yöntem : {p['method']}\n"
                f"Kritik FoS = {fos:.3f}\n"
                f"Durum  : {status}\n"
                f"Daire  : ({circle.xc:.1f}, {circle.yc:.1f}), "
                f"R={circle.radius:.1f}")


class ConsolidationTab(AnalysisTab):
    title = "Konsolidasyon"
    fields = [
        ("thickness", "Tabaka kalınlığı, H [m]", "6"),
        ("cv", "cv [m²/yıl]", "2.0"),
        ("cc", "Sıkışma indisi, Cc", "0.3"),
        ("e0", "Boşluk oranı, e0", "0.9"),
        ("sigma0", "σ'0 [kPa]", "80"),
        ("dsigma", "Δσ [kPa]", "100"),
        ("drainage", "Drenaj", "double", "combo:double,single"),
    ]

    def run(self, p) -> str:
        model = Consolidation1D(p["cv"], p["thickness"], drainage=p["drainage"])
        s_c = (p["cc"] / (1 + p["e0"]) * p["thickness"]
               * math.log10((p["sigma0"] + p["dsigma"]) / p["sigma0"])) * 1000
        t90 = model.time_for_degree(0.90)
        t50 = model.time_for_degree(0.50)
        times = np.linspace(1e-3, model.time_for_degree(0.95), 200)
        settle = np.array([model.average_degree(t) * s_c for t in times])
        ax = self.figure.add_subplot(111)
        ax.plot(times, settle, "C3", lw=1.6)
        ax.set_xlabel("Zaman (yıl)")
        ax.set_ylabel("Oturma (mm)")
        ax.set_ylim(s_c * 1.02, 0)
        ax.set_xlim(left=0)
        ax.set_title("Zaman–oturma", fontsize=9)
        return (f"Birincil oturma s_c = {s_c:.1f} mm\n"
                f"t50 = {t50:.2f} yıl\n"
                f"t90 = {t90:.2f} yıl")


class LiquefactionTab(AnalysisTab):
    title = "Sıvılaşma"
    fields = [
        ("a_max", "a_max [g]", "0.35"),
        ("mw", "Mw", "7.5"),
        ("wt", "Su tablası [m]", "2.0"),
        ("gamma", "γ [kN/m³]", "18"),
        ("gamma_sat", "γsat [kN/m³]", "20"),
        ("bore", "Sondaj (z N60 ince%):",
         "4 10 12\n6 9 20\n8 12 8\n10 15 25\n12 20 10\n14 25 5", "text"),
    ]

    def run(self, p) -> str:
        a_max = p["a_max"] * 9.81
        z_list, fs_list, liq_list = [], [], []
        for line in str(p["bore"]).splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            z, n60, fc = float(parts[0]), float(parts[1]), float(parts[2])
            sv0 = p["gamma"] * min(z, p["wt"]) + p["gamma_sat"] * max(
                0, z - p["wt"])
            sve = max(sv0 - GAMMA_W * max(0, z - p["wt"]), 1.0)
            if z <= p["wt"]:
                continue
            n1 = corrected_n1_60(n60, sve)
            res = liquefaction_factor_of_safety(n1, fc, a_max, p["mw"], sv0,
                                                sve, z)
            z_list.append(z)
            fs_list.append(min(res.factor_of_safety, 3.0))
            liq_list.append(res.liquefiable)
        z = np.array(z_list)
        fs = np.array(fs_list)
        liq = np.array(liq_list)
        w = np.maximum(0, 10 - 0.5 * z)
        lpi = float(np.trapz(np.maximum(0, 1 - fs) * w, z)) if z.size > 1 else 0
        ax = self.figure.add_subplot(111)
        ax.plot(fs, z, "ko-", lw=1.4, ms=4)
        ax.axvline(1.0, color="C3", lw=1.2, ls="--")
        ax.fill_betweenx(z, 0, 1.0, color="C3", alpha=0.12)
        if liq.any():
            ax.scatter(fs[liq], z[liq], color="C3", s=45, zorder=5)
        ax.set_xlabel("Güvenlik sayısı, FS")
        ax.set_ylabel("Derinlik (m)")
        ax.set_ylim(z.max() + 1, 0)
        ax.set_xlim(0, 3)
        ax.set_title("Sıvılaşma FS profili", fontsize=9)
        n_liq = int(liq.sum())
        return (f"LPI = {lpi:.1f}\n"
                f"Sıvılaşan tabaka sayısı: {n_liq}/{len(z)}\n"
                f"{'RİSK VAR' if lpi > 5 else 'düşük risk'}")


# ==========================================================================
# Application shell
# ==========================================================================
_TABS = [ClassificationTab, StressTab, ShallowFoundationTab, PileTab,
         SlopeTab, ConsolidationTab, LiquefactionTab]


def main() -> None:
    root = tk.Tk()
    root.title("pyGeotech Suite — Geoteknik Analiz Arayüzü")
    root.geometry("1000x640")
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    tabs = []
    for tab_cls in _TABS:
        tab = tab_cls(notebook)
        notebook.add(tab, text=tab_cls.title)
        tabs.append(tab)

    def on_tab_changed(_event):
        tab = tabs[notebook.index(notebook.select())]
        if not tab._computed:
            tab.calculate()

    notebook.bind("<<NotebookTabChanged>>", on_tab_changed)
    tabs[0].calculate()          # compute the first tab on startup
    root.mainloop()


if __name__ == "__main__":
    main()
