# -*- coding: utf-8 -*-
"""生成项目①报告配图（matplotlib PNG）。"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from plating.config import load_config
from plating.synthetic import SyntheticBattery, _PlatingSpec
from plating.analysis import load_all_cells
from plating.methods import run_all
from plating.methods.m1_relaxation import detect as m1_detect

FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})


def fig_synthetic_features(cfg):
    """2x2 合成析锂电芯四方法特征。"""
    b = SyntheticBattery(cfg, "Synth_08", _PlatingSpec(True, 80))
    cell = b.build(150, 3600)
    res = run_all(cell, cfg)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    # M1: 残差肩部
    seg = cell.cycles[105].by_type("rest")[0]
    r = m1_detect(seg, cell.cell_id, 105, cfg)
    t = r.series["t"]
    ax[0, 0].plot(t, r.series["residual"], color="#2b6cb0")
    ax[0, 0].set_title("M1: Relaxation shoulder (isotonic residual)")
    ax[0, 0].set_xlabel("Rest time (s)"); ax[0, 0].set_ylabel("Rate residual (V/s)")

    # M2: dQ/dV 峰
    m2r = [x for x in res["M2"] if x.positive][0]
    ax[0, 1].plot(m2r.series["v"], m2r.series["dqdv"], color="#c05621")
    ax[0, 1].plot(m2r.series["positions"], m2r.series["positions"] * 0 +
                  np.interp(m2r.series["positions"], m2r.series["v"],
                            m2r.series["dqdv"]), "k*", ms=10)
    ax[0, 1].set_title("M2: dQ/dV peak tracking")
    ax[0, 1].set_xlabel("Voltage (V)"); ax[0, 1].set_ylabel("dQ/dV (Ah/V)")

    # M3: CE 轨迹
    m3 = res["M3"]
    ce = [x.detail["ce"] for x in m3]
    ax[1, 0].plot([x.cycle for x in m3], ce, color="#2f855a")
    ax[1, 0].set_title("M3: Coulomb efficiency decline")
    ax[1, 0].set_xlabel("Cycle"); ax[1, 0].set_ylabel("CE")

    # M4: 自行重算 CC 段基准与 aged 对照
    from plating.methods.m4_kink import _cc_charge_curve
    grid = np.linspace(0.10, 0.95, 100)
    base_curves, aged = [], None
    for cyc in cell.cycles:
        out = _cc_charge_curve(cyc.by_type("charge")[0], cfg)
        if out is None:
            continue
        s, d = out
        rg = np.interp(grid, s, d)
        if cyc.cycle_index < 70:
            base_curves.append(rg)
        elif cyc.cycle_index == 105:
            aged = rg
    ax[1, 1].plot(grid, aged, color="#6b46c1", label="aged (cycle 105)")
    ax[1, 1].plot(grid, np.median(np.vstack(base_curves), axis=0), color="gray",
                  ls="--", label="baseline median")
    ax[1, 1].set_title("M4: Charge kink vs baseline")
    ax[1, 1].set_xlabel("SOC"); ax[1, 1].set_ylabel("dV/dQ")
    ax[1, 1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_synthetic_features.png"))
    plt.close(fig)


def fig_real_heatmap(cfg, cells, results):
    cell_tbl = pd.read_csv(os.path.join(ROOT, "results", "cell_level_summary.csv"))
    methods = ["M1", "M2", "M3", "M4"]
    mat = np.zeros((len(methods), len(cell_tbl)), dtype=int)
    for i, m in enumerate(methods):
        mat[i] = cell_tbl[f"{m}_positive"].astype(int)
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.imshow(mat, cmap=ListedColormap(["#f7fafc", "#e53e3e"]), aspect="auto")
    ax.set_xticks(range(len(cell_tbl)))
    ax.set_xticklabels(cell_tbl["cell_id"], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j]:
                ax.text(j, i, "1", ha="center", va="center", fontsize=8)
    ax.set_title("Real-data cell-level positive flags (16 cells)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_real_positive_heatmap.png"))
    plt.close(fig)


def fig_threshold(cfg):
    tbl = pd.read_csv(os.path.join(ROOT, "results", "threshold_sensitivity.csv"))
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, c in zip(["M1", "M2", "M3", "M4"],
                    ["#2b6cb0", "#c05621", "#2f855a", "#6b46c1"]):
        sub = tbl[tbl.method == m]
        ax.plot(sub.multiplier, sub.n_positive_cells, "-o", label=m, color=c)
    ax.axvline(1.0, color="k", ls=":", alpha=0.6)
    ax.set_xlabel("Threshold multiplier (baseline = 1.0)")
    ax.set_ylabel("Number of positive cells")
    ax.set_title("Threshold sensitivity on real data")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_threshold_sensitivity.png"))
    plt.close(fig)


def fig_m2_tracking(cfg, cells, results):
    # Oxford_Cell1 峰位置随循环
    cell = [c for c in cells if c.cell_id == "Oxford_Cell1"][0]
    from plating.methods.m2_dqdv import run_cell
    res = run_cell(cell, cfg)
    fig, ax = plt.subplots(figsize=(9, 5))
    cycs = [r.cycle for r in res]
    n_pk = [r.detail["n_peaks"] for r in res]
    ax.plot(cycs, n_pk, "-o", color="#c05621", ms=3)
    ax.set_xlabel("Characterisation cycle index")
    ax.set_ylabel("Number of tracked peaks")
    ax.set_title("M2 peak count over aging (Oxford_Cell1)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_m2_peak_count.png"))
    plt.close(fig)


def main():
    cfg = load_config()
    print("合成特征图 ...")
    fig_synthetic_features(cfg)
    print("加载真实电芯 ...")
    cells = load_all_cells(cfg, 5)
    results = {c.cell_id: run_all(c, cfg) for c in cells}
    fig_real_heatmap(cfg, cells, results)
    fig_threshold(cfg)
    fig_m2_tracking(cfg, cells, results)
    print("完成，图片位于", FIG)
    for f in sorted(os.listdir(FIG)):
        print(" ", f)


if __name__ == "__main__":
    main()
