# -*- coding: utf-8 -*-
"""
强析锂工况验证：在 4 个工况（基线/低温/高倍率/极端低温高倍率）下生成合成数据，
计算四方法的灵敏度/特异度与报警时机，输出汇总表与配图。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import pandas as pd

from plating.config import load_config
from plating.synthetic import generate_stress_dataset, ground_truth
from plating.methods import run_all, METHOD_NAMES
from plating.degradation import capacity_hold_curve


def main(n_cells=8, n_cycles=120):
    cfg = load_config()
    scenarios = list(cfg.get("stress_scenarios").as_dict().keys())
    rows = []
    for sc in scenarios:
        cells = generate_stress_dataset(cfg, sc, n_cells=n_cells,
                                        n_cycles=n_cycles)
        gt = ground_truth(cells)
        for m in METHOD_NAMES:
            tp = fn = fp = tn = 0
            alarm_cycles = []
            for cell in cells:
                truth = gt[cell.cell_id]["plating"]
                res = run_all(cell, cfg)[m]
                pred = any(r.positive for r in res)
                if truth and pred:
                    tp += 1
                elif truth:
                    fn += 1
                elif pred:
                    fp += 1
                else:
                    tn += 1
                if truth and pred:
                    first = min(r.cycle for r in res if r.positive)
                    soh = capacity_hold_curve(cell).get(first)
                    alarm_cycles.append((first, soh))
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            med_soh = (pd.Series([s for _, s in alarm_cycles]).median()
                       if alarm_cycles else None)
            rows.append({
                "scenario": sc, "method": m,
                "sensitivity": round(sens, 3), "specificity": round(spec, 3),
                "n_plated": tp + fn, "n_control": tn + fp,
                "median_soh_at_alarm": med_soh,
            })

    df = pd.DataFrame(rows)
    df.to_csv("results/stress_scenario_validation.csv",
              index=False, encoding="utf-8-sig")
    plot_sensitivity(df, os.path.join("results", "figures",
                                      "fig7_stress_sensitivity.png"))
    print(df.to_string(index=False))


def plot_sensitivity(df, fig_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = df.pivot(index="method", columns="scenario",
                     values="sensitivity")
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", vmin=0, vmax=1,
                   aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns,
                  rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.to_numpy()[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="black", fontsize=11)
    ax.set_title("Sensitivity across stress scenarios (synthetic truth)")
    fig.colorbar(im, ax=ax, label="Sensitivity")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
