# -*- coding: utf-8 -*-
"""
合成闭环 ROC 分析：生成较大合成数据集，计算四方法 ROC/AUC，
输出 ROC 图、AUC 汇总 CSV 与电芯级分数 CSV。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import numpy as np
import pandas as pd

from plating.config import load_config
from plating.roc import build_roc_table, plot_roc
from plating.methods import METHOD_NAMES


def main(n_cells=24, n_cycles=140):
    cfg = load_config()
    table = build_roc_table(cfg, n_cells=n_cells, n_cycles=n_cycles)

    fig_dir = "results/figures"
    os.makedirs(fig_dir, exist_ok=True)
    plot_roc(table, os.path.join(fig_dir, "fig5_synthetic_roc.png"))

    # AUC 汇总
    summary = pd.DataFrame([
        {"method": m, "auc": table[m]["auc"],
         "n_positive": int(table[m]["labels"].sum()),
         "n_negative": int((~table[m]["labels"]).sum())}
        for m in METHOD_NAMES
    ])
    summary.to_csv("results/roc_auc_summary.csv", index=False, encoding="utf-8-sig")

    # 电芯级分数（含真值），便于复核
    base = table[METHOD_NAMES[0]]
    rows = pd.DataFrame({"cell": [f"cell_{i}" for i in range(len(base["labels"]))],
                         "truth": base["labels"]})
    for m in METHOD_NAMES:
        rows[f"score_{m}"] = table[m]["scores"]
    rows.to_csv("results/roc_cell_scores.csv", index=False, encoding="utf-8-sig")

    print(summary.to_string(index=False))
    print("\n正/负样本 score 范围：")
    for m in METHOD_NAMES:
        s, y = table[m]["scores"], table[m]["labels"]
        print(f"{m}: 阳性[{s[y].min():.4g},{s[y].max():.4g}] "
              f"对照[{s[~y].min():.4g},{s[~y].max():.4g}]")


if __name__ == "__main__":
    main()
