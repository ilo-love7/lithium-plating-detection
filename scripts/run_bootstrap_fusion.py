# -*- coding: utf-8 -*-
"""
科研深化：AUC 的 bootstrap 置信区间 + 多方法多数投票融合。
输出 bootstrap AUC 汇总、融合规则 CSV 与融合性能图。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import pandas as pd

from plating.config import load_config
from plating.roc import build_roc_table, bootstrap_auc
from plating.ensemble import build_fusion_table, plot_fusion_rules
from plating.robustness import run_noise_robustness, plot_noise_robustness
from plating.methods import METHOD_NAMES


def main(n_cells=24, n_cycles=140):
    cfg = load_config()
    n_boot = cfg.get("statistics", "bootstrap_n", default=1000)
    confidence = cfg.get("statistics", "confidence", default=0.95)

    # 1) ROC + bootstrap AUC
    table = build_roc_table(cfg, n_cells=n_cells, n_cycles=n_cycles)
    boot_rows = []
    for m in METHOD_NAMES:
        d = table[m]
        mean, lo, hi, _ = bootstrap_auc(
            d["scores"], d["labels"], n_boot=n_boot, confidence=confidence)
        boot_rows.append({"method": m, "auc_point": d["auc"],
                          "auc_boot_mean": mean, "ci_lower": lo, "ci_upper": hi})
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv("results/roc_auc_bootstrap.csv", index=False,
                   encoding="utf-8-sig")

    # 2) 多方法融合投票
    fusion = build_fusion_table(cfg, n_cells=n_cells, n_cycles=n_cycles)
    rules_df = pd.DataFrame(fusion["rules"])
    rules_df.to_csv("results/fusion_vote_rules.csv", index=False,
                    encoding="utf-8-sig")

    fig_dir = "results/figures"
    os.makedirs(fig_dir, exist_ok=True)
    plot_fusion_rules(fusion, os.path.join(fig_dir, "fig11_fusion_votes.png"))

    # 3) 测量噪声鲁棒性
    robust = run_noise_robustness(cfg, n_boot=500, confidence=confidence)
    robust.to_csv("results/noise_robustness.csv", index=False,
                  encoding="utf-8-sig")
    plot_noise_robustness(robust, os.path.join(
        fig_dir, "fig12_noise_robustness.png"))

    print("=== AUC bootstrap 置信区间 ===")
    print(boot_df.to_string(index=False))
    print("\n=== 融合投票规则 ===")
    print(rules_df.to_string(index=False))
    print("\n=== 噪声鲁棒性（末段） ===")
    print(robust.tail(6).to_string(index=False))


if __name__ == "__main__":
    main()
