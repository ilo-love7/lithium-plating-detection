# -*- coding: utf-8 -*-
"""
测量噪声鲁棒性：在不同电压噪声水平下量化各方法 AUC（含 bootstrap 区间）
与多数投票融合的灵敏度/特异度，界定方法的稳健性边界。零训练。
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .config import Config
from .synthetic import generate_dataset
from .roc import collect_cell_scores, roc_curve, auc, bootstrap_auc
from .ensemble import collect_votes, fusion_metrics
from .methods import METHOD_NAMES


def config_with_noise(base_cfg: Config, noise: float) -> Config:
    """返回仅替换 voltage noise 的配置副本。"""
    data = base_cfg.as_dict()
    data["synthetic"] = dict(data["synthetic"])
    data["synthetic"]["noise_std_v"] = noise
    return Config(data)


def run_noise_robustness(base_cfg: Config, n_cells: int = 24, n_cycles: int = 140,
                         n_boot: int = 500, confidence: float = 0.95,
                         fusion_rule: int = 3) -> pd.DataFrame:
    levels: List[float] = list(
        base_cfg.get("synthetic", "robustness_noise_levels"))
    rows = []
    for noise in levels:
        cfg = config_with_noise(base_cfg, noise)
        cells = generate_dataset(cfg, n_cells=n_cells, n_cycles=n_cycles)
        for m in METHOD_NAMES:
            scores, labels = collect_cell_scores(cfg, cells, m)
            fpr, tpr = roc_curve(scores, labels)
            point_auc = auc(fpr, tpr)
            _, lo, hi, _ = bootstrap_auc(scores, labels, n_boot=n_boot,
                                         confidence=confidence)
            rows.append({"noise_std_v": noise, "method": m, "auc": point_auc,
                         "ci_lower": lo, "ci_upper": hi,
                         "sensitivity": np.nan, "specificity": np.nan})
        votes, labels, _ = collect_votes(cfg, cells)
        fm = fusion_metrics(votes, labels, fusion_rule)
        rows.append({"noise_std_v": noise, "method": f"FUSION{fusion_rule}",
                     "auc": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
                     "sensitivity": fm["sensitivity"],
                     "specificity": fm["specificity"]})
    return pd.DataFrame(rows)


def plot_noise_robustness(df: pd.DataFrame, fig_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    for m in METHOD_NAMES:
        sub = df[df["method"] == m].sort_values("noise_std_v")
        ax.plot(sub["noise_std_v"], sub["auc"], marker="o", linewidth=1.6,
                label=m)
        ax.fill_between(sub["noise_std_v"], sub["ci_lower"], sub["ci_upper"],
                        alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("Voltage measurement noise std [V] (log scale)")
    ax.set_ylabel("AUC (shaded: bootstrap 95% CI)")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("AUC robustness to measurement noise")

    ax2 = axes[1]
    fus = df[df["method"].str.startswith("FUSION")].sort_values("noise_std_v")
    ax2.plot(fus["noise_std_v"], fus["sensitivity"], marker="o",
             linewidth=1.8, label="Sensitivity")
    ax2.plot(fus["noise_std_v"], fus["specificity"], marker="s",
             linewidth=1.8, label="Specificity")
    ax2.set_xscale("log")
    ax2.set_xlabel("Voltage measurement noise std [V] (log scale)")
    ax2.set_ylim(-0.03, 1.05)
    ax2.grid(alpha=0.3)
    ax2.legend()
    ax2.set_title("Majority-vote fusion (>=3 methods) robustness")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
