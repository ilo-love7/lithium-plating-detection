# -*- coding: utf-8 -*-
"""
多方法融合决策：对 M1–M4 的电芯级二值判定做多数投票，量化融合性能。
零训练、零参数拟合，投票规则仅为可解释的计数阈值。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .config import Config
from .synthetic import generate_dataset, ground_truth
from .methods import run_all, METHOD_NAMES


def collect_votes(cfg: Config, cells):
    """
    返回每个电芯的方法投票矩阵（方法在任一循环阳性即投该方法一票）、
    真值标签与电芯 id。
    """
    gt = ground_truth(cells)
    cell_ids: List[str] = []
    labels: List[bool] = []
    votes = []
    for cell in cells:
        res = run_all(cell, cfg)
        votes.append([any(r.positive for r in res[m]) for m in METHOD_NAMES])
        cell_ids.append(cell.cell_id)
        labels.append(bool(gt[cell.cell_id]["plating"]))
    return (np.asarray(votes, dtype=bool), np.asarray(labels, dtype=bool),
            cell_ids)


def fusion_metrics(votes: np.ndarray, labels: np.ndarray,
                   min_votes: int) -> Dict:
    """阳性方法数 >= min_votes 即融合判正，返回混淆矩阵与灵敏度/特异度。"""
    n_positive = votes.sum(axis=1)
    pred = n_positive >= min_votes
    tp = int((pred & labels).sum())
    fn = int((~pred & labels).sum())
    fp = int((pred & ~labels).sum())
    tn = int((~pred & ~labels).sum())
    return {
        "min_votes": min_votes, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
    }


def build_fusion_table(cfg: Config, n_cells: int = 24,
                       n_cycles: int = 140) -> Dict:
    """生成合成数据并扫描 1..M 投票规则的融合性能。"""
    cells = generate_dataset(cfg, n_cells=n_cells, n_cycles=n_cycles)
    votes, labels, cell_ids = collect_votes(cfg, cells)
    rules = [fusion_metrics(votes, labels, k)
             for k in range(1, len(METHOD_NAMES) + 1)]
    return {"votes": votes, "labels": labels, "cell_ids": cell_ids,
            "rules": rules}


def plot_fusion_rules(table: Dict, fig_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rules = table["rules"]
    x = np.arange(len(rules))
    sens = [r["sensitivity"] for r in rules]
    spec = [r["specificity"] for r in rules]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(x, sens, marker="o", linewidth=1.8, label="Sensitivity")
    ax.plot(x, spec, marker="s", linewidth=1.8, label="Specificity")
    ax.set_xticks(x)
    ax.set_xticklabels([f"≥{r['min_votes']} method(s)" for r in rules])
    ax.set_xlabel("Fusion rule (minimum number of positive methods)")
    ax.set_ylabel("Metric")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Majority-vote fusion performance (synthetic closed-loop)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
