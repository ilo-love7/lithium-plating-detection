# -*- coding: utf-8 -*-
"""合成闭环 ROC 曲线 / AUC：在连续 score 上扫描阈值，量化方法区分能力。"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .config import Config
from .synthetic import generate_dataset, ground_truth
from .methods import run_all, METHOD_NAMES


def collect_cell_scores(cfg: Config, cells, method: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    对每个电芯取该方法在所有循环上 score 的最大值作为电芯级连续响应，
    并返回对应真值标签（True=析锂）。
    """
    gt = ground_truth(cells)
    scores: List[float] = []
    labels: List[bool] = []
    for cell in cells:
        res = run_all(cell, cfg)[method]
        scores.append(max((r.score for r in res), default=0.0))
        labels.append(bool(gt[cell.cell_id]["plating"]))
    return np.asarray(scores), np.asarray(labels, dtype=bool)


def roc_curve(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """从高到低扫描阈值，返回单调不减的 (fpr, tpr)。"""
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    # 降序阈值：从 inf（全部判负）到 -inf（全部判正）
    thresholds = np.unique(np.concatenate([[np.inf], np.sort(scores), [-np.inf]]))[::-1]
    tpr = np.empty(len(thresholds))
    fpr = np.empty(len(thresholds))
    for k, th in enumerate(thresholds):
        pred = scores >= th
        tp = int(np.sum(pred & labels))
        fp = int(np.sum(pred & (~labels)))
        tpr[k] = tp / max(n_pos, 1)
        fpr[k] = fp / max(n_neg, 1)
    return fpr, tpr


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """ROC 曲线下面积（梯形积分）。"""
    order = np.argsort(fpr)
    return float(np.trapezoid(tpr[order], fpr[order]))


def build_roc_table(cfg: Config, n_cells: int = 24, n_cycles: int = 140) -> Dict:
    """生成合成数据集并计算四方法 ROC/AUC。"""
    cells = generate_dataset(cfg, n_cells=n_cells, n_cycles=n_cycles)
    out = {}
    for m in METHOD_NAMES:
        scores, labels = collect_cell_scores(cfg, cells, m)
        fpr, tpr = roc_curve(scores, labels)
        out[m] = {
            "fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr),
            "scores": scores, "labels": labels,
        }
    return out


def plot_roc(table: Dict, fig_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="Chance (AUC=0.5)")
    for m in METHOD_NAMES:
        d = table[m]
        ax.plot(d["fpr"], d["tpr"], marker="o", markersize=4,
                linewidth=1.6, label=f"{m} (AUC={d['auc']:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Synthetic Closed-Loop ROC (cell-level max score)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
