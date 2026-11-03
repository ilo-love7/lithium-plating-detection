# -*- coding: utf-8 -*-
"""
退化对齐分析：以容量保持率（SOH）而非循环索引对齐，
量化各方法在退化轨迹上的报警时机（相对析锂 onset 的滞后与当时 SOH）。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import Config
from .methods import run_all, METHOD_NAMES
from .synthetic import ground_truth


def capacity_hold_curve(cell) -> Dict[int, float]:
    """返回 {cycle_index: 容量保持率}，以首循环容量为基准。"""
    pairs = [(c.cycle_index, c.rated_capacity_ah) for c in cell.cycles]
    pairs = [(ci, cap) for ci, cap in pairs if cap is not None]
    if not pairs:
        return {}
    base = pairs[0][1]
    return {ci: float(cap / base) for ci, cap in pairs}


def _first_alarm_cycle(cell, cfg: Config, method: str) -> Optional[int]:
    res = run_all(cell, cfg)[method]
    pos = [r.cycle for r in res if r.positive]
    return min(pos) if pos else None


def alignment_table(cfg: Config, cells) -> pd.DataFrame:
    """对析锂电芯逐方法计算报警循环、滞后与报警时 SOH。"""
    gt = ground_truth(cells)
    rows = []
    for cell in cells:
        if not gt[cell.cell_id]["plating"]:
            continue
        soh_map = capacity_hold_curve(cell)
        onset = int(cell.meta.get("plating_onset", -1))
        for m in METHOD_NAMES:
            alarm = _first_alarm_cycle(cell, cfg, m)
            rows.append({
                "cell_id": cell.cell_id,
                "method": m,
                "plating_onset_cycle": onset,
                "alarm_cycle": alarm,
                "detection_lag_cycles": (alarm - onset) if alarm is not None else None,
                "soh_at_alarm": soh_map.get(alarm) if alarm is not None else None,
            })
    return pd.DataFrame(rows)


def summarize_alignment(table: pd.DataFrame) -> pd.DataFrame:
    """按方法汇总报警滞后与 SOH 的中位数/均值。"""
    g = table.groupby("method")
    return pd.DataFrame({
        "n_alarms": g["alarm_cycle"].count(),
        "lag_cycles_median": g["detection_lag_cycles"].median(),
        "lag_cycles_mean": g["detection_lag_cycles"].mean(),
        "soh_at_alarm_median": g["soh_at_alarm"].median(),
        "soh_at_alarm_min": g["soh_at_alarm"].min(),
    }).reset_index()


def plot_alignment(table: pd.DataFrame, fig_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左：各方法报警时 SOH 分布
    data_soh = [table[table["method"] == m]["soh_at_alarm"].dropna()
                for m in METHOD_NAMES]
    axes[0].boxplot(data_soh, tick_labels=METHOD_NAMES, showmeans=True)
    for k, m in enumerate(METHOD_NAMES):
        y = table[table["method"] == m]["soh_at_alarm"].dropna()
        axes[0].scatter(np.full(len(y), k + 1) + np.random.default_rng(k).uniform(
            -0.08, 0.08, len(y)), y, alpha=0.6, s=22)
    axes[0].set_ylabel("Capacity retention (SOH) at first alarm")
    axes[0].set_title("Detection timing aligned by SOH")
    axes[0].grid(alpha=0.3, axis="y")

    # 右：检测滞后循环数
    data_lag = [table[table["method"] == m]["detection_lag_cycles"].dropna()
                for m in METHOD_NAMES]
    axes[1].boxplot(data_lag, tick_labels=METHOD_NAMES, showmeans=True)
    axes[1].axhline(0, color="r", linestyle="--", linewidth=1,
                    label="Plating onset")
    axes[1].set_ylabel("Detection lag (cycles after onset)")
    axes[1].set_title("Detection lag vs. plating onset")
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
