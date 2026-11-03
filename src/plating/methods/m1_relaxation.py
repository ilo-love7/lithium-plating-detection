# -*- coding: utf-8 -*-
"""M1 电压弛豫平台法（单调基线残差）。

充电结束后静置段，若发生析锂，析出锂重新嵌回的混合电位效应使弛豫速率出现
特征肩部。弛豫速率本身随时间单调指数衰减，直接找局部极小会被强下降趋势
淹没。因此：
1. SG 平滑后求弛豫速率 rate = -dV/dt；
2. 用保序回归拟合单调弛豫基线；
3. 在残差 rate - baseline 上检测局部峰（肩部），prominence 超阈值即判阳性。
"""
from __future__ import annotations

from typing import List

import numpy as np
from scipy.signal import find_peaks
from sklearn.isotonic import IsotonicRegression

from ..config import Config
from ..schema import CellData, Segment
from .common import MethodResult, sg_smooth


def detect(seg: Segment, cell_id: str, cycle: int, cfg: Config) -> MethodResult:
    window = cfg.get("m1_relaxation", "sg_window", default=11)
    poly = cfg.get("m1_relaxation", "sg_polyorder", default=3)
    prominence = cfg.get("m1_relaxation", "residual_prominence", default=4e-6)
    min_used = cfg.get("m1_relaxation", "min_relax_used_s", default=600)
    search_frac = cfg.get("m1_relaxation", "shoulder_search_frac", default=0.8)

    t, v = seg.time, seg.voltage
    if len(t) < 8 or seg.duration_s < min_used:
        return _empty(cell_id, cycle, prominence, reason="too_short")

    vs = sg_smooth(v, window, poly)
    rate = -np.gradient(vs, t)
    # 单调递减基线（保序回归）
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    baseline = iso.fit_transform(t, rate)
    residual = rate - baseline

    t_limit = (t[-1] - t[0]) * search_frac
    search = (t - t[0]) <= t_limit
    peaks, props = find_peaks(residual[search], prominence=prominence)
    best_prom = float(np.max(props["prominences"])) if len(peaks) else 0.0
    positive = bool(len(peaks) > 0)
    feature_t = None
    if len(peaks):
        j = int(np.argmax(props["prominences"]))
        search_idx = np.where(search)[0]
        feature_t = float(t[search_idx[peaks[j]]] - t[0])

    return MethodResult(
        method="M1",
        cell_id=cell_id,
        cycle=cycle,
        positive=positive,
        score=best_prom,
        threshold=float(prominence),
        feature_time_s=feature_t,
        detail={"n_shoulders": int(len(peaks)), "n_points": len(t)},
        series={"t": t - t[0], "v": vs, "rate": rate,
                "baseline": baseline, "residual": residual},
    )


def run_cell(cell: CellData, cfg: Config) -> List[MethodResult]:
    persist_n = cfg.get("m1_relaxation", "persist_n", default=2)
    raw: List[MethodResult] = []
    for cyc in cell.cycles:
        for seg in cyc.by_type("rest"):
            raw.append(detect(seg, cell.cell_id, cyc.cycle_index, cfg))
    # 基于原始标志独立判断，避免原地修改造成连锁归零
    raw_flags = [r.positive for r in raw]
    for k, r in enumerate(raw):
        confirmed = False
        if raw_flags[k] and k >= persist_n - 1:
            confirmed = all(raw_flags[j] for j in range(k - persist_n + 1, k + 1))
        r.positive = confirmed
    return raw


def _empty(cell_id, cycle, threshold, reason: str) -> MethodResult:
    return MethodResult(
        method="M1", cell_id=cell_id, cycle=cycle, positive=False,
        score=0.0, threshold=float(threshold), detail={"reason": reason},
    )
