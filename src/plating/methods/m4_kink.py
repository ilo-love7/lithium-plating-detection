# -*- coding: utf-8 -*-
"""M4 充电电压 kink 拐点法（跨循环基准形态对照）。

原理：正常石墨 staging 在 dV/dQ 上本就存在显著局部极小，且位置/形态在健康
循环间稳定；析锂引起的 kink 是随老化在高 SOC 区域**新出现并增强**的凹陷。
因此不以绝对凹陷判定，而是：
1. 各循环 CC 充电段 dV/dQ 对齐重采样到统一 SOC 网格；
2. 以前若干健康循环的中位曲线为基准；
3. 检测后续循环相对基准在高 SOC 出现的显著负向凹陷（新 kink），并要求连续
   多个循环持续出现，以抑制抖动与 staging 混淆。
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from ..config import Config
from ..schema import CellData
from .common import MethodResult, sg_smooth, fill_nonfinite


def _cc_charge_curve(seg, cfg: Config):
    """返回单循环 CC 充电段的 (soc, dvdq)，数据不足返回 None。"""
    t, i, v = seg.time, seg.current, seg.voltage
    n = len(t)
    if n < 30:
        return None
    i_level = np.percentile(i, cfg.get("m4_kink", "cc_current_percentile",
                                       default=90))
    cc_frac = cfg.get("m4_kink", "cc_current_frac", default=0.85)
    cc = i >= cc_frac * i_level
    if cc.sum() < 30:
        return None
    t, i, v = t[cc], i[cc], v[cc]
    # 估计 SOC：以累计充电容量归一化
    dt = np.diff(t)
    dq = i[:-1] * dt / 3600.0
    q = np.concatenate([[0], np.cumsum(dq)])
    if q[-1] <= 0:
        return None
    soc = q / q[-1]
    order = np.argsort(soc)
    soc, v = soc[order], v[order]
    # 去重
    soc, idx = np.unique(soc, return_index=True)
    v = v[idx]
    if len(soc) < 30:
        return None
    win = min(cfg.get("m4_kink", "sg_window", default=101), len(v) // 2 * 2 - 1)
    if win < 7:
        win = 7
    v_s = sg_smooth(v, win, cfg.get("m4_kink", "sg_polyorder", default=3))
    dvdq = np.gradient(v_s, soc)
    return soc, dvdq


def detect(cell: CellData, cfg: Config) -> list:
    grid_n = cfg.get("m4_kink", "soc_grid_n", default=100)
    soc_min = cfg.get("m4_kink", "soc_min", default=0.10)
    soc_max = cfg.get("m4_kink", "soc_max", default=0.95)
    grid = np.linspace(soc_min, soc_max, grid_n)
    k_lo = cfg.get("m4_kink", "kink_min_soc", default=0.50)
    k_hi = cfg.get("m4_kink", "kink_max_soc", default=0.90)
    n_base = cfg.get("m4_kink", "baseline_cycles", default=10)
    min_prom = cfg.get("m4_kink", "min_prominence", default=0.02)
    rel_thr = cfg.get("m4_kink", "rel_prominence", default=0.30)
    persist_n = cfg.get("m4_kink", "persist_n", default=2)

    # 收集各循环重采样后的 dV/dQ 曲线
    curves = {}
    for cyc in cell.cycles:
        ch_list = cyc.by_type("charge")
        if not ch_list:
            continue
        ch = ch_list[0]
        out = _cc_charge_curve(ch, cfg)
        if out is None:
            continue
        soc, dvdq = out
        in_range = (soc >= soc_min) & (soc <= soc_max)
        if in_range.sum() < 20:
            continue
        curves[cyc.cycle_index] = np.interp(grid, soc[in_range],
                                            dvdq[in_range])
    if len(curves) < n_base + 2:
        return []

    cyc_ids = sorted(curves)
    base_ids = cyc_ids[:n_base]
    baseline = np.nanmedian(np.vstack([curves[c] for c in base_ids]), axis=0)
    baseline = fill_nonfinite(baseline)

    search = (grid >= k_lo) & (grid <= k_hi)
    edge = int(cfg.get("m4_kink", "edge_exclude_frac", default=0.05) * search.sum())
    search_idx = np.where(search)[0]

    # 先逐循环判定，再做持续性确认
    raw_flag = {}
    for cid in cyc_ids:
        if cid in base_ids:
            raw_flag[cid] = (False, None, 0.0)
            continue
        diff = fill_nonfinite(curves[cid]) - baseline
        # 新凹陷：diff 为负的局部峰（在 -diff 上找峰）
        seg = -diff[search_idx]
        peaks, props = find_peaks(seg, prominence=min_prom)
        hit = False
        best_soc = None
        best_rel = 0.0
        for p in peaks:
            if p < edge or p > len(seg) - edge:
                continue
            gi = search_idx[p]
            local_base = abs(np.median(baseline[max(0, gi - 5):gi + 6]))
            rel = props["prominences"][list(peaks).index(p)] / max(local_base, 1e-9)
            if rel > rel_thr and rel > best_rel:
                hit, best_soc, best_rel = True, grid[gi], rel
        raw_flag[cid] = (hit, best_soc, best_rel)

    # 持续性确认：当前及前 persist_n-1 个循环均命中
    results = []
    for k, cid in enumerate(cyc_ids):
        hit, soc_at, rel = raw_flag[cid]
        if hit and k >= persist_n - 1:
            window = cyc_ids[k - persist_n + 1:k + 1]
            if all(raw_flag[w][0] for w in window):
                results.append(MethodResult(
                    method="M4", cell_id=cell.cell_id, cycle=cid,
                    positive=True, score=float(rel), threshold=float(rel_thr),
                    detail={"kink_soc": float(soc_at),
                            "rel_prominence_vs_baseline": float(rel)},
                ))
    return results


def run_cell(cell: CellData, cfg: Config) -> list:
    return detect(cell, cfg)
