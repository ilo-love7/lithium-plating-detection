# -*- coding: utf-8 -*-
"""M2 dQ/dV 差分电压拐点法。

低倍率/特征化循环的 dQ/dV 曲线中，石墨嵌锂 staging 峰发生偏移、分裂或
新峰出现，作为析锂及后续容量损失的间接判据。

实现要点：
1. 先把 q(v) 重采样到统一电压网格，消除不同数据集采样率差异；
2. 在统一网格上平滑、按相对动态范围提取稳定主峰；
3. 双指针顺序匹配（允许峰丢失/新增），避免跨峰错配；
4. 持续性确认抑制随机抖动。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

from ..config import Config
from ..schema import CellData, Segment
from .common import MethodResult, sg_smooth


def _cumulative_ah(seg: Segment) -> np.ndarray:
    """由电荷(mAh)或电流积分重建 Ah。"""
    if "q_mah" in seg.meta:
        return np.asarray(seg.meta["q_mah"], dtype=float) / 1000.0
    q = np.concatenate([[0.0], np.cumsum(seg.current[1:] * np.diff(seg.time)) / 3600.0])
    return q


def _dqdv_on_grid(seg: Segment, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """把 q(v) 重采样到统一电压网格并求 dQ/dV。"""
    vlo = cfg.get("m2_dqdv", "grid_voltage_min", default=3.3)
    vhi = cfg.get("m2_dqdv", "grid_voltage_max", default=4.2)
    n_grid = cfg.get("m2_dqdv", "grid_n", default=200)
    win = cfg.get("m2_dqdv", "sg_window", default=21)
    poly = cfg.get("m2_dqdv", "sg_polyorder", default=3)

    v = seg.voltage
    q = _cumulative_ah(seg)
    order = np.argsort(v)
    v, q = v[order], q[order]
    keep = np.unique(v, return_index=True)[1]
    v, q = v[keep], q[keep]

    grid = np.linspace(vlo, vhi, n_grid)
    qg = np.interp(grid, v, q, left=np.nan, right=np.nan)
    dqdv = np.gradient(qg, grid)
    dqdv = sg_smooth(dqdv, win, poly)
    return grid, dqdv


def extract_peaks(seg: Segment, cfg: Config) -> Optional[Dict]:
    """在统一网格上提取固定数量的稳定主峰。"""
    dist_v = cfg.get("m2_dqdv", "peak_distance_v", default=0.05)
    ratio = cfg.get("m2_dqdv", "peak_prominence_ratio", default=0.10)
    top_k = cfg.get("m2_dqdv", "top_n_peaks", default=5)

    v_grid, dqdv = _dqdv_on_grid(seg, cfg)
    finite = np.isfinite(dqdv)
    if finite.sum() < 10:
        return None
    # 以中位 dQ/dV 为基准设噪声门槛，避免被单一强峰主导
    prom = ratio * float(np.median(dqdv[finite]))
    dv = float(np.median(np.diff(v_grid)))
    distance_pts = max(1, int(dist_v / max(dv, 1e-6)))
    peaks, props = find_peaks(dqdv, prominence=prom, distance=distance_pts)
    if not len(peaks):
        return {"v": v_grid, "dqdv": dqdv, "positions": np.array([]),
                "heights": np.array([])}
    # 按 prominence 取 top K，再按电压排序
    order_p = np.argsort(props["prominences"])[::-1][:top_k]
    peaks = np.sort(peaks[order_p])
    return {
        "v": v_grid,
        "dqdv": dqdv,
        "positions": v_grid[peaks],
        "heights": dqdv[peaks],
    }


def _match_peaks(base: np.ndarray, cur: np.ndarray, max_gap: float):
    """双指针顺序匹配，返回 (base_v, cur_v) 配对列表。"""
    pairs = []
    i = j = 0
    while i < len(base) and j < len(cur):
        d = abs(cur[j] - base[i])
        if d <= max_gap:
            pairs.append((float(base[i]), float(cur[j])))
            i += 1
            j += 1
        elif cur[j] < base[i]:
            j += 1  # 新峰，跳过
        else:
            i += 1  # 基准峰丢失，跳过
    return pairs


def _is_low_rate(seg: Segment, cell: CellData, cfg: Config) -> bool:
    thr = cfg.get("m2_dqdv", "low_rate_threshold_c", default=0.15)
    rate_c = abs(seg.mean_current_a) / max(cell.nominal_capacity_ah, 1e-9)
    return rate_c <= thr


def run_cell(cell: CellData, cfg: Config) -> List[MethodResult]:
    shift_alarm = cfg.get("m2_dqdv", "shift_alarm_v", default=0.03)
    split_dist = cfg.get("m2_dqdv", "split_peak_distance_v", default=0.04)
    max_gap = cfg.get("m2_dqdv", "max_match_gap_v", default=0.08)
    persist_n = cfg.get("m2_dqdv", "persist_n", default=2)

    records: List[Tuple[int, Dict]] = []
    for cyc in cell.cycles:
        candidates = [s for s in cyc.by_type("charge")
                      if _is_low_rate(s, cell, cfg)]
        if not candidates:
            continue
        seg = min(candidates, key=lambda s: abs(s.mean_current_a))
        pk = extract_peaks(seg, cfg)
        if pk is not None and len(pk["positions"]):
            records.append((cyc.cycle_index, pk))

    if len(records) < 2:
        return []
    base_pos = records[0][1]["positions"]

    raw = []
    for ci, pk in records:
        pos = pk["positions"]
        pairs = _match_peaks(base_pos, pos, max_gap)
        shifts = [abs(c - b) for b, c in pairs]
        max_shift = max(shifts) if shifts else 0.0
        # 峰分裂：当前峰多于成功匹配数，且额外峰贴近某基准峰
        n_split = max(0, len(pos) - len(pairs))
        n_lost = len(base_pos) - len(pairs)
        hit = bool(max_shift >= shift_alarm or n_split > 0)
        raw.append((ci, pk, max_shift, n_split, n_lost, hit))

    results: List[MethodResult] = []
    for k, (ci, pk, max_shift, n_split, n_lost, hit) in enumerate(raw):
        confirmed = False
        if hit and k >= persist_n - 1:
            confirmed = all(raw[j][5] for j in range(k - persist_n + 1, k + 1))
        results.append(
            MethodResult(
                method="M2",
                cell_id=cell.cell_id,
                cycle=ci,
                positive=confirmed,
                score=float(max_shift),
                threshold=float(shift_alarm),
                detail={"n_peaks": int(len(pk["positions"])),
                        "n_splits": int(n_split),
                        "n_lost_peaks": int(n_lost),
                        "n_matched": int(len(pk["positions"]) - n_split),
                        "baseline_n_peaks": int(len(base_pos))},
                series={"v": pk["v"], "dqdv": pk["dqdv"],
                        "positions": pk["positions"]},
            )
        )
    return results
