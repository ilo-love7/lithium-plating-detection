# -*- coding: utf-8 -*-
"""M3 库仑效率 + 不可逆容量增长变点法。

逐循环库仑效率 CE = 放电容量/充电容量，滚动均值后做变点检测；
析锂发生时 CE 下降、不可逆锂库存损失加速。变点用 ruptures（PELT，备选 CUSUM）。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..config import Config
from ..schema import CellData, Cycle
from .common import MethodResult


def _cycle_capacities(cyc: Cycle) -> Tuple[float, float]:
    ch, dis = 0.0, 0.0
    for seg in cyc.segments:
        if len(seg) < 2:
            continue
        energy = np.trapezoid(seg.current, seg.time) / 3600.0  # Ah（带符号）
        if seg.segment_type == "charge":
            ch += abs(energy)
        elif seg.segment_type == "discharge":
            dis += abs(energy)
    if "discharge_capacity_ah" in cyc.meta:
        dis = cyc.meta["discharge_capacity_ah"]
    return ch, dis


def _changepoints(signal: np.ndarray, method: str, pen: float, min_size: int) -> List[int]:
    import ruptures as rpt

    sig = signal.reshape(-1, 1)
    if method == "cusum":
        algo = rpt.Cusum(model="l2", min_size=min_size).fit(sig)
    else:
        algo = rpt.Pelt(model="rbf", min_size=min_size).fit(sig)
    bks = algo.predict(pen=pen)
    # bks 为变点结束位置（含末尾 len），返回变点起始索引
    return [b for b in bks if b < len(signal)]


def run_cell(cell: CellData, cfg: Config) -> List[MethodResult]:
    window = cfg.get("m3_coulomb", "rolling_window", default=5)
    drop_alarm = cfg.get("m3_coulomb", "ce_drop_alarm", default=0.02)
    method = cfg.get("m3_coulomb", "changepoint_method", default="pelt")
    pen = cfg.get("m3_coulomb", "pelt_pen", default=1.0)
    min_size = cfg.get("m3_coulomb", "min_segment_size", default=5)

    import pandas as pd

    rows: List[Dict] = []
    for cyc in cell.cycles:
        ch, dis = _cycle_capacities(cyc)
        # 仅保留充放电都完整的循环，排除一次性 RPT 的首尾残段
        if ch <= 1e-9 or dis <= 1e-9:
            continue
        ce = float(np.clip(dis / ch, 0.5, 1.05))
        rows.append({"cycle": cyc.cycle_index, "charge": ch,
                     "discharge": dis, "ce": ce})
    if len(rows) < min_size * 2:
        return []

    ce = np.array([r["ce"] for r in rows])
    cap = np.array([r["discharge"] for r in rows])
    # 边界稳健的居中滚动均值（min_periods=1，避免 same 卷积边界假信号）
    ce_smooth = (
        pd.Series(ce).rolling(window, center=True, min_periods=1).mean().to_numpy()
    )
    # 变点：在 CE 与容量斜率上分别检测
    ce_bk = set(_changepoints(ce_smooth, method, pen, min_size))
    cap_slope = np.gradient(cap)
    cap_bk = set(_changepoints(cap_slope, method, pen, min_size))
    all_bk = ce_bk | cap_bk

    # 基准 CE：取早期稳定窗口的中位数，抗离群
    baseline_ce = float(np.median(ce_smooth[: max(window, 3)]))
    results: List[MethodResult] = []
    for k, r in enumerate(rows):
        ce_drop = baseline_ce - ce_smooth[k]
        is_bk = k in all_bk
        # 阳性以持续 CE 下降为准；变点仅记录用于起始定位，不单独触发，
        # 避免周期性/可恢复的 CE 波动被误判为析锂
        positive = bool(ce_drop >= drop_alarm)
        results.append(
            MethodResult(
                method="M3",
                cell_id=cell.cell_id,
                cycle=r["cycle"],
                positive=positive,
                score=float(max(ce_drop, 0.0)),
                threshold=float(drop_alarm),
                detail={
                    "ce": round(float(ce_smooth[k]), 5),
                    "charge_ah": round(r["charge"], 5),
                    "discharge_ah": round(r["discharge"], 5),
                    "is_changepoint": bool(is_bk),
                },
                series={"index": np.arange(len(rows)), "ce": ce_smooth, "capacity": cap},
            )
        )
    return results
