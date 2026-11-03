# -*- coding: utf-8 -*-
"""静置/弛豫段自动检测。

规则：电流绝对值 < C/50（可配置）且持续 >= 阈值时长，判为有效弛豫段。
含 CV 尾电流混淆排查、小间隙合并、边界安全边距，并产出索引表。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from .config import Config
from .schema import CellData, Segment


@dataclass
class RelaxationEvent:
    cell_id: str
    cycle: int
    source_segment: str
    start_time: float
    end_time: float
    duration_s: float
    mean_current_a: float
    start_voltage: float
    end_voltage: float
    is_cv_suspect: bool
    # 指向原始 segment 内的数组索引（闭区间）
    start_idx: int
    end_idx: int
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> Dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "cycle": self.cycle,
            "source_segment": self.source_segment,
            "start_time_s": round(self.start_time, 2),
            "end_time_s": round(self.end_time, 2),
            "duration_s": round(self.duration_s, 2),
            "mean_current_a": round(self.mean_current_a, 5),
            "start_voltage_v": round(self.start_voltage, 4),
            "end_voltage_v": round(self.end_voltage, 4),
            "is_cv_suspect": self.is_cv_suspect,
        }


def _runs(mask: np.ndarray, merge_gap: int):
    """把布尔数组转为 (s,e) 区间，间隔 <= merge_gap 的相邻区间合并。"""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    splits = np.where(np.diff(idx) > (merge_gap + 1))[0]
    starts = np.concatenate([[idx[0]], idx[splits + 1]])
    ends = np.concatenate([idx[splits], [idx[-1]]])
    return list(zip(starts.tolist(), ends.tolist()))


def detect_in_segment(
    seg: Segment,
    cell: CellData,
    cycle_index: int,
    cfg: Config,
) -> List[RelaxationEvent]:
    thr_c = cfg.get("relaxation", "current_threshold_c", default=0.02)
    min_dur = cfg.get("relaxation", "min_duration_s", default=1800)
    merge_gap_s = cfg.get("relaxation", "merge_gap_s", default=60)
    margin_s = cfg.get("relaxation", "boundary_margin_s", default=30)
    cv_warn_c = cfg.get("relaxation", "cv_warning_current_c", default=0.05)
    cv_voltage_std = cfg.get("relaxation", "cv_voltage_std", default=0.005)
    reject_cv = cfg.get("relaxation", "reject_cv", default=True)
    only_rest = cfg.get("relaxation", "only_trust_rest_segment", default=True)

    threshold_a = thr_c * cell.nominal_capacity_ah
    cv_warn_a = cv_warn_c * cell.nominal_capacity_ah
    if len(seg) < 2:
        return []
    # 仪器已准确分段时，非 rest 段直接跳过，避免慢放电/CV 误判
    if only_rest and seg.segment_type != "rest":
        return []

    low = np.abs(seg.current) < threshold_a
    if not np.any(low):
        return []

    # 用采样间隔把 merge_gap 秒换算成点数
    med_dt = float(np.median(np.diff(seg.time))) if len(seg) >= 2 else 1.0
    merge_gap_pts = max(1, int(round(merge_gap_s / max(med_dt, 1e-6))))
    intervals = _runs(low, merge_gap_pts)

    events: List[RelaxationEvent] = []
    for s, e in intervals:
        t0, t1 = seg.time[s], seg.time[e]
        dur = t1 - t0
        if dur < min_dur:
            continue
        # 边界安全边距
        if margin_s > 0:
            s = min(e, s + int(round(margin_s / max(med_dt, 1e-6))))
            e = max(s, e - int(round(margin_s / max(med_dt, 1e-6))))
            t0, t1 = seg.time[s], seg.time[e]
        mean_i = float(np.mean(np.abs(seg.current[s : e + 1])))
        # CV 嫌疑：位于 charge 段内，且该区间之前存在明显充电电流（>cv_warn）
        # 或区间电压几乎恒定（电压钳位特征）
        is_cv = False
        sub_v = seg.voltage[s : e + 1]
        v_clamped = bool(np.std(sub_v) < cv_voltage_std)
        if seg.segment_type == "charge":
            pre = seg.current[: max(1, s)]
            is_cv = bool((s > 0 and np.max(np.abs(pre)) > cv_warn_a) or v_clamped)
        if is_cv and reject_cv:
            continue
        events.append(
            RelaxationEvent(
                cell_id=cell.cell_id,
                cycle=cycle_index,
                source_segment=seg.name or seg.segment_type,
                start_time=float(t0),
                end_time=float(t1),
                duration_s=float(t1 - t0),
                mean_current_a=mean_i,
                start_voltage=float(seg.voltage[s]),
                end_voltage=float(seg.voltage[e]),
                is_cv_suspect=is_cv,
                start_idx=int(s),
                end_idx=int(e),
            )
        )
    return events


def detect_cell(cell: CellData, cfg: Config) -> List[RelaxationEvent]:
    events: List[RelaxationEvent] = []
    for cyc in cell.cycles:
        for seg in cyc.segments:
            events.extend(detect_in_segment(seg, cell, cyc.cycle_index, cfg))
    return events


def extract_segment(seg: Segment, ev: RelaxationEvent) -> Segment:
    """根据事件从原始 segment 切出弛豫子段（时间归零）。"""
    s, e = ev.start_idx, ev.end_idx
    sub = Segment(
        segment_type="rest",
        time=seg.time[s : e + 1] - seg.time[s],
        current=seg.current[s : e + 1],
        voltage=seg.voltage[s : e + 1],
        temperature=seg.temperature[s : e + 1],
        name=f"relax_cyc{ev.cycle}",
    )
    return sub


def events_to_index_table(events: List[RelaxationEvent]):
    import pandas as pd

    if not events:
        return pd.DataFrame(
            columns=[
                "cell_id", "cycle", "source_segment", "start_time_s", "end_time_s",
                "duration_s", "mean_current_a", "start_voltage_v",
                "end_voltage_v", "is_cv_suspect",
            ]
        )
    return pd.DataFrame([e.as_row() for e in events])
