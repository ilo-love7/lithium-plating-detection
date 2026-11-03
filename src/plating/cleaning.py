# -*- coding: utf-8 -*-
"""数据清洗：电压越界、时间戳断点、传感器漂移、短段过滤，并产出质量报告。"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .config import Config
from .schema import CellData, Cycle, Segment


def clean_segment(seg: Segment, cfg: Config) -> Tuple[Segment, Dict[str, Any]]:
    vmin = cfg.get("cleaning", "voltage_min", default=2.5)
    vmax = cfg.get("cleaning", "voltage_max", default=4.5)
    max_gap = cfg.get("cleaning", "max_time_gap_s", default=3600)
    drift_lim = cfg.get("cleaning", "drift_dv_per_s", default=0.01)
    n0 = len(seg)
    report: Dict[str, Any] = {
        "segment": seg.name or seg.segment_type,
        "type": seg.segment_type,
        "n_before": n0,
        "n_out_of_range": 0,
        "n_time_breaks": 0,
        "drift_flag": False,
        "dropped": False,
    }
    if n0 == 0:
        report["dropped"] = True
        return seg, report

    # 1) 电压越界
    in_range = (seg.voltage >= vmin) & (seg.voltage <= vmax)
    report["n_out_of_range"] = int(np.sum(~in_range))
    keep = in_range

    # 2) 时间戳断点（仅标注，不删除——保留连续性信息）
    if n0 >= 2:
        dt = np.diff(seg.time)
        report["n_time_breaks"] = int(np.sum(dt > max_gap))

    # 3) 静置段传感器漂移
    if seg.segment_type == "rest" and n0 >= 2:
        slope = abs(seg.voltage[-1] - seg.voltage[0]) / max(seg.duration_s, 1e-9)
        report["drift_flag"] = bool(slope > drift_lim)

    idx = np.where(keep)[0]
    cleaned = Segment(
        segment_type=seg.segment_type,
        time=seg.time[idx],
        current=seg.current[idx],
        voltage=seg.voltage[idx],
        temperature=seg.temperature[idx],
        name=seg.name,
        meta=dict(seg.meta),
    )
    min_pts = cfg.get("cleaning", "min_segment_points", default=10)
    if len(cleaned) < min_pts:
        report["dropped"] = True
    report["n_after"] = len(cleaned)
    return cleaned, report


def clean_cell(cell: CellData, cfg: Config) -> Tuple[CellData, List[Dict[str, Any]]]:
    out = CellData(
        cell_id=cell.cell_id,
        source=cell.source,
        nominal_capacity_ah=cell.nominal_capacity_ah,
        meta=dict(cell.meta),
    )
    quality: List[Dict[str, Any]] = []
    for cyc in cell.cycles:
        new_cyc = Cycle(
            cycle_index=cyc.cycle_index,
            ambient_temperature_c=cyc.ambient_temperature_c,
            rated_capacity_ah=cyc.rated_capacity_ah,
            meta=dict(cyc.meta),
        )
        for seg in cyc.segments:
            cleaned, rep = clean_segment(seg, cfg)
            rep["cycle"] = cyc.cycle_index
            rep["cell_id"] = cell.cell_id
            quality.append(rep)
            if not rep["dropped"]:
                new_cyc.add(cleaned)
        if new_cyc.segments:
            out.add(new_cyc)
    return out, quality


def quality_summary(quality: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(quality)
    dropped = sum(1 for r in quality if r["dropped"])
    oor = sum(r["n_out_of_range"] for r in quality)
    breaks = sum(r["n_time_breaks"] for r in quality)
    drift = sum(1 for r in quality if r["drift_flag"])
    return {
        "n_segments": total,
        "n_dropped": dropped,
        "n_points_out_of_range": oor,
        "n_time_breaks": breaks,
        "n_drift_flags": drift,
        "drop_rate": round(dropped / total, 4) if total else 0.0,
    }
