# -*- coding: utf-8 -*-
"""退化对齐模块单元测试。"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from plating.config import load_config
from plating.degradation import (
    alignment_table, capacity_hold_curve, summarize_alignment,
)
from plating.schema import CellData, Cycle
from plating.synthetic import generate_dataset


def _make_cell(caps):
    cell = CellData(cell_id="c1", source="test", nominal_capacity_ah=3.0)
    for i, cap in enumerate(caps):
        cell.add(Cycle(cycle_index=i, rated_capacity_ah=cap))
    return cell


def test_capacity_hold_curve_ratios():
    cell = _make_cell([3.0, 2.85, 2.7])
    soh = capacity_hold_curve(cell)
    assert soh[0] == 1.0
    assert soh[1] == pytest.approx(0.95)
    assert soh[2] == pytest.approx(0.9)


def test_capacity_hold_curve_empty():
    cell = CellData(cell_id="c", source="t", nominal_capacity_ah=1.0)
    assert capacity_hold_curve(cell) == {}


def test_summarize_alignment_stats():
    table = pd.DataFrame({
        "method": ["M1", "M1", "M3", "M3"],
        "alarm_cycle": [50, 60, 30, 40],
        "detection_lag_cycles": [10, 20, 2, 8],
        "soh_at_alarm": [0.92, 0.90, 0.95, 0.93],
    })
    summary = summarize_alignment(table)
    m1 = summary[summary["method"] == "M1"].iloc[0]
    assert m1["n_alarms"] == 2
    assert m1["lag_cycles_median"] == 15.0
    assert m1["soh_at_alarm_median"] == 0.91
    m3 = summary[summary["method"] == "M3"].iloc[0]
    assert m3["lag_cycles_median"] == 5.0


def test_alignment_table_plated_only():
    cfg = load_config()
    cells = generate_dataset(cfg, n_cells=8, n_cycles=60)
    table = alignment_table(cfg, cells)
    # 4 个析锂电芯 × 4 方法 = 16 行
    assert len(table) == 16
    assert set(table["method"]) == {"M1", "M2", "M3", "M4"}
    # 报警循环不早于 onset
    valid = table.dropna(subset=["alarm_cycle"])
    assert (valid["alarm_cycle"] >= valid["plating_onset_cycle"]).all()
