# -*- coding: utf-8 -*-
"""数据清洗模块单元测试。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from plating.cleaning import clean_segment, quality_summary
from plating.config import load_config
from plating.schema import Segment


def _segment(voltages, seg_type="rest", n=40):
    t = np.arange(n) * 10.0
    return Segment(
        segment_type=seg_type,
        time=t,
        current=np.zeros(n),
        voltage=np.asarray(voltages, dtype=float),
        temperature=np.full(n, 25.0),
    )


def test_clean_segment_removes_out_of_range():
    cfg = load_config()
    voltages = np.full(40, 3.7)
    voltages[:5] = 5.0  # 越上界
    seg = _segment(voltages)
    cleaned, report = clean_segment(seg, cfg)
    assert report["n_out_of_range"] == 5
    assert len(cleaned) == 35
    assert np.all((cleaned.voltage >= 2.5) & (cleaned.voltage <= 4.5))


def test_clean_empty_segment_dropped():
    cfg = load_config()
    seg = _segment([], n=0)
    _, report = clean_segment(seg, cfg)
    assert report["dropped"] is True


def test_clean_short_segment_dropped():
    cfg = load_config()
    # 仅 5 点，低于 min_segment_points
    seg = _segment(np.full(5, 3.7), n=5)
    _, report = clean_segment(seg, cfg)
    assert report["dropped"] is True


def test_quality_summary_aggregates():
    summary = quality_summary([
        {"dropped": True, "n_out_of_range": 3, "n_time_breaks": 0,
         "drift_flag": False},
        {"dropped": False, "n_out_of_range": 1, "n_time_breaks": 2,
         "drift_flag": True},
    ])
    assert summary["n_segments"] == 2
    assert summary["n_dropped"] == 1
    assert summary["n_points_out_of_range"] == 4
    assert summary["n_time_breaks"] == 2
    assert summary["n_drift_flags"] == 1
    assert summary["drop_rate"] == 0.5
