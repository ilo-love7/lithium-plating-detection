# -*- coding: utf-8 -*-
"""弛豫段检测模块单元测试。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from plating.config import load_config
from plating.relaxation import (
    _runs, detect_in_segment, events_to_index_table,
)
from plating.schema import CellData, Segment


def test_runs_basic_and_merge():
    mask = np.array([True, True, False, True, True])
    # merge_gap=1 时间隔 1 点会被合并
    assert _runs(mask, 1) == [(0, 4)]
    # merge_gap=0 时拆成两段
    assert _runs(mask, 0) == [(0, 1), (3, 4)]


def test_runs_empty():
    assert _runs(np.zeros(5, dtype=bool), 1) == []


def _rest_segment(n=40, dt=100.0):
    return Segment(
        segment_type="rest",
        time=np.arange(n) * dt,
        current=np.zeros(n),
        voltage=np.full(n, 3.7),
        temperature=np.full(n, 25.0),
    )


def test_detect_relaxation_event_in_rest():
    cfg = load_config()
    cell = CellData(cell_id="c1", source="t", nominal_capacity_ah=3.0)
    events = detect_in_segment(_rest_segment(), cell, 0, cfg)
    assert len(events) == 1
    ev = events[0]
    assert ev.duration_s >= 1800
    assert ev.is_cv_suspect is False


def test_non_rest_segment_skipped():
    cfg = load_config()
    cell = CellData(cell_id="c1", source="t", nominal_capacity_ah=3.0)
    seg = _rest_segment()
    seg.segment_type = "charge"
    # only_trust_rest 默认开启，charge 段不判
    assert detect_in_segment(seg, cell, 0, cfg) == []


def test_events_to_index_table_empty():
    df = events_to_index_table([])
    assert df.empty
    assert "cell_id" in df.columns
