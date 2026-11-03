# -*- coding: utf-8 -*-
"""合成数据端到端测试：生成器、真值与四方法判定。"""
from plating.config import load_config
from plating.synthetic import generate_dataset, ground_truth
from plating.methods import run_all, METHOD_NAMES


def _make_dataset():
    cfg = load_config()
    cells = generate_dataset(cfg, n_cells=8, n_cycles=140)
    return cfg, cells


def test_dataset_shape_and_truth():
    _, cells = _make_dataset()
    gt = ground_truth(cells)
    assert len(cells) == 8
    n_plated = sum(1 for v in gt.values() if v["plating"])
    assert n_plated == 4          # 一半析锂一半对照


def test_methods_perfect_on_synthetic():
    cfg, cells = _make_dataset()
    gt = ground_truth(cells)
    for m in METHOD_NAMES:
        tp = fn = fp = tn = 0
        for cell in cells:
            truth = gt[cell.cell_id]["plating"]
            pred = any(r.positive for r in run_all(cell, cfg)[m])
            if truth and pred:
                tp += 1
            elif truth and not pred:
                fn += 1
            elif (not truth) and pred:
                fp += 1
            else:
                tn += 1
        assert tp + fn == 4 and tn + fp == 4
        assert fn == 0 and fp == 0, f"{m} 合成判定未达双1"
