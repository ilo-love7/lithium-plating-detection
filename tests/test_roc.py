# -*- coding: utf-8 -*-
"""ROC/AUC 纯函数单元测试。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from plating.config import load_config
from plating.roc import (
    auc, build_roc_table, collect_cell_scores, roc_curve,
)
from plating.synthetic import generate_dataset


def test_roc_perfect_separation_auc_one():
    # 负样本分数低、正样本分数高，完全可分
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([False, False, True, True])
    fpr, tpr = roc_curve(scores, labels)
    assert auc(fpr, tpr) == 1.0
    # 端点：全负 (0,0) 与全正 (1,1)
    assert fpr[0] == 0.0 and tpr[0] == 0.0
    assert fpr[-1] == 1.0 and tpr[-1] == 1.0
    # 单调不减
    assert np.all(np.diff(fpr) >= 0)
    assert np.all(np.diff(tpr) >= 0)


def test_roc_all_negative_scores_auc_half():
    # 正负样本分数完全相同，无法区分，AUC=0.5
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([False, False, True, True])
    fpr, tpr = roc_curve(scores, labels)
    assert auc(fpr, tpr) == 0.5


def test_roc_inverted_labels_auc_zero():
    # 正样本分数反而更低，完全反向，AUC=0
    scores = np.array([0.8, 0.9, 0.1, 0.2])
    labels = np.array([False, False, True, True])
    fpr, tpr = roc_curve(scores, labels)
    assert auc(fpr, tpr) == 0.0


def test_roc_ties_partial():
    # 含并列分数，AUC 介于 0 与 1
    scores = np.array([0.3, 0.5, 0.5, 0.7])
    labels = np.array([False, True, False, True])
    fpr, tpr = roc_curve(scores, labels)
    a = auc(fpr, tpr)
    assert 0.0 < a < 1.0


def test_collect_cell_scores_shapes():
    cfg = load_config()
    cells = generate_dataset(cfg, n_cells=8, n_cycles=60)
    scores, labels = collect_cell_scores(cfg, cells, "M1")
    assert len(scores) == 8 and len(labels) == 8
    assert labels.sum() == 4  # 一半析锂
    assert (scores >= 0).all()


def test_build_roc_table_all_methods_auc_one():
    cfg = load_config()
    table = build_roc_table(cfg, n_cells=8, n_cycles=100)
    assert set(table.keys()) == {"M1", "M2", "M3", "M4"}
    for m, d in table.items():
        assert d["auc"] == 1.0
        assert len(d["fpr"]) == len(d["tpr"])
