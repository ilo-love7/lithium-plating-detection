# -*- coding: utf-8 -*-
"""bootstrap AUC、融合投票与噪声鲁棒性测试。"""
import numpy as np

from plating.config import load_config
from plating.roc import bootstrap_auc
from plating.synthetic import generate_dataset
from plating.ensemble import collect_votes, fusion_metrics, build_fusion_table
from plating import robustness


def test_bootstrap_perfect_separation():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([False, False, True, True])
    mean, lo, hi, boot = bootstrap_auc(scores, labels, n_boot=100)
    assert mean == 1.0 and lo == 1.0 and hi == 1.0
    assert 0 < len(boot) <= 100
    assert (boot == 1.0).all()


def test_bootstrap_interval_ordering():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 30).astype(bool)
    scores = labels.astype(float) + rng.normal(0, 0.8, 30)
    mean, lo, hi, boot = bootstrap_auc(scores, labels, n_boot=200)
    assert lo <= mean <= hi
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0
    assert 0 < len(boot) <= 200


def test_fusion_metrics_pure():
    votes = np.array([[True, True, False], [True, False, False],
                      [False, False, False], [True, True, True]])
    labels = np.array([True, True, False, True])
    # >=2 票：前两个电芯判正，第三个阴性，第四个阳性
    r2 = fusion_metrics(votes, labels, 2)
    assert (r2["TP"], r2["FN"], r2["FP"], r2["TN"]) == (2, 1, 0, 1)
    # >=3 票：仅第四个判正
    r3 = fusion_metrics(votes, labels, 3)
    assert (r3["TP"], r3["FP"], r3["TN"]) == (1, 0, 1)


def test_collect_votes_shapes():
    cfg = load_config()
    cells = generate_dataset(cfg, n_cells=8, n_cycles=60)
    votes, labels, ids = collect_votes(cfg, cells)
    assert votes.shape == (8, 4)
    assert labels.sum() == 4
    assert len(ids) == 8


def test_build_fusion_table_rules():
    cfg = load_config()
    table = build_fusion_table(cfg, n_cells=8, n_cycles=80)
    assert len(table["rules"]) == 4
    for r in table["rules"]:
        assert 0.0 <= r["sensitivity"] <= 1.0
        assert 0.0 <= r["specificity"] <= 1.0


def test_config_with_noise_replaces_value():
    cfg = load_config()
    new_cfg = robustness.config_with_noise(cfg, 0.03)
    assert new_cfg.get("synthetic", "noise_std_v") == 0.03
    # 原始配置未被修改
    assert cfg.get("synthetic", "noise_std_v") != 0.03


def test_run_noise_robustness_small(monkeypatch):
    cfg = load_config()
    monkeypatch.setattr(robustness, "generate_dataset",
                        lambda c, n_cells, n_cycles:
                        generate_dataset(c, n_cells=8, n_cycles=40))
    df = robustness.run_noise_robustness(
        cfg, n_cells=8, n_cycles=40, n_boot=30, fusion_rule=3)
    # 每噪声水平：4 方法 + 1 融合行
    n_levels = len(cfg.get("synthetic", "robustness_noise_levels"))
    assert len(df) == n_levels * 5
    method_rows = df[df["method"].isin(["M1", "M2", "M3", "M4"])]
    assert method_rows["auc"].between(0, 1).all()
