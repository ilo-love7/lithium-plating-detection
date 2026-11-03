# -*- coding: utf-8 -*-
"""通用工具函数单元测试（不依赖外部数据）。"""
import numpy as np

from plating.methods.common import (
    MethodResult, fill_nonfinite, sg_smooth, safe_div,
)


def test_fill_nonfinite_interpolates_middle():
    x = np.array([1.0, np.nan, 3.0, np.inf, 5.0])
    out = fill_nonfinite(x)
    assert np.isfinite(out).all()
    assert np.isclose(out[1], 2.0)       # 中点线性插值
    assert np.isclose(out[3], 4.0)


def test_fill_nonfinite_all_bad_returns_zeros():
    out = fill_nonfinite(np.array([np.nan, np.inf]))
    assert np.allclose(out, 0.0)


def test_sg_smooth_keeps_length_and_finite():
    x = np.linspace(0, 1, 50) + np.random.default_rng(0).normal(0, 0.01, 50)
    out = sg_smooth(x, window=11, polyorder=2)
    assert len(out) == len(x)
    assert np.isfinite(out).all()


def test_sg_smooth_short_array():
    x = np.array([1.0, 2.0])
    out = sg_smooth(x, window=11, polyorder=2)
    assert len(out) == 2


def test_safe_div_no_inf():
    a = np.array([1.0, 2.0])
    b = np.array([0.0, 4.0])
    out = safe_div(a, b)
    assert np.isfinite(out).all()
    assert np.isclose(out[1], 0.5)


def test_method_result_as_row():
    r = MethodResult(method="M1", cell_id="c", cycle=3, positive=True,
                     score=0.5, threshold=0.4)
    row = r.as_row()
    assert row["method"] == "M1" and row["positive"] is True
    assert set(["method", "cell_id", "cycle", "score", "threshold"]).issubset(row)
