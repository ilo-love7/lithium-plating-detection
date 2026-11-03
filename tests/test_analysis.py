# -*- coding: utf-8 -*-
"""analysis 编排与表格函数单元测试（合成数据）。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from plating.analysis import (
    cell_level_table, get_data_root, kappa_matrix, long_result_table,
    operating_condition_table,
)
from plating.config import load_config
from plating.methods import METHOD_NAMES, run_all
from plating.synthetic import generate_dataset


def _run_small():
    cfg = load_config()
    cells = generate_dataset(cfg, n_cells=8, n_cycles=60)
    results = {c.cell_id: run_all(c, cfg) for c in cells}
    return cells, results


def test_get_data_root_env_override(monkeypatch):
    monkeypatch.setenv("PLATING_DATA_ROOT", "D:/some/custom/root")
    assert get_data_root().endswith(os.path.join("custom", "root"))


def test_long_result_table_rows():
    cells, results = _run_small()
    tbl = long_result_table(cells, results)
    assert len(tbl) > 0
    assert "method" in tbl.columns


def test_cell_level_table_flags():
    cells, results = _run_small()
    tbl = cell_level_table(cells, results)
    assert len(tbl) == 8
    for m in METHOD_NAMES:
        assert f"{m}_positive" in tbl.columns
        assert f"{m}_first" in tbl.columns
        assert f"{m}_n" in tbl.columns


def test_kappa_matrix_shape_and_diagonal():
    cells, results = _run_small()
    tbl = cell_level_table(cells, results)
    kappa = kappa_matrix(tbl)
    assert list(kappa.index) == list(METHOD_NAMES)
    # 对角线自相关 = 1
    for m in METHOD_NAMES:
        assert kappa.loc[m, m] == 1.0


def test_operating_condition_table():
    cells, results = _run_small()
    tbl = operating_condition_table(cells)
    assert len(tbl) == 8
    assert "median_charge_c" in tbl.columns
    assert "has_real_rest" in tbl.columns
