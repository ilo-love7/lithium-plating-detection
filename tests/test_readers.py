# -*- coding: utf-8 -*-
"""readers 注册表与 generic 读取器单元测试。"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from plating.readers import auto_load
from plating.readers.base import available_readers, get_reader
from plating.readers.generic import GenericReader, _resolve_columns


def test_registry_lists_and_gets():
    assert "generic" in available_readers()
    reader = get_reader("generic", nominal_capacity_ah=2.5)
    assert isinstance(reader, GenericReader)
    assert reader.nominal_capacity_ah == 2.5


def test_get_unknown_reader_raises():
    with pytest.raises(KeyError):
        get_reader("does-not-exist")


def test_resolve_columns_aliases():
    df = pd.DataFrame(columns=["t", "I", "V", "temp", "cycle_index"])
    rc = _resolve_columns(df)
    assert rc["time"] == "t"
    assert rc["current"] == "I"
    assert rc["voltage"] == "V"
    assert rc["temperature"] == "temp"
    assert rc["cycle"] == "cycle_index"


def _schema_df():
    return pd.DataFrame({
        "time": np.arange(20) * 10.0,
        "current": np.full(20, 1.0),
        "voltage": np.linspace(3.5, 4.0, 20),
        "temperature": np.full(20, 25.0),
        "cycle": [0] * 10 + [1] * 10,
        "segment": ["charge"] * 20,
    })


def test_generic_build_structure():
    reader = GenericReader(nominal_capacity_ah=2.0)
    cells = reader._build(_schema_df(), "test_cell")
    assert len(cells) == 1
    cell = cells[0]
    assert cell.cell_id == "test_cell"
    assert len(cell.cycles) == 2
    assert cell.cycles[0].by_type("charge")[0].voltage.shape == (10,)


def test_generic_build_missing_column_raises():
    reader = GenericReader()
    bad = pd.DataFrame({"time": [1], "current": [1]})  # 缺 voltage
    with pytest.raises(ValueError):
        reader._build(bad, "x")


def test_generic_read_csv_roundtrip(tmp_path):
    path = tmp_path / "cell.csv"
    _schema_df().to_csv(path, index=False)
    reader = GenericReader(nominal_capacity_ah=2.0)
    cells = reader.read_path(str(path))
    assert cells[0].cell_id == "cell"
    assert len(cells[0].cycles) == 2


def test_generic_read_directory_raises(tmp_path):
    reader = GenericReader()
    with pytest.raises(IsADirectoryError):
        reader.read_path(str(tmp_path))


def test_generic_bad_extension(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("x")
    reader = GenericReader()
    with pytest.raises(ValueError):
        reader.read_path(str(p))


def test_auto_load_generic_fallback(tmp_path):
    path = tmp_path / "plain.csv"
    _schema_df().to_csv(path, index=False)
    cells = auto_load(str(path), nominal_capacity_ah=2.0)
    assert cells[0].source == "generic"


def test_auto_load_routes_by_keyword():
    # 路径含关键词应路由到对应 reader；文件不存在时抛 FileNotFoundError
    # （而非 KeyError），证明来源选择正确
    for kw in ["oxford", "nasa", "resval"]:
        with pytest.raises(FileNotFoundError):
            auto_load(f"D:/fake/{kw}/missing.mat")
