# -*- coding: utf-8 -*-
"""readers 注册表与 generic 读取器单元测试。"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio

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


# ---------------- 真实数据 reader 合成文件测试 ----------------

def test_arbin_reader_synthetic(tmp_path):
    rows = []

    def add(state, cyc, n, t0):
        amp = 1.0 if state == "C" else (-1.0 if state == "D" else 0.0)
        for k in range(n):
            rows.append({"Test (Sec)": t0 + k * 10, "Amps": amp,
                         "Volts": 3.7 + k * 0.02, "State": state, "Cyc#": cyc})

    add("C", 0, 5, 0)
    add("R", 0, 2, 50)
    add("D", 0, 5, 70)
    add("C", 1, 5, 120)
    add("D", 1, 5, 170)

    fp = tmp_path / "ResVal_cell01_test.txt"
    with open(fp, "w", encoding="utf-8") as f:
        f.write("Date info line\n")
    pd.DataFrame(rows).to_csv(fp, sep="\t", index=False, mode="a")

    from plating.readers.arbin import ArbinReader
    cells = ArbinReader(nominal_capacity_ah=4.84).read_path(str(fp))
    assert len(cells) == 1
    cell = cells[0]
    assert cell.cell_id == "MATR10_cell01"
    assert cell.source == "arbin"
    assert len(cell.cycles) == 2
    types = [s.segment_type for c in cell.cycles for s in c.segments]
    assert "charge" in types and "discharge" in types and "rest" in types


def _nasa_seg_data(n, with_capacity=False):
    d = {"Time": np.arange(n) * 10.0,
         "Voltage_measured": np.linspace(3.5, 4.0, n),
         "Current_measured": np.ones(n),
         "Temperature_measured": np.full(n, 24.0)}
    if with_capacity:
        d["Capacity"] = 1.85
    return d


def test_nasa_reader_synthetic(tmp_path):
    charge = {"type": "charge", "ambient_temperature": 24,
              "data": _nasa_seg_data(5)}
    disc = {"type": "discharge", "ambient_temperature": 24,
            "data": _nasa_seg_data(5, with_capacity=True)}
    fp = tmp_path / "B0005.mat"
    sio.savemat(str(fp), {"B0005": {"cycle": [charge, disc]}})

    from plating.readers.nasa import NASAReader
    cells = NASAReader(nominal_capacity_ah=2.0).read_path(str(fp))
    assert len(cells) == 1
    cell = cells[0]
    assert cell.cell_id == "NASA_B0005"
    assert cell.source == "nasa"
    # charge+discharge 配对为一个逻辑循环
    assert len(cell.cycles) == 1
    cyc = cell.cycles[0]
    assert len(cyc.by_type("charge")) == 1
    assert len(cyc.by_type("discharge")) == 1
    assert cyc.meta["discharge_capacity_ah"] == pytest.approx(1.85)


def _oxford_seg(n, t_base):
    return {"t": t_base + np.arange(n) * 5 / 86400.0,
            "v": np.linspace(3.7, 4.0, n),
            "q": np.linspace(0, 100, n),
            "T": np.full(n, 40.0)}


def test_oxford_reader_synthetic(tmp_path):
    fp = tmp_path / "Oxford_Cell1.mat"
    sio.savemat(str(fp), {"Cell1": {"cyc0000": {
        "C1ch": _oxford_seg(5, 738000), "C1dc": _oxford_seg(5, 738001)}}})

    from plating.readers.oxford import OxfordReader
    cells = OxfordReader(nominal_capacity_ah=0.74).read_path(str(fp))
    assert len(cells) == 1
    cell = cells[0]
    assert cell.cell_id == "Oxford_Cell1"
    assert cell.source == "oxford"
    assert len(cell.cycles) == 1
    seg_types = sorted(s.segment_type for s in cell.cycles[0].segments)
    assert seg_types == ["charge", "discharge"]
    # 电流由 q 差分还原，应为有限值
    for s in cell.cycles[0].segments:
        assert np.isfinite(s.current).all()
