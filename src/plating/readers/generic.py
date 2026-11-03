# -*- coding: utf-8 -*-
"""通用 CSV / HDF5 读取器：读取已统一 schema 的中间层文件。

CSV 列名自动映射（支持常见别名）；HDF5 按扁平 DataFrame 数据集读取。
"""
from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd

from ..schema import CellData, Cycle, Segment
from .base import BaseReader, register_reader

_COL_ALIASES = {
    "time": ["time", "t", "timestamp", "时间"],
    "current": ["current", "i", "电流", "current_measured"],
    "voltage": ["voltage", "v", "电压", "voltage_measured"],
    "temperature": ["temperature", "temp", "t_c", "温度", "temperature_measured"],
    "cycle": ["cycle", "cycle_index", "循环", "cycle_number"],
    "segment": ["segment", "segment_type", "段类型"],
}


def _resolve_columns(df: pd.DataFrame) -> dict:
    lower = {str(c).lower(): c for c in df.columns}
    resolved = {}
    for canon, aliases in _COL_ALIASES.items():
        for a in aliases:
            if a.lower() in lower:
                resolved[canon] = lower[a.lower()]
                break
    return resolved


@register_reader("generic")
class GenericReader(BaseReader):
    def read_path(self, path: str) -> List[CellData]:
        if os.path.isdir(path):
            raise IsADirectoryError("generic 读取器仅接受单个文件")
        ext = os.path.splitext(path)[1].lower()
        if ext in (".csv",):
            df = pd.read_csv(path)
        elif ext in (".h5", ".hdf5"):
            df = pd.read_hdf(path)
        else:
            raise ValueError(f"不支持的扩展名 {ext}")
        return self._build(df, os.path.splitext(os.path.basename(path))[0])

    def _build(self, df: pd.DataFrame, cell_id: str) -> List[CellData]:
        rc = _resolve_columns(df)
        for need in ("time", "current", "voltage"):
            if need not in rc:
                raise ValueError(f"缺少必要列 {need}，实际列：{list(df.columns)}")
        n = len(df)
        cycle_idx = df[rc["cycle"]].to_numpy() if "cycle" in rc else np.zeros(n, int)
        seg_col = rc.get("segment")
        cell = CellData(
            cell_id=cell_id,
            source="generic",
            nominal_capacity_ah=self.nominal_capacity_ah or 2.0,
        )
        for cval in sorted(set(np.asarray(cycle_idx))):
            mask = np.asarray(cycle_idx) == cval
            sub = df[mask]
            cyc = Cycle(cycle_index=int(cval))
            if seg_col is not None:
                groups = sub.groupby(seg_col)
            else:
                groups = [("unknown", sub)]
            for stype, g in groups:
                cyc.add(
                    Segment(
                        segment_type=str(stype),
                        time=g[rc["time"]].to_numpy(),
                        current=g[rc["current"]].to_numpy(),
                        voltage=g[rc["voltage"]].to_numpy(),
                        temperature=(
                            g[rc["temperature"]].to_numpy()
                            if "temperature" in rc
                            else np.full(len(g), 25.0)
                        ),
                    )
                )
            cell.add(cyc)
        return [cell]
