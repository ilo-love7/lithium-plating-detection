# -*- coding: utf-8 -*-
"""Arbin 原始文件读取器（MATR-10 ResVal）。

制表符分隔文本，首行为日期信息，第二行为列名。
按连续 State 列分段：C->charge, D->discharge, R->rest, O->unknown。
本数据集无温度通道（恒温箱），温度填占位常量并在 meta 标注。
"""
from __future__ import annotations

import glob
import os
from typing import List

import numpy as np
import pandas as pd

from ..schema import CellData, Cycle, Segment
from .base import BaseReader, register_reader

_STATE_MAP = {"C": "charge", "D": "discharge", "R": "rest", "O": "unknown"}


@register_reader("arbin")
class ArbinReader(BaseReader):
    def read_path(self, path: str) -> List[CellData]:
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "ResVal_*.*")))
            if not files:
                raise FileNotFoundError(f"目录下无 ResVal 文件：{path}")
        else:
            files = [path]
        cap = self.nominal_capacity_ah or 4.84
        cells: List[CellData] = []
        for fp in files:
            try:
                cells.append(self._read_one(fp, cap))
            except Exception as e:  # 单文件失败不阻断整批
                print(f"[arbin] 跳过 {os.path.basename(fp)}: {e!r}")
        return cells

    def _read_one(self, fp: str, cap: float) -> CellData:
        df = pd.read_csv(fp, sep="\t", skiprows=1, engine="python", on_bad_lines="skip")
        df.columns = [str(c).strip() for c in df.columns]
        t = pd.to_numeric(df["Test (Sec)"], errors="coerce").to_numpy()
        i = pd.to_numeric(df["Amps"], errors="coerce").to_numpy()
        v = pd.to_numeric(df["Volts"], errors="coerce").to_numpy()
        state = df["State"].astype(str).to_numpy()
        cyc_no = pd.to_numeric(df["Cyc#"], errors="coerce").fillna(0).astype(int).to_numpy()
        # 无温度通道，占位 25℃
        temp = np.full(len(df), 25.0)

        valid = np.isfinite(t) & np.isfinite(i) & np.isfinite(v)
        t, i, v, state, cyc_no, temp = (
            t[valid], i[valid], v[valid], state[valid], cyc_no[valid], temp[valid]
        )

        cell_id = "MATR10_" + os.path.basename(fp).split("_")[1]
        cell = CellData(
            cell_id=cell_id,
            source="arbin",
            nominal_capacity_ah=cap,
            meta={"file": os.path.basename(fp), "temperature_recorded": False},
        )
        # 按连续 State 切分
        change = np.concatenate([[0], np.where(state[1:] != state[:-1])[0] + 1, [len(df)]])
        cycles: dict = {}
        for k in range(len(change) - 1):
            a, b = change[k], change[k + 1]
            stype = _STATE_MAP.get(state[a], "unknown")
            seg = Segment(
                segment_type=stype,
                time=t[a:b] - t[a],
                current=i[a:b],
                voltage=v[a:b],
                temperature=temp[a:b],
                name=f"{stype}_{a}",
                meta={"global_start_s": float(t[a])},
            )
            cval = int(cyc_no[a])
            if cval not in cycles:
                cycles[cval] = Cycle(cycle_index=cval, meta={"global_time": True})
            cycles[cval].add(seg)
        for cval in sorted(cycles):
            cell.add(cycles[cval])
        return cell
