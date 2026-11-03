# -*- coding: utf-8 -*-
"""Oxford OBD1 读取器。

文件结构：Cell1..8 -> cyc0000/0100/... -> C1ch/C1dc/OCVch/OCVdc -> t/v/q/T
原始无电流字段，电流由 q(mAh) 差分还原：i[A] = 3.6 * dq[mAh]/dt[s]。
"""
from __future__ import annotations

import os
from typing import List

import numpy as np
import scipy.io as sio

from ..schema import CellData, Cycle, Segment
from .base import BaseReader, register_reader

_SEG_MAP = {
    "C1ch": "charge",
    "OCVch": "charge",
    "C1dc": "discharge",
    "OCVdc": "discharge",
}


def _q_to_current(q_mah: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    """由电荷(mAh)与时间(s)还原电流(A)。"""
    if len(q_mah) < 2:
        return np.zeros_like(q_mah)
    dq = np.gradient(q_mah, t_s)
    return 3.6 * dq  # mAh/s -> A


@register_reader("oxford")
class OxfordReader(BaseReader):
    def read_path(self, path: str) -> List[CellData]:
        if os.path.isdir(path):
            mats = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.lower().endswith(".mat") and "ExampleDC" not in f
            ]
            if not mats:
                raise FileNotFoundError(f"目录下无 .mat：{path}")
            path = mats[0]
        cap = self.nominal_capacity_ah or 0.740
        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        cell_keys = sorted(k for k in mat.keys() if k.startswith("Cell"))
        cells: List[CellData] = []
        for ck in cell_keys:
            cell_node = mat[ck]
            cell_id = f"Oxford_{ck}"
            cell = CellData(
                cell_id=cell_id,
                source="oxford",
                nominal_capacity_ah=cap,
                meta={"chamber_temp_c": 40.0, "file": os.path.basename(path)},
            )
            cyc_names = sorted(
                n for n in cell_node._fieldnames if n.startswith("cyc")
            )
            for cn in cyc_names:
                cyc_node = getattr(cell_node, cn)
                try:
                    cycle_index = int(cn.replace("cyc", ""))
                except ValueError:
                    cycle_index = 0
                cycle = Cycle(
                    cycle_index=cycle_index,
                    ambient_temperature_c=40.0,
                    rated_capacity_ah=cap,
                    meta={"characterisation": True},
                )
                for seg_name in cyc_node._fieldnames:
                    seg_node = getattr(cyc_node, seg_name)
                    # t 为 MATLAB datenum（天），转为相对秒
                    t_raw = np.asarray(seg_node.t, dtype=float)
                    t = (t_raw - t_raw[0]) * 86400.0
                    v = np.asarray(seg_node.v, dtype=float)
                    q = np.asarray(seg_node.q, dtype=float)
                    T = np.asarray(seg_node.T, dtype=float)
                    i = _q_to_current(q, t)
                    seg_type = _SEG_MAP.get(seg_name, "unknown")
                    rate_label = "1C" if seg_name.startswith("C1") else "pseudoOCV"
                    cycle.add(
                        Segment(
                            segment_type=seg_type,
                            time=t,
                            current=i,
                            voltage=v,
                            temperature=T,
                            name=seg_name,
                            meta={"rate": rate_label, "q_mah": q},
                        )
                    )
                cell.add(cycle)
            cells.append(cell)
        return cells
