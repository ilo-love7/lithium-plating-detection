# -*- coding: utf-8 -*-
"""NASA PCoE 读取器。

B*.mat -> Bxxxx.cycle[]：每条含 type(charge/discharge/impedance) 与 data。
charge/discharge 交替出现，配对为一个逻辑循环；impedance(EIS) 跳过。
放电段 data.Capacity 作为该循环实测放电容量。
"""
from __future__ import annotations

import glob
import os
from typing import List

import numpy as np
import scipy.io as sio

from ..schema import CellData, Cycle, Segment
from .base import BaseReader, register_reader


@register_reader("nasa")
class NASAReader(BaseReader):
    def read_path(self, path: str) -> List[CellData]:
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "**", "B*.mat"), recursive=True))
            if not files:
                raise FileNotFoundError(f"目录下无 B*.mat：{path}")
        else:
            files = [path]
        cap_default = self.nominal_capacity_ah or 2.0
        cells: List[CellData] = []
        for fp in files:
            cells.extend(self._read_one(fp, cap_default))
        return cells

    def _read_one(self, fp: str, cap_default: float) -> List[CellData]:
        mat = sio.loadmat(fp, squeeze_me=True, struct_as_record=False)
        top = [k for k in mat.keys() if not k.startswith("__")]
        out: List[CellData] = []
        for tk in top:
            node = mat[tk]
            if not hasattr(node, "cycle"):
                continue
            raw_cycles = node.cycle
            cell_id = f"NASA_{tk}"
            cell = CellData(
                cell_id=cell_id,
                source="nasa",
                nominal_capacity_ah=cap_default,
                meta={"file": os.path.basename(fp)},
            )
            logical = 0
            pending_charge = None
            for rc in raw_cycles:
                ctype = str(rc.type)
                if ctype == "impedance":
                    continue
                d = rc.data
                t = np.asarray(d.Time, dtype=float)
                v = np.asarray(d.Voltage_measured, dtype=float)
                i = np.asarray(d.Current_measured, dtype=float)
                T = np.asarray(d.Temperature_measured, dtype=float)
                ambient = float(rc.ambient_temperature)
                seg = Segment(
                    segment_type="charge" if ctype == "charge" else "discharge",
                    time=t,
                    current=i,
                    voltage=v,
                    temperature=T,
                    name=ctype,
                    meta={"ambient_c": ambient},
                )
                if ctype == "charge":
                    # 先落一个待配对
                    pending_charge = seg
                elif ctype == "discharge":
                    cyc = Cycle(
                        cycle_index=logical,
                        ambient_temperature_c=ambient,
                        rated_capacity_ah=cap_default,
                    )
                    if pending_charge is not None:
                        cyc.add(pending_charge)
                        pending_charge = None
                    cyc.add(seg)
                    if hasattr(d, "Capacity"):
                        cyc.meta["discharge_capacity_ah"] = float(d.Capacity)
                    cell.add(cyc)
                    logical += 1
            out.append(cell)
        return out
