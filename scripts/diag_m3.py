# -*- coding: utf-8 -*-
"""诊断 M3 容量与 CE 计算。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from plating.config import load_config
from plating.readers import load
from plating.cleaning import clean_cell
from plating.methods.m3_coulomb import _cycle_capacities

f = r"D:\项目\02_数据集\01_国际公开集\MATR-10_DiagnosticFree\ResValData\ResVal_000013_000185.001"
cfg = load_config()
cell = clean_cell(load("arbin", f)[0], cfg)[0]
for cyc in cell.cycles:
    ch, dis = _cycle_capacities(cyc)
    ce = dis / ch if ch > 1e-9 else float("nan")
    print(f"cyc{cyc.cycle_index:3d} nseg={len(cyc.segments):2d} "
          f"charge={ch:6.4f} discharge={dis:6.4f} CE={ce:6.4f}")
    # 段明细
    for s in cyc.segments:
        if len(s) >= 2:
            e = np.trapezoid(s.current, s.time) / 3600
            print(f"      {s.segment_type:9s} n={len(s):5d} signed_ah={e:7.5f} "
                  f"Imean={s.mean_current_a:7.4f}")
