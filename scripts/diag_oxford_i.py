# -*- coding: utf-8 -*-
"""诊断 Oxford 充电段电流。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from plating.readers import load

path = r"D:\项目\02_数据集\01_国际公开集\Oxford_OBD1\Oxford_Battery_Degradation_Dataset_1.mat"
cell = load("oxford", path)[0]
for cyc in cell.cycles[:3]:
    for seg in cyc.by_type("charge"):
        dt = np.diff(seg.time)
        print(f"{seg.name}: n={len(seg)} 电流中位={np.median(seg.current):.3f} "
              f"mean={seg.mean_current_a:.3f} max={seg.current.max():.2f} "
              f"dt范围=({dt.min():.2f},{dt.max():.2f})")
