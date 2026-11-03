# -*- coding: utf-8 -*-
"""确定 Oxford OCVch 的合适平滑窗口。"""
import os
import sys

import numpy as np
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from plating.config import load_config
from plating.readers import load
from plating.methods.common import sg_smooth
from plating.methods.m2_dqdv import _cumulative_ah

path = r"D:\项目\02_数据集\01_国际公开集\Oxford_OBD1\Oxford_Battery_Degradation_Dataset_1.mat"
cell = load("oxford", path)[0]
seg = cell.cycles[0].by_type("charge")[1]  # OCVch
v = seg.voltage
q = _cumulative_ah(seg)
order = np.argsort(v)
v, q = v[order], q[order]
keep = np.sort(np.unique(v, return_index=True)[1])
v, q = v[keep], q[keep]
print(f"OCVch npts={len(v)} 电压范围=({v.min():.2f},{v.max():.2f})")
raw = np.gradient(q, v)
for win in (51, 151, 301, 601):
    dqdv = sg_smooth(raw, win, 3)
    # prominence 用曲线动态范围的比例
    prom = 0.3 * np.median(dqdv)
    peaks, _ = find_peaks(dqdv, prominence=prom)
    print(f"win={win}: 峰数={len(peaks)} 位置={np.round(v[peaks], 3)}")
