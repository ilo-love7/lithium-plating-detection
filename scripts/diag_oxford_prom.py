# -*- coding: utf-8 -*-
"""诊断 Oxford dQ/dV 峰 prominence 排序。"""
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
from plating.methods.m2_dqdv import _cumulative_ah, _is_low_rate

path = r"D:\项目\02_数据集\01_国际公开集\Oxford_OBD1\Oxford_Battery_Degradation_Dataset_1.mat"
cell = load("oxford", path)[0]
cfg = load_config()
for cyc in cell.cycles[:1]:
    seg = [s for s in cyc.by_type("charge") if _is_low_rate(s, cell, cfg)][0]
    v = seg.voltage
    q = _cumulative_ah(seg)
    order = np.argsort(v)
    v, q = v[order], q[order]
    keep = np.sort(np.unique(v, return_index=True)[1])
    v, q = v[keep], q[keep]
    dqdv = sg_smooth(np.gradient(q, v), 51, 3)
    peaks, props = find_peaks(dqdv, prominence=0.1)
    order_p = np.argsort(props["prominences"])[::-1]
    print("全部峰（按prominence降序）:")
    for p in order_p:
        idx = peaks[p]
        print(f"  V={v[idx]:.3f} prom={props['prominences'][p]:.3f} height={dqdv[idx]:.2f}")
