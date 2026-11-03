# -*- coding: utf-8 -*-
"""诊断统一网格上 dQ/dV 峰 prominence。"""
import os
import sys

import numpy as np
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from plating.config import load_config
from plating.readers import load
from plating.methods.m2_dqdv import _dqdv_on_grid, _is_low_rate

path = r"D:\项目\02_数据集\01_国际公开集\Oxford_OBD1\Oxford_Battery_Degradation_Dataset_1.mat"
cell = load("oxford", path)[0]
cfg = load_config()
seg = min([s for s in cell.cycles[0].by_type("charge") if _is_low_rate(s, cell, cfg)],
          key=lambda s: abs(s.mean_current_a))
v, dqdv = _dqdv_on_grid(seg, cfg)
fin = dqdv[np.isfinite(dqdv)]
print("dQ/dV: min", fin.min(), "max", fin.max(), "median", np.median(fin),
      "ptp", np.ptp(fin))
peaks, props = find_peaks(dqdv, prominence=0.0, distance=10)
order = np.argsort(props["prominences"])[::-1]
for p in order:
    idx = peaks[p]
    print(f"  V={v[idx]:.3f} prom={props['prominences'][p]:.4f} h={dqdv[idx]:.3f}")
