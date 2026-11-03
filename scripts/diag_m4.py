# -*- coding: utf-8 -*-
"""诊断 M4 dV/dQ 曲线与局部极小的 prominence 分布。"""
import os
import sys

import numpy as np
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from plating.config import load_config
from plating.readers import load
from plating.cleaning import clean_cell
from plating.methods.m4_kink import _cumulative_ah
from plating.methods.common import sg_smooth

cfg = load_config()
f = r"D:\项目\02_数据集\01_国际公开集\MATR-10_DiagnosticFree\ResValData\ResVal_000013_000185.001"
cell = clean_cell(load("arbin", f)[0], cfg)[0]

# 取一个完整充电段
ch = next(s for s in cell.all_segments("charge") if len(s) > 2000)
q = _cumulative_ah(ch)
v = ch.voltage
dq = np.gradient(q)
dq[np.abs(dq) < 1e-9] = 1e-9
for win in [31, 101, 201]:
    dvdq = sg_smooth(np.gradient(v) / dq, win, 3)
    soc = np.clip((q - q[0]) / (q[-1] - q[0]), 0, 1)
    m = (soc >= 0.2) & (soc <= 0.9)
    d = dvdq[m]
    peaks, props = find_peaks(-d, prominence=0.05)
    print(f"win={win:4d} npts={len(d):5d} dV/dQ range=({d.min():.3f},{d.max():.3f}) "
          f"peaks(prom>.05)={len(peaks)}")
    if len(peaks):
        pr = props["prominences"]
        print("   prominences: min/median/max =", round(pr.min(), 4),
              round(np.median(pr), 4), round(pr.max(), 4))
# 看 dV/dQ 整体形态（抽样）
dvdq = sg_smooth(np.gradient(v) / dq, 101, 3)
for pct in range(0, 101, 10):
    k = int(len(dvdq) * pct / 100)
    print(f"  pct{pct:3d} soc={soc[k]:.2f} dvdq={dvdq[k]:.4f}")
