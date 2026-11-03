# -*- coding: utf-8 -*-
"""诊断 Arbin / Oxford CC 段 dV/dQ 候选 kink 形态。"""
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


def analyze(label, seg, cap_ah):
    q = _cumulative_ah(seg)
    v = seg.voltage
    dq = np.gradient(q)
    dq[np.abs(dq) < 1e-9] = 1e-9
    dvdq = sg_smooth(np.gradient(v) / dq, 101, 3)
    soc = np.clip((q - q[0]) / (q[-1] - q[0]), 0, 1)
    cc_rated = np.percentile(np.abs(seg.current), 90)
    m = (soc >= 0.2) & (soc <= 0.9) & (np.abs(seg.current) >= 0.85 * cc_rated)
    d = dvdq[m]
    if len(d) < 10:
        print(label, "CC 窗口不足")
        return
    lvl = np.median(d)
    peaks, props = find_peaks(-d, prominence=0.05)
    print(f"\n{label}: CC npts={len(d)} median dV/dQ={lvl:.4f} range=({d.min():.3f},{d.max():.3f})")
    if len(peaks):
        for j, p in enumerate(peaks):
            rel = props["prominences"][j] / lvl
            print(f"   valley soc~{soc[np.where(m)[0][p]]:.2f} abs_prom={props['prominences'][j]:.4f} rel={rel:.3f}")


ar = clean_cell(load("arbin", r"D:\项目\02_数据集\01_国际公开集\MATR-10_DiagnosticFree\ResValData\ResVal_000013_000185.001")[0], cfg)[0]
ach = next(s for s in ar.all_segments("charge") if len(s) > 2000)
analyze("Arbin C/36", ach, 4.84)

ox = clean_cell(load("oxford", r"D:\项目\02_数据集\01_国际公开集\Oxford_OBD1\Oxford_Battery_Degradation_Dataset_1.mat")[0], cfg)[0]
# 取一个 C1ch（1C）
c1 = next(s for s in ox.cycles[0].by_type("charge") if s.name == "C1ch")
analyze("Oxford C1ch(1C)", c1, 0.74)
ocv = next(s for s in ox.cycles[0].by_type("charge") if s.name == "OCVch")
analyze("Oxford OCVch(0.05C)", ocv, 0.74)
