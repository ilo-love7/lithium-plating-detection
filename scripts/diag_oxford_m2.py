# -*- coding: utf-8 -*-
"""诊断 Oxford_Cell1 新 M2 峰位置与顺序匹配。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from plating.config import load_config
from plating.readers import load
from plating.methods.m2_dqdv import extract_peaks, _is_low_rate, _match_peaks

path = r"D:\项目\02_数据集\01_国际公开集\Oxford_OBD1\Oxford_Battery_Degradation_Dataset_1.mat"
cell = load("oxford", path)[0]
cfg = load_config()
recs = []
for cyc in cell.cycles:
    cands = [s for s in cyc.by_type("charge") if _is_low_rate(s, cell, cfg)]
    if not cands:
        continue
    pk = extract_peaks(min(cands, key=lambda s: abs(s.mean_current_a)), cfg)
    if pk is not None and len(pk["positions"]):
        recs.append((cyc.cycle_index, pk))
max_gap = cfg.get("m2_dqdv", "max_match_gap_v", default=0.04)
base = recs[0][1]["positions"]
print("基准峰:", np.round(base, 3), "共", len(recs), "特征化循环")
for ci, pk in recs[:12]:
    pairs = _match_peaks(base, pk["positions"], max_gap)
    shifts = [abs(c - b) for b, c in pairs]
    print(f"cycle {ci}: n={len(pk['positions'])} 匹配={len(pairs)} "
          f"maxshift={max(shifts) if shifts else 0:.4f} pos={np.round(pk['positions'], 3)}")
