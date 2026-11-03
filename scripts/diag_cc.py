# -*- coding: utf-8 -*-
"""诊断充电段 CC/CV 电流模式。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from plating.config import load_config
from plating.readers import load
from plating.cleaning import clean_cell

cfg = load_config()

# NASA 充电段
nasa = clean_cell(load("nasa", r"D:\项目\02_数据集\01_国际公开集\NASA_PCoE\1. BatteryAgingARC-FY08Q4\B0005.mat")[0], cfg)[0]
ch = nasa.cycles[0].by_type("charge")[0]
i = np.abs(ch.current)
print("NASA charge: n=", len(i), "I p90=", round(np.percentile(i, 90), 4),
      "I max=", round(i.max(), 4), "I last=", round(i[-1], 5))
for frac in [0.3, 0.5, 0.7, 0.8, 0.9]:
    k = int(len(i) * frac)
    print(f"  frac{frac}: I={i[k]:.4f}")
cc_mask = i >= 0.8 * np.percentile(i, 90)
print("  CC 占比:", round(cc_mask.mean(), 3))

# Arbin 充电段（慢充，看是否恒流）
ar = clean_cell(load("arbin", r"D:\项目\02_数据集\01_国际公开集\MATR-10_DiagnosticFree\ResValData\ResVal_000013_000185.001")[0], cfg)[0]
ach = next(s for s in ar.all_segments("charge") if len(s) > 2000)
ai = np.abs(ach.current)
print("\nArbin charge: n=", len(ai), "I median=", round(np.median(ai), 4),
      "I std/median=", round(ai.std() / np.median(ai), 4), "I last=", round(ai[-1], 5))
