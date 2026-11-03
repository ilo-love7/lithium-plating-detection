# -*- coding: utf-8 -*-
"""检查 Arbin 各 rest 段的电流纯净度、电压水平与弛豫幅度。"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from plating.readers import load

f = r"D:\项目\02_数据集\01_国际公开集\MATR-10_DiagnosticFree\ResValData\ResVal_000013_000185.001"
cell = load("arbin", f)[0]

rests = cell.all_segments("rest")
print("rest 段数:", len(rests))
for k, s in enumerate(rests):
    nz = int(np.sum(np.abs(s.current) > 1e-9))
    dv = s.voltage[0] - s.voltage[-1]
    print(f"  rest{k:2d} n={len(s):4d} dur={s.duration_s:6.0f} nonzeroI={nz:3d} "
          f"V0={s.voltage[0]:.4f} V1={s.voltage[-1]:.4f} dV={dv:+.5f} "
          f"Vstd={np.std(s.voltage):.5f}")
