# -*- coding: utf-8 -*-
"""诊断 NASA B0005 的 M3 CE 序列。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from plating.config import load_config
from plating.readers import load
from plating.methods.m3_coulomb import run_cell

path = r"D:\项目\02_数据集\01_国际公开集\NASA_PCoE\1. BatteryAgingARC-FY08Q4\B0005.mat"
cell = load("nasa", path)[0]
res = run_cell(cell, load_config())
print("阳性数:", sum(1 for r in res if r.positive))
for r in res[:18]:
    print(f"cycle {r.cycle}: pos={r.positive} score={r.score:.4f} detail={r.detail}")
