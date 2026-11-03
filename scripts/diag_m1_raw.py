# -*- coding: utf-8 -*-
"""诊断合成 M1 各 rest 原始肩部检测。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from plating.config import load_config
from plating.synthetic import SyntheticBattery, _PlatingSpec
from plating.methods.m1_relaxation import detect

cfg = load_config()
b = SyntheticBattery(cfg, "Synth_08", _PlatingSpec(True, 80))
cell = b.build(150, 3600)
for cyc in cell.cycles[78:112]:
    seg = cyc.by_type("rest")[0]
    r = detect(seg, cell.cell_id, cyc.cycle_index, cfg)
    print("cycle", cyc.cycle_index, "raw_pos", r.positive, "score", round(r.score, 7),
          r.detail)
