# -*- coding: utf-8 -*-
"""确认修复后 M4 在 3C 常规段的真实 rel 与判定（排除诊断段）。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import numpy as np

from plating.config import load_config
from plating.synthetic import generate_stress_dataset
from plating.methods.m4_kink import detect as m4_detect


def main():
    cfg = load_config()
    rel_thr = cfg.get("m4_kink", "rel_prominence", default=0.30)
    for sc in ["baseline_25c_1c", "high_rate_25c_3c", "extreme_cold_minus10_2c"]:
        cells = generate_stress_dataset(cfg, sc, n_cells=8, n_cycles=120)
        plated = [c for c in cells if c.meta.get("plating")][0]
        res = m4_detect(plated, cfg)
        print(f"{sc}: M4阳性循环数={len(res)} rel阈值={rel_thr}")


if __name__ == "__main__":
    main()
