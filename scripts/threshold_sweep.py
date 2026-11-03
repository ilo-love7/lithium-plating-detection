# -*- coding: utf-8 -*-
"""阈值敏感性扫描：量化阳性结论对各方法关键阈值的依赖。"""
import os
import sys
import copy

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from plating.config import load_config, Config
from plating.analysis import load_all_cells
from plating.methods import METHOD_NAMES
from plating.methods.m1_relaxation import run_cell as m1
from plating.methods.m2_dqdv import run_cell as m2
from plating.methods.m3_coulomb import run_cell as m3
from plating.methods.m4_kink import run_cell as m4

RUNNERS = {"M1": m1, "M2": m2, "M3": m3, "M4": m4}
# (section, key)
PARAMS = {
    "M1": ("m1_relaxation", "residual_prominence"),
    "M2": ("m2_dqdv", "shift_alarm_v"),
    "M3": ("m3_coulomb", "ce_drop_alarm"),
    "M4": ("m4_kink", "min_prominence"),
}
MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5]


def main():
    base_cfg = load_config()
    print("加载电芯（仅一次）...")
    cells = load_all_cells(base_cfg, 5)
    rows = []
    for m in METHOD_NAMES:
        section, key = PARAMS[m]
        base_val = base_cfg.get(section, key)
        runner = RUNNERS[m]
        for mult in MULTIPLIERS:
            data = base_cfg.as_dict()
            val = base_val * mult
            data[section][key] = val
            cfg = Config(data)
            n_pos = 0
            for cell in cells:
                if any(r.positive for r in runner(cell, cfg)):
                    n_pos += 1
            rows.append({"method": m, "param": key, "multiplier": mult,
                         "value": round(float(val), 6),
                         "n_positive_cells": n_pos})
            print(f"{m} {key} x{mult}: {n_pos} 阳性电芯")
    tbl = pd.DataFrame(rows)
    out = os.path.join(ROOT, "results", "threshold_sensitivity.csv")
    tbl.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n", tbl.to_string(index=False))


if __name__ == "__main__":
    main()
