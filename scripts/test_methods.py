# -*- coding: utf-8 -*-
"""M1~M4 在真实数据上的冒烟测试。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from plating.config import load_config
from plating.readers import load
from plating.cleaning import clean_cell
from plating.methods import run_all, METHOD_NAMES

DATA = r"D:\项目\02_数据集\01_国际公开集"


def run_case(label, path, source):
    print("\n" + "#" * 60)
    print("CASE:", label)
    cfg = load_config()
    cell = load(source, path)[0]
    cell, _ = clean_cell(cell, cfg)
    print("cell:", cell.cell_id, "cycles", len(cell.cycles))
    results = run_all(cell, cfg)
    for m in METHOD_NAMES:
        rs = results[m]
        pos = sum(1 for r in rs if r.positive)
        print(f"  {m}: n_results={len(rs):3d} positives={pos:3d}")
        if rs:
            print("    示例:", {k: v for k, v in rs[0].as_row().items()
                          if k in ("cycle", "positive", "score", "threshold")})


if __name__ == "__main__":
    run_case("Arbin/MATR-10",
             os.path.join(DATA, "MATR-10_DiagnosticFree", "ResValData",
                          "ResVal_000013_000185.001"), "arbin")
    run_case("Oxford OBD1",
             os.path.join(DATA, "Oxford_OBD1",
                          "Oxford_Battery_Degradation_Dataset_1.mat"), "oxford")
    run_case("NASA B0005",
             os.path.join(DATA, "NASA_PCoE", "1. BatteryAgingARC-FY08Q4",
                          "B0005.mat"), "nasa")
