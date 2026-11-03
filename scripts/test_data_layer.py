# -*- coding: utf-8 -*-
"""数据层冒烟测试：读取 -> 清洗 -> 弛豫段检测（真实数据）。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from plating.config import load_config
from plating.readers import load
from plating.cleaning import clean_cell, quality_summary
from plating.relaxation import detect_cell, events_to_index_table

DATA = r"D:\项目\02_数据集\01_国际公开集"


def main():
    cfg = load_config()
    print("config loaded:", cfg.get("project", "name"))

    # ---- Arbin (MATR-10) 单文件 ----
    arbin_fp = os.path.join(DATA, "MATR-10_DiagnosticFree", "ResValData",
                            "ResVal_000013_000185.001")
    ar_cells = load("arbin", arbin_fp)
    ac = ar_cells[0]
    print("\n[Arbin/MATR-10]", ac.summary())
    ac_clean, ac_q = clean_cell(ac, cfg)
    print("  clean:", quality_summary(ac_q))
    ac_ev = detect_cell(ac_clean, cfg)
    print("  relaxation events:", len(ac_ev))
    print("  CV suspects:", sum(1 for e in ac_ev if e.is_cv_suspect))
    if ac_ev:
        print("  示例事件:", ac_ev[0].as_row())

    # ---- Oxford ----
    ox_path = os.path.join(DATA, "Oxford_OBD1", "Oxford_Battery_Degradation_Dataset_1.mat")
    ox_cells = load("oxford", ox_path)
    print("\n[Oxford] cells:", len(ox_cells), ox_cells[0].summary())
    oc = ox_cells[0]
    oc_clean, _ = clean_cell(oc, cfg)
    oc_ev = detect_cell(oc_clean, cfg)
    print("  relaxation events:", len(oc_ev))

    # ---- NASA（CV 应被排除）----
    nasa_path = os.path.join(DATA, "NASA_PCoE", "1. BatteryAgingARC-FY08Q4", "B0005.mat")
    nasa_cells = load("nasa", nasa_path)
    nc = nasa_cells[0]
    nc_clean, nc_q = clean_cell(nc, cfg)
    nc_ev = detect_cell(nc_clean, cfg)
    print("\n[NASA]", nc.summary())
    print("  clean:", quality_summary(nc_q))
    print("  relaxation events after CV rejection:", len(nc_ev))


if __name__ == "__main__":
    main()
