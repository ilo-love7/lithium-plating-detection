# -*- coding: utf-8 -*-
"""确认三数据集读取器返回结构。"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from plating.readers import load


def main():
    jobs = [
        ("oxford", r"D:\项目\02_数据集\01_国际公开集\Oxford_OBD1\Oxford_Battery_Degradation_Dataset_1.mat"),
        ("nasa", r"D:\项目\02_数据集\01_国际公开集\NASA_PCoE\1. BatteryAgingARC-FY08Q4\B0005.mat"),
        ("arbin", r"D:\项目\02_数据集\01_国际公开集\MATR-10_DiagnosticFree\ResValData\ResVal_000013_000185.001"),
    ]
    for src, path in jobs:
        t0 = time.time()
        cells = load(src, path)
        dt = time.time() - t0
        print(f"{src}: {len(cells)} 电芯, 耗时 {dt:.1f}s")
        for c in cells[:2]:
            s = c.summary()
            print("   ", s)


if __name__ == "__main__":
    main()
