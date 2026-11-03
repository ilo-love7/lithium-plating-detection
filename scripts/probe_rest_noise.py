# -*- coding: utf-8 -*-
"""探测真实 MATR-10 rest 段的采样间隔与噪声水平。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from plating.config import load_config
from plating.readers import load


def main():
    path = (r"D:\项目\02_数据集\01_国际公开集\MATR-10_DiagnosticFree"
            r"\ResValData\ResVal_000013_000185.001")
    cell = load("arbin", path)[0]
    print("电芯", cell.cell_id, "循环数", len(cell.cycles))
    rests = cell.all_segments("rest")
    print("rest 段数:", len(rests))
    for r in rests[:4]:
        dt = np.diff(r.time)
        v = r.voltage
        # 高频测量噪声：SG 强平滑后的残差 std
        from plating.methods.common import sg_smooth
        sm = sg_smooth(v, 21, 3)
        noise = np.std(v - sm)
        print(f"npts={len(r)} 时长={r.duration_s:.0f}s 中位dt={np.median(dt):.1f}s "
              f"电压范围=({v.min():.4f},{v.max():.4f}) SG残差测量噪声std={noise:.2e}V")


if __name__ == "__main__":
    main()
