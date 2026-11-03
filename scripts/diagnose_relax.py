# -*- coding: utf-8 -*-
"""诊断 NASA 充电段 CV/静置边界，并盘点其他数据集的真实弛豫段。"""
import os
import sys

import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))


def diag_nasa():
    p = r"D:\项目\02_数据集\01_国际公开集\NASA_PCoE\1. BatteryAgingARC-FY08Q4\B0005.mat"
    m = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
    cy = m["B0005"].cycle
    # 取一个 charge 段细看
    ch = next(c for c in cy if c.type == "charge")
    d = ch.data
    t, i, v = d.Time, d.Current_measured, d.Voltage_measured
    print("NASA charge 总点数", len(t), "时长(s)", round(t[-1], 1))
    # 电流分布
    for q in [0, 1, 5, 10, 25, 50]:
        k = int(len(i) * q / 100)
        print(f"  pct{q}: t={t[k]:.0f} I={i[k]:.4f} V={v[k]:.4f}")
    # 电流严格为0的点
    zero = np.sum(np.abs(i) < 1e-6)
    print("  电流严格~0 点数:", zero)
    # 低电流区间电压是否恒定（CV特征：电压std极小）
    low = np.abs(i) < 0.04
    if np.sum(low) > 5:
        print("  低电流区电压 mean=%.4f std=%.5f" % (v[low].mean(), v[low].std()))


def peek_dir(name, path, depth=0):
    print("\n=====", name, "=====")
    if not os.path.exists(path):
        print("  路径不存在", path)
        return
    for root, dirs, files in os.walk(path):
        depth_now = root[len(path):].count(os.sep)
        if depth_now > depth:
            dirs[:] = []
            continue
        indent = "  " * (depth_now + 1)
        for f in files[:15]:
            fp = os.path.join(root, f)
            try:
                sz = os.path.getsize(fp)
            except OSError:
                sz = -1
            print(indent, f, round(sz / 1e6, 2), "MB")
        if depth_now == 0:
            for dd in dirs:
                print(indent, "[D]", dd)


if __name__ == "__main__":
    diag_nasa()
    base = r"D:\项目\02_数据集"
    peek_dir("MATR-10", os.path.join(base, "01_国际公开集", "MATR-10_DiagnosticFree"), depth=1)
    peek_dir("同济大学", os.path.join(base, "02_国内公开集", "同济大学_电压弛豫"), depth=1)
