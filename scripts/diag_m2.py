# -*- coding: utf-8 -*-
"""诊断 M2 低倍率 dQ/dV 峰量级。"""
import os
import sys

import numpy as np
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from plating.config import load_config
from plating.synthetic import generate_dataset
from plating.methods.common import sg_smooth
from plating.methods.m2_dqdv import _cumulative_ah, extract_peaks


def main():
    cfg = load_config()
    cells = generate_dataset(cfg, 8, 120)
    for ci in (0, 100):
        seg = cells[0].cycles[ci].by_type("charge")[0]
        v = seg.voltage
        q = _cumulative_ah(seg)
        order = np.argsort(v)
        v, q = v[order], q[order]
        keep = np.sort(np.unique(v, return_index=True)[1])
        v, q = v[keep], q[keep]
        dqdv = sg_smooth(np.gradient(q, v), 51, 3)
        print(f"cycle {ci}: dQ/dV 范围=({dqdv.min():.2f},{dqdv.max():.2f}) "
              f"中位={np.median(dqdv):.2f}")
        # 用不同 prominence 看峰数
        for prom in (0.5, 2, 5, 10):
            peaks, _ = find_peaks(dqdv, prominence=prom)
            print(f"  prom={prom}: 峰数={len(peaks)} 位置={np.round(v[peaks], 3)}")
        pk = extract_peaks(seg, cfg)
        print(f"  extract_peaks: 峰数={len(pk['positions'])} 位置={np.round(pk['positions'], 3)}")


if __name__ == "__main__":
    main()
