# -*- coding: utf-8 -*-
"""验证 M1 单调基线残差法。"""
import os
import sys

import numpy as np
from scipy.signal import find_peaks
from sklearn.isotonic import IsotonicRegression

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from plating.config import load_config
from plating.synthetic import generate_dataset
from plating.methods.common import sg_smooth


def residual_shoulder(rest, win=21, sm_win=11):
    vs = sg_smooth(rest.voltage, sm_win, 3)
    rate = -np.gradient(vs, rest.time)
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    base = iso.fit_transform(rest.time, rate)
    return rate - base, rate


def main():
    cfg = load_config()
    cells = generate_dataset(cfg, 8, 120)
    print("=== 对照电芯残差峰 ===")
    for ci in (0, 50, 100):
        rest = cells[0].cycles[ci].by_type("rest")[0]
        res, rate = residual_shoulder(rest)
        peaks, props = find_peaks(res, prominence=1e-6)
        mx = props['prominences'].max() if len(peaks) else 0
        print(f"ctrl {ci}: 残差峰数={len(peaks)} 最大prom={mx:.2e}")
    print("=== 析锂电芯残差峰 ===")
    for ci in (80, 90, 100, 110):
        rest = cells[4].cycles[ci].by_type("rest")[0]
        res, rate = residual_shoulder(rest)
        peaks, props = find_peaks(res, prominence=1e-6)
        mx = props['prominences'].max() if len(peaks) else 0
        print(f"plated {ci}: 残差峰数={len(peaks)} 最大prom={mx:.2e}")


if __name__ == "__main__":
    main()
