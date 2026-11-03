# -*- coding: utf-8 -*-
"""诊断合成析锂电芯 M1/M4 的中间量。"""
import os
import sys

import numpy as np
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from plating.config import load_config
from plating.synthetic import generate_dataset
from plating.methods.m4_kink import _cc_charge_curve
from plating.methods.common import sg_smooth, fill_nonfinite


def main():
    cfg = load_config()
    cells = generate_dataset(cfg, n_cells=8, n_cycles=120)
    plated = cells[4]  # onset=80

    # ---- M1 诊断：对比循环 0 与 80 的 rest dV/dt ----
    print("=== M1 rest dV/dt ===")
    for ci in (0, 80, 100):
        rest = plated.cycles[ci].by_type("rest")[0]
        vs = sg_smooth(rest.voltage, 21, 3)
        dvdt = np.gradient(vs, rest.time)
        neg = -dvdt
        peaks, props = find_peaks(neg, prominence=0.0002)
        print(f"cycle {ci}: dvdt范围=({dvdt.min():.2e},{dvdt.max():.2e}) "
              f"峰数={len(peaks)} proms={np.round(props['prominences'], 2 + 6) if len(peaks) else []}")

    # ---- M4 诊断：循环 80/100 相对基准的 diff ----
    print("\n=== M4 diff vs baseline ===")
    grid = np.linspace(0.10, 0.95, 100)
    curves = {}
    for cyc in plated.cycles:
        ch = cyc.by_type("charge")[0]
        out = _cc_charge_curve(ch, cfg)
        if out is None:
            continue
        soc, dvdq = out
        m = (soc >= 0.10) & (soc <= 0.95)
        curves[cyc.cycle_index] = np.interp(grid, soc[m], dvdq[m])
    ids = sorted(curves)
    baseline = np.nanmedian(np.vstack([curves[c] for c in ids[:10]]), axis=0)
    search = (grid >= 0.50) & (grid <= 0.90)
    for ci in (80, 100, 119):
        diff = fill_nonfinite(curves[ci]) - baseline
        seg = -diff[search]
        peaks, props = find_peaks(seg, prominence=0.02)
        print(f"cycle {ci}: diff范围=({diff[search].min():.3e},{diff[search].max():.3e}) "
              f"峰数={len(peaks)}")
        for p in peaks:
            gi = np.where(search)[0][p]
            local = abs(np.median(baseline[max(0, gi - 5):gi + 6]))
            rel = props["prominences"][list(peaks).index(p)] / max(local, 1e-9)
            print(f"   soc={grid[gi]:.2f} prom={props['prominences'][list(peaks).index(p)]:.3e} rel={rel:.3f}")


if __name__ == "__main__":
    main()
