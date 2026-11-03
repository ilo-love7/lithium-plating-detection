# -*- coding: utf-8 -*-
"""诊断对照电芯 M2/M3 误报与 M1 噪声底线。"""
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
from plating.methods.m2_dqdv import extract_peaks, _is_low_rate


def main():
    cfg = load_config()
    cells = generate_dataset(cfg, n_cells=8, n_cycles=120)
    ctrl, plated = cells[0], cells[4]

    # ---- M1 噪声底线：对照电芯 rest 的 -dv/dt ----
    print("=== M1 对照 dvdt 噪声 (window51) ===")
    for ci in (0, 50, 100):
        rest = ctrl.cycles[ci].by_type("rest")[0]
        vs = sg_smooth(rest.voltage, 21, 3)
        dvdt = np.gradient(vs, rest.time)
        peaks, props = find_peaks(-dvdt, prominence=1e-5)
        mx = props['prominences'].max() if len(peaks) else 0
        print(f"ctrl cycle {ci}: dvdt std={np.std(dvdt):.2e} 峰数={len(peaks)} 最大prom={mx:.2e}")
    # 析锂电芯各循环肩部
    print("=== M1 析锂 dvdt 肩部 (window51) ===")
    for ci in (80, 90, 100, 110):
        rest = plated.cycles[ci].by_type("rest")[0]
        vs = sg_smooth(rest.voltage, 21, 3)
        dvdt = np.gradient(vs, rest.time)
        peaks, props = find_peaks(-dvdt, prominence=1e-5)
        mx = props['prominences'].max() if len(peaks) else 0
        print(f"plated cycle {ci}: 峰数={len(peaks)} 最大prom={mx:.2e}")

    # ---- M2 对照峰位置随循环 ----
    print("\n=== M2 对照低倍率峰位置 ===")
    for cyc in ctrl.cycles:
        cands = [s for s in cyc.by_type("charge") if _is_low_rate(s, ctrl, cfg)]
        if not cands:
            continue
        pk = extract_peaks(cands[0], cfg)
        if pk is not None and len(pk["positions"]):
            print(f"cycle {cyc.cycle_index}: n={len(pk['positions'])} pos={np.round(pk['positions'], 3)}")

    # ---- M3 对照 CE 序列 ----
    print("\n=== M3 对照 CE（前 20 个有值循环） ===")
    from plating.methods.m3_coulomb import run_cell as m3_run
    r3 = m3_run(ctrl, cfg)
    print("M3 对照阳性数:", sum(1 for r in r3 if r.positive))
    for r in r3[:20]:
        print(f"cycle {r.cycle} pos={r.positive} score={r.score:.4f} detail={r.detail}")


if __name__ == "__main__":
    main()

