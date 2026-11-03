# -*- coding: utf-8 -*-
"""逐对照电芯输出 M2 每循环峰移。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from plating.config import load_config
from plating.synthetic import generate_dataset
from plating.methods.m2_dqdv import extract_peaks, _is_low_rate


def main():
    cfg = load_config()
    cells = generate_dataset(cfg, 8, 120)
    for cell in cells[4:5]:  # 首个析锂电芯
        recs = []
        for cyc in cell.cycles:
            cands = [s for s in cyc.by_type("charge") if _is_low_rate(s, cell, cfg)]
            if not cands:
                continue
            pk = extract_peaks(cands[0], cfg)
            if pk is not None and len(pk["positions"]):
                recs.append((cyc.cycle_index, pk["positions"]))
        base = recs[0][1]
        print(f"\n{cell.cell_id} 基准峰={np.round(base, 3)}")
        for ci, pos in recs:
            shifts = [abs(pos[np.argmin(np.abs(pos - bp))] - bp) for bp in base]
            print(f"  cycle {ci}: n={len(pos)} maxshift={max(shifts):.4f} pos={np.round(pos, 3)}")


if __name__ == "__main__":
    main()
