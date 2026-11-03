# -*- coding: utf-8 -*-
"""合成数据真值验证：各方法灵敏度/特异度。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from plating.config import load_config
from plating.synthetic import generate_dataset, ground_truth
from plating.methods import run_all, METHOD_NAMES


def main():
    n_cells = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n_cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    cfg = load_config()
    cells = generate_dataset(cfg, n_cells=n_cells, n_cycles=n_cycles)
    gt = ground_truth(cells)
    print("生成电芯:", [(c.cell_id, gt[c.cell_id]["plating"]) for c in cells])

    # 汇总每方法在每电芯上的判定（任一循环阳性即电芯阳性，并记录首次阳性循环）
    for m in METHOD_NAMES:
        tp = fp = tn = fn = 0
        onset_errors = []
        for cell in cells:
            res = run_all(cell, cfg)[m]
            pos_cycles = [r.cycle for r in res if r.positive]
            detected = len(pos_cycles) > 0
            truth = gt[cell.cell_id]["plating"]
            if truth and detected:
                tp += 1
                onset_errors.append(min(pos_cycles) - gt[cell.cell_id]["onset"])
            elif truth and not detected:
                fn += 1
            elif not truth and detected:
                fp += 1
            else:
                tn += 1
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        print(f"\n{m}: 灵敏度={sens:.2f} 特异度={spec:.2f} (TP{tp} FN{fn} FP{fp} TN{tn})")
        if onset_errors:
            print("   首次检测相对真值起始的循环偏差:", onset_errors)


if __name__ == "__main__":
    main()
