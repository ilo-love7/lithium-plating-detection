# -*- coding: utf-8 -*-
"""重放 M4 内部计算，打印高倍率工况下 kink 的相对 prominence。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import numpy as np
from scipy.signal import find_peaks

from plating.config import load_config
from plating.synthetic import generate_stress_dataset
from plating.methods.m4_kink import _cc_charge_curve
from plating.methods.common import fill_nonfinite


def replay(cfg, sc):
    cells = generate_stress_dataset(cfg, sc, n_cells=8, n_cycles=120)
    cell = [c for c in cells if c.meta.get("plating")][0]
    grid_n = cfg.get("m4_kink", "soc_grid_n", default=100)
    soc_min = cfg.get("m4_kink", "soc_min", default=0.10)
    soc_max = cfg.get("m4_kink", "soc_max", default=0.95)
    grid = np.linspace(soc_min, soc_max, grid_n)
    n_base = cfg.get("m4_kink", "baseline_cycles", default=10)
    min_prom = cfg.get("m4_kink", "min_prominence", default=0.02)
    rel_thr = cfg.get("m4_kink", "rel_prominence", default=0.30)

    curves = {}
    for cyc in cell.cycles:
        ch = cyc.by_type("charge")
        if not ch:
            continue
        out = _cc_charge_curve(ch[0], cfg)
        if out is None:
            continue
        soc, dvdq = out
        msk = (soc >= soc_min) & (soc <= soc_max)
        if msk.sum() < 20:
            continue
        curves[cyc.cycle_index] = np.interp(grid, soc[msk], dvdq[msk])

    ids = sorted(curves)
    base = fill_nonfinite(np.nanmedian(
        np.vstack([curves[c] for c in ids[:n_base]]), axis=0))
    k_lo = cfg.get("m4_kink", "kink_min_soc", default=0.50)
    k_hi = cfg.get("m4_kink", "kink_max_soc", default=0.90)
    search_idx = np.where((grid >= k_lo) & (grid <= k_hi))[0]
    max_rels = []
    hit_ids = []
    for cid in ids[n_base:]:
        diff = fill_nonfinite(curves[cid]) - base
        seg = -diff[search_idx]
        peaks, props = find_peaks(seg, prominence=min_prom)
        rel = 0.0
        for p in peaks:
            gi = search_idx[p]
            local_base = abs(np.median(base[max(0, gi - 5):gi + 6]))
            r = props["prominences"][list(peaks).index(p)] / max(local_base, 1e-9)
            rel = max(rel, r)
        max_rels.append(rel)
        if rel > rel_thr:
            hit_ids.append(cid)
    print(f"=== {sc} === rel阈值={rel_thr} min_prom={min_prom}")
    print(f"后期循环最大rel: 最大={max(max_rels):.4f} "
          f"中位={np.median(max_rels):.4f} 达标循环数={len(hit_ids)}")
    print(f"命中循环索引: {hit_ids}")
    diag_every = cfg.get("synthetic", "diagnostic_every", default=10)
    print(f"低倍率诊断段出现在每 {diag_every} 循环的倍数处")


def main():
    cfg = load_config()
    for sc in ["baseline_25c_1c", "high_rate_25c_3c"]:
        replay(cfg, sc)


if __name__ == "__main__":
    main()
