# -*- coding: utf-8 -*-
"""诊断 M4 在高倍率/极端工况失效的具体环节。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from plating.config import load_config
from plating.synthetic import generate_stress_dataset
from plating.methods.m4_kink import _cc_charge_curve


def main():
    cfg = load_config()
    for sc in ["high_rate_25c_3c", "extreme_cold_minus10_2c", "baseline_25c_1c"]:
        cells = generate_stress_dataset(cfg, sc, n_cells=8, n_cycles=120)
        plated = [c for c in cells if c.meta.get("plating")][0]
        n_base = cfg.get("m4_kink", "baseline_cycles", default=10)
        valid = 0
        sample_n = []
        for cyc in plated.cycles:
            ch = cyc.by_type("charge")
            if not ch:
                continue
            out = _cc_charge_curve(ch[0], cfg)
            if out is not None:
                valid += 1
            sample_n.append(len(ch[0]))
        print(f"=== {sc} ===")
        print(f"总循环={len(plated.cycles)} M4有效曲线数={valid} "
              f"需要>={n_base + 2}")
        print(f"充电段点数 前5={sample_n[:5]} 最小={min(sample_n)}")


if __name__ == "__main__":
    main()
