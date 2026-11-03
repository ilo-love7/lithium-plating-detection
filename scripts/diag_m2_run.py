# -*- coding: utf-8 -*-
"""在 run_cell 层面诊断 Synth_04 的 M2 判定。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from plating.config import load_config
from plating.synthetic import generate_dataset
from plating.methods.m2_dqdv import run_cell


def main():
    cfg = load_config()
    cell = generate_dataset(cfg, 8, 120)[4]
    for r in run_cell(cell, cfg):
        print(f"cycle {r.cycle}: pos={r.positive} score={r.score:.4f} detail={r.detail}")


if __name__ == "__main__":
    main()
