# -*- coding: utf-8 -*-
"""退化对齐分析：生成合成数据，输出报警时机对齐表、汇总与配图。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from plating.config import load_config
from plating.synthetic import generate_dataset
from plating.degradation import (
    alignment_table, summarize_alignment, plot_alignment,
)


def main(n_cells=24, n_cycles=140):
    cfg = load_config()
    cells = generate_dataset(cfg, n_cells=n_cells, n_cycles=n_cycles)

    table = alignment_table(cfg, cells)
    summary = summarize_alignment(table)

    fig_dir = "results/figures"
    os.makedirs(fig_dir, exist_ok=True)
    plot_alignment(table, os.path.join(fig_dir, "fig6_degradation_alignment.png"))

    table.to_csv("results/degradation_alignment_detail.csv",
                 index=False, encoding="utf-8-sig")
    summary.to_csv("results/degradation_alignment_summary.csv",
                   index=False, encoding="utf-8-sig")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
