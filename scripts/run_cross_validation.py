# -*- coding: utf-8 -*-
"""真实数据多数据集交叉验证主入口。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from plating.config import load_config
from plating.analysis import (
    run_inventory, long_result_table, cell_level_table, kappa_matrix,
    operating_condition_table,
)


def main():
    n_matr = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    cfg = load_config()
    out_dir = os.path.join(ROOT, "results")
    print("加载并运行电芯清单 ...")
    cells, results = run_inventory(cfg, n_matr)
    print(f"共 {len(cells)} 电芯")

    long = long_result_table(cells, results)
    long.to_csv(os.path.join(out_dir, "result_matrix_long.csv"),
                index=False, encoding="utf-8-sig")

    cell_tbl = cell_level_table(cells, results)
    cell_tbl.to_csv(os.path.join(out_dir, "cell_level_summary.csv"),
                    index=False, encoding="utf-8-sig")

    kappa = kappa_matrix(cell_tbl)
    kappa.to_csv(os.path.join(out_dir, "method_kappa.csv"),
                 encoding="utf-8-sig")

    cond = operating_condition_table(cells)
    cond.to_csv(os.path.join(out_dir, "operating_conditions.csv"),
                index=False, encoding="utf-8-sig")

    print("\n=== 电芯级汇总 ===")
    print(cell_tbl.to_string(index=False))
    print("\n=== 方法一致性 Kappa ===")
    print(kappa.to_string())
    print("\n=== 工况对照 ===")
    print(cond.to_string(index=False))


if __name__ == "__main__":
    main()
