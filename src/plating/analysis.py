# -*- coding: utf-8 -*-
"""真实数据交叉验证分析：清单、批量运行、结果矩阵、一致性与工况对照。"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .schema import CellData
from .readers import load
from .methods import run_all, METHOD_NAMES

def get_data_root() -> str:
    """
    数据根目录解析：
      1) 优先环境变量 PLATING_DATA_ROOT；
      2) 否则回退到仓库内 data/raw（用户自行下载数据集后放入）。
    """
    env = os.environ.get("PLATING_DATA_ROOT")
    if env:
        return os.path.abspath(env)
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "raw"))


# 向后兼容的模块级属性
DATA_ROOT = get_data_root()


def build_inventory(cfg: Config, n_matr: int = 5) -> List[Tuple[str, str]]:
    """返回 (source, path) 清单：Oxford 8 电芯 + NASA 3 电芯 + MATR-10 若干。"""
    data_root = get_data_root()
    jobs = [
        ("oxford", os.path.join(data_root, "Oxford_OBD1",
                                "Oxford_Battery_Degradation_Dataset_1.mat")),
    ]
    nasa_dir = os.path.join(data_root, "NASA_PCoE",
                            "1. BatteryAgingARC-FY08Q4")
    for f in ("B0005.mat", "B0006.mat", "B0007.mat"):
        jobs.append(("nasa", os.path.join(nasa_dir, f)))
    matr_dir = os.path.join(data_root, "MATR-10_DiagnosticFree", "ResValData")
    files = sorted(os.listdir(matr_dir))[:n_matr]
    for f in files:
        jobs.append(("arbin", os.path.join(matr_dir, f)))
    return jobs


def load_all_cells(cfg: Config, n_matr: int = 5) -> List[CellData]:
    cells: List[CellData] = []
    for src, path in build_inventory(cfg, n_matr):
        cells.extend(load(src, path))
    return cells


def run_inventory(cfg: Config, n_matr: int = 5):
    """运行全部电芯，返回 (cells, 每电芯结果字典)。"""
    cells = load_all_cells(cfg, n_matr)
    results = {}
    for cell in cells:
        results[cell.cell_id] = run_all(cell, cfg)
    return cells, results


def long_result_table(cells: List[CellData], results: Dict) -> pd.DataFrame:
    """逐循环/段长表。"""
    rows = []
    for cell in cells:
        for m in METHOD_NAMES:
            for r in results[cell.cell_id][m]:
                rows.append(r.as_row())
    return pd.DataFrame(rows)


def cell_level_table(cells: List[CellData], results: Dict) -> pd.DataFrame:
    """电芯级：每方法是否阳性、首次阳性循环、阳性计数。"""
    rows = []
    for cell in cells:
        row = {"cell_id": cell.cell_id, "source": cell.source,
               "n_cycles": len(cell.cycles)}
        for m in METHOD_NAMES:
            pos = [r.cycle for r in results[cell.cell_id][m] if r.positive]
            row[f"{m}_positive"] = len(pos) > 0
            row[f"{m}_first"] = min(pos) if pos else None
            row[f"{m}_n"] = len(pos)
        rows.append(row)
    return pd.DataFrame(rows)


def kappa_matrix(cell_tbl: pd.DataFrame) -> pd.DataFrame:
    """电芯级二分类两两 Cohen Kappa。"""
    from sklearn.metrics import cohen_kappa_score

    labels = {m: cell_tbl[f"{m}_positive"].astype(int).to_numpy()
              for m in METHOD_NAMES}
    mat = pd.DataFrame(np.eye(len(METHOD_NAMES)), index=METHOD_NAMES,
                       columns=METHOD_NAMES)
    for i, a in enumerate(METHOD_NAMES):
        for b in METHOD_NAMES[i + 1:]:
            k = cohen_kappa_score(labels[a], labels[b])
            mat.loc[a, b] = mat.loc[b, a] = round(float(k), 3)
    return mat


def operating_condition_table(cells: List[CellData]) -> pd.DataFrame:
    """工况对照：每电芯来源、温度、典型倍率与是否有真实 rest。"""
    rows = []
    for cell in cells:
        rests = cell.all_segments("rest")
        temps, rates = [], []
        for cyc in cell.cycles:
            if cyc.ambient_temperature_c is not None:
                temps.append(cyc.ambient_temperature_c)
            for seg in cyc.segments:
                if seg.segment_type == "charge":
                    rates.append(abs(seg.mean_current_a)
                                 / max(cell.nominal_capacity_ah, 1e-9))
        rows.append({
            "cell_id": cell.cell_id, "source": cell.source,
            "ambient_c": round(float(np.median(temps)), 1) if temps else None,
            "median_charge_c": round(float(np.median(rates)), 2) if rates else None,
            "has_real_rest": len(rests) > 0,
            "n_rest": len(rests),
        })
    return pd.DataFrame(rows)
