# -*- coding: utf-8 -*-
"""检测方法包：M1~M4 统一调度。"""
from typing import Dict, List

from ..schema import CellData
from ..config import Config
from .common import MethodResult
from . import m1_relaxation, m2_dqdv, m3_coulomb, m4_kink

_RUNNERS = {
    "M1": m1_relaxation.run_cell,
    "M2": m2_dqdv.run_cell,
    "M3": m3_coulomb.run_cell,
    "M4": m4_kink.run_cell,
}

METHOD_NAMES = list(_RUNNERS)


def run_method(name: str, cell: CellData, cfg: Config) -> List[MethodResult]:
    if name not in _RUNNERS:
        raise KeyError(f"未知方法 {name}，可选：{METHOD_NAMES}")
    return _RUNNERS[name](cell, cfg)


def run_all(cell: CellData, cfg: Config) -> Dict[str, List[MethodResult]]:
    return {name: _RUNNERS[name](cell, cfg) for name in METHOD_NAMES}


__all__ = ["run_method", "run_all", "METHOD_NAMES", "MethodResult"]
