# -*- coding: utf-8 -*-
"""读取器包：导入即注册，提供统一 load 入口。"""
from typing import List, Optional

from ..schema import CellData
from . import arbin, generic, nasa, oxford  # noqa: F401  触发注册
from .base import BaseReader, available_readers, get_reader


def load(source: str, path: str, nominal_capacity_ah: Optional[float] = None) -> List[CellData]:
    """高层入口：指定来源名与路径，读取 CellData 列表。"""
    reader: BaseReader = get_reader(source, nominal_capacity_ah=nominal_capacity_ah)
    return reader.read(path)


def auto_load(path: str, nominal_capacity_ah: Optional[float] = None) -> List[CellData]:
    """按路径/文件名猜测来源。"""
    low = path.lower()
    if "oxford" in low:
        return load("oxford", path, nominal_capacity_ah)
    if "nasa" in low or os_basename(low).startswith("b0"):
        return load("nasa", path, nominal_capacity_ah)
    if "resval" in low or "matr-10" in low or "matr10" in low:
        return load("arbin", path, nominal_capacity_ah)
    return load("generic", path, nominal_capacity_ah)


def os_basename(p: str) -> str:
    import os

    return os.path.basename(p)


__all__ = ["load", "auto_load", "get_reader", "available_readers", "BaseReader"]
