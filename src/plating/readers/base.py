# -*- coding: utf-8 -*-
"""读取器基类与注册表。"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Dict, List

from ..schema import CellData


class BaseReader(ABC):
    """所有数据集读取器的基类。"""

    source_name: str = "base"

    def __init__(self, nominal_capacity_ah: float = None):
        self.nominal_capacity_ah = nominal_capacity_ah

    @abstractmethod
    def read_path(self, path: str) -> List[CellData]:
        """读取一个文件或目录，返回若干 CellData。"""

    # 便捷入口
    def read(self, path: str) -> List[CellData]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return self.read_path(path)


_READERS: Dict[str, type] = {}


def register_reader(name: str):
    def deco(cls: type):
        cls.source_name = name
        _READERS[name] = cls
        return cls

    return deco


def get_reader(name: str, **kwargs) -> BaseReader:
    if name not in _READERS:
        raise KeyError(f"未知读取器 {name}，已注册：{list(_READERS)}")
    return _READERS[name](**kwargs)


def available_readers() -> List[str]:
    return sorted(_READERS)
