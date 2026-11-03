# -*- coding: utf-8 -*-
"""配置加载：读取 config.yaml，支持路径覆盖与点号访问。"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional

import yaml

_DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs",
    "config.yaml",
)


class Config:
    """对 dict 的轻量包装，支持属性式访问与 .get 合并。"""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        val = self._data.get(key)
        return _wrap(val)

    def __getitem__(self, key: str) -> Any:
        return _wrap(self._data[key])

    def get(self, *keys, default: Any = None) -> Any:
        node: Any = self._data
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return _wrap(node)

    def as_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        return f"Config({list(self._data.keys())})"


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return Config(value)
    return value


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """加载 YAML 配置。overrides 用于 CLI 覆盖顶层键。"""
    path = path or _DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if overrides:
        _deep_update(data, overrides)
    return Config(data)


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> None:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
