# -*- coding: utf-8 -*-
"""方法共享：统一结果结构、Savitzky-Golay 微分工具。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from scipy.signal import savgol_filter


@dataclass
class MethodResult:
    """单个方法在单个循环/段上的判定结果。"""

    method: str
    cell_id: str
    cycle: int
    positive: bool
    score: float
    threshold: float
    feature_time_s: Optional[float] = None
    detail: Dict[str, Any] = field(default_factory=dict)
    # 绘图用曲线（不写入结果 CSV）
    series: Dict[str, np.ndarray] = field(default_factory=dict)

    def as_row(self) -> Dict[str, Any]:
        row = {
            "method": self.method,
            "cell_id": self.cell_id,
            "cycle": self.cycle,
            "positive": bool(self.positive),
            "score": round(float(self.score), 6),
            "threshold": round(float(self.threshold), 6),
            "feature_time_s": (
                round(float(self.feature_time_s), 2)
                if self.feature_time_s is not None
                else None
            ),
        }
        for k, v in self.detail.items():
            if isinstance(v, (int, float, bool, str)) or v is None:
                row[k] = v
        return row


def fill_nonfinite(x: np.ndarray) -> np.ndarray:
    """把 NaN/inf 替换为线性插值结果，端点用最近有限值填充。"""
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if finite.all():
        return x
    if not finite.any():
        return np.zeros_like(x)
    idx = np.arange(len(x))
    x = x.copy()
    x[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    return x


def sg_smooth(x: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay 平滑，自动保证窗口为奇数且不超过数据长度。"""
    x = fill_nonfinite(x)
    n = len(x)
    if n < 3:
        return x.astype(float)
    w = min(window if window % 2 == 1 else window + 1, n if n %2 == 1 else n - 1)
    w = max(w, polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3)
    if w >= n:
        w = n - 1 if (n - 1) % 2 == 1 else n - 2
    if w <= polyorder:
        return x.astype(float)
    return savgol_filter(x, w, polyorder)


def sg_derivative(y: np.ndarray, x: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """SG 平滑后对 x 求一阶导。"""
    ys = sg_smooth(y, window, polyorder)
    return np.gradient(ys, x)


def safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """带保护的除法，避免除零产生 inf。"""
    return np.asarray(a) / np.where(np.abs(b) < eps, np.sign(b) * eps + (b == 0) * eps, b)
