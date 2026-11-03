# -*- coding: utf-8 -*-
"""统一数据结构：Segment / Cycle / CellData。

统一 schema 五列：time, current, voltage, temperature, cycle。
电流符号约定：充电为正、放电为负、静置为 0。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

SEGMENT_TYPES = ("charge", "discharge", "rest", "unknown")


@dataclass
class Segment:
    """一段连续测量。"""

    segment_type: str
    time: np.ndarray            # 秒
    current: np.ndarray         # A（充电正）
    voltage: np.ndarray         # V
    temperature: np.ndarray     # ℃
    name: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.time = np.asarray(self.time, dtype=float)
        self.current = np.asarray(self.current, dtype=float)
        self.voltage = np.asarray(self.voltage, dtype=float)
        self.temperature = np.asarray(self.temperature, dtype=float)
        if self.segment_type not in SEGMENT_TYPES:
            self.segment_type = "unknown"

    def __len__(self) -> int:
        return len(self.time)

    def to_frame(self, cycle: Optional[int] = None) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "time": self.time,
                "current": self.current,
                "voltage": self.voltage,
                "temperature": self.temperature,
            }
        )
        df["cycle"] = -1 if cycle is None else cycle
        df["segment"] = self.segment_type
        return df

    @property
    def duration_s(self) -> float:
        return float(self.time[-1] - self.time[0]) if len(self.time) >= 2 else 0.0

    @property
    def mean_current_a(self) -> float:
        return float(np.mean(self.current)) if len(self.current) else 0.0


@dataclass
class Cycle:
    """一个完整循环，由若干 Segment 组成。"""

    cycle_index: int
    segments: List[Segment] = field(default_factory=list)
    ambient_temperature_c: Optional[float] = None
    rated_capacity_ah: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def add(self, seg: Segment) -> None:
        self.segments.append(seg)

    def by_type(self, segment_type: str) -> List[Segment]:
        return [s for s in self.segments if s.segment_type == segment_type]

    def to_frame(self) -> pd.DataFrame:
        frames = [s.to_frame(self.cycle_index) for s in self.segments]
        if not frames:
            return pd.DataFrame(columns=["time", "current", "voltage", "temperature", "cycle", "segment"])
        return pd.concat(frames, ignore_index=True)


@dataclass
class CellData:
    """一只电芯的全部数据。"""

    cell_id: str
    source: str
    nominal_capacity_ah: float
    cycles: List[Cycle] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def add(self, cyc: Cycle) -> None:
        self.cycles.append(cyc)

    def all_segments(self, segment_type: Optional[str] = None) -> List[Segment]:
        out: List[Segment] = []
        for c in self.cycles:
            out.extend(c.segments if segment_type is None else c.by_type(segment_type))
        return out

    def to_frame(self) -> pd.DataFrame:
        frames = [c.to_frame() for c in self.cycles]
        if not frames:
            return pd.DataFrame(columns=["time", "current", "voltage", "temperature", "cycle", "segment"])
        df = pd.concat(frames, ignore_index=True)
        df["cell_id"] = self.cell_id
        df["source"] = self.source
        return df

    def summary(self) -> Dict[str, Any]:
        n_charge = sum(len(c.by_type("charge")) for c in self.cycles)
        n_rest = sum(len(c.by_type("rest")) for c in self.cycles)
        return {
            "cell_id": self.cell_id,
            "source": self.source,
            "n_cycles": len(self.cycles),
            "n_charge_segments": n_charge,
            "n_rest_segments": n_rest,
            "nominal_capacity_ah": self.nominal_capacity_ah,
        }
