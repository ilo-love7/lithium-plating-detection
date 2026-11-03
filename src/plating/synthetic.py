# -*- coding: utf-8 -*-
"""合成电池数据生成器（含析锂真值）。

基于简化电化学：OCV(SOC) + 内阻 + RC 极化 + 弛豫，可在指定循环注入析锂，并
让四类可观测特征随析锂进度自然出现：
- M1 弛豫肩部：在 -dV/dt 上制造局部谷（沉积锂重新嵌回的混合电位）；
- M2 低倍率特征化 dQ/dV 峰移；
- M3 库仑效率下降（充电电荷多于放电电荷，差值为沉积量）；
- M4 充电 kink：高 SOC 宽平台扭曲。
用于在无拆解真值时量化各方法的灵敏度/特异度，并支撑全流程冒烟。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .config import Config
from .schema import CellData, Cycle, Segment


def _ocv_curve(soc: np.ndarray) -> np.ndarray:
    """典型石墨/正极 OCV：基准线 + 若干 staging 平滑台阶。"""
    v = 3.0 + 0.9 * soc
    for x0, w, a in [(0.12, 0.03, 0.06), (0.28, 0.04, 0.05),
                      (0.52, 0.05, 0.07), (0.78, 0.04, 0.05)]:
        v += a * np.tanh((soc - x0) / w)
    return v


@dataclass
class _PlatingSpec:
    enabled: bool
    onset: int
    fraction: float = 0.10        # 析锂后用于沉积的电流比例


class SyntheticBattery:
    def __init__(self, cfg: Config, cell_id: str, plating: _PlatingSpec,
                 nominal_ah: float = 1.0, ambient_t: float = 25.0,
                 charge_c: float = 1.0):
        self.cfg = cfg
        self.cell_id = cell_id
        self.plating = plating
        self.nominal_ah = nominal_ah
        self.ambient_t = ambient_t
        self.charge_c = charge_c
        self.dt = cfg.get("synthetic", "sample_interval_s", default=10)
        self.noise = cfg.get("synthetic", "noise_std_v", default=0.002)
        self.r0 = 0.03
        self.rp = 0.02
        seed = int(hashlib.md5(cell_id.encode("utf-8")).hexdigest()[:8], 16)
        self.rng = np.random.default_rng(seed)

    def _progress(self, cycle: int) -> float:
        if not self.plating.enabled or cycle < self.plating.onset:
            return 0.0
        return float(np.clip((cycle - self.plating.onset) / 20.0, 0, 1))

    def build(self, n_cycles: int, relax_s: int) -> CellData:
        cell = CellData(
            cell_id=self.cell_id, source="synthetic",
            nominal_capacity_ah=self.nominal_ah,
            meta={"plating": self.plating.enabled,
                  "plating_onset": self.plating.onset},
        )
        diag_every = self.cfg.get("synthetic", "diagnostic_every", default=10)
        capacity = self.nominal_ah
        for c in range(n_cycles):
            prog = self._progress(c)
            capacity *= 1 - (0.0008 + self.plating.fraction * 0.012 * prog)
            cyc = Cycle(cycle_index=c, ambient_temperature_c=self.ambient_t,
                        rated_capacity_ah=capacity)
            if c % diag_every == 0:
                cyc.add(self._low_rate_charge(c, capacity, prog))
            else:
                cyc.add(self._charge(c, capacity, prog))
            cyc.add(self._relax(c, relax_s, prog))
            cyc.add(self._discharge(c, capacity))
            cell.add(cyc)
        return cell

    def _noise(self, v: np.ndarray) -> np.ndarray:
        return v + self.rng.normal(0, self.noise, len(v))

    def _charge(self, c: int, cap: float, prog: float) -> Segment:
        i_charge = self.nominal_ah * self.charge_c
        frac = self.plating.fraction * prog
        # 析锂时部分电流被沉积分流，充满活性材料需更多电荷
        total_ah = cap / max(1 - frac, 1e-6)
        n = int((total_ah / i_charge * 3600) / self.dt)
        t = np.arange(n) * self.dt
        active_ah = np.linspace(0, cap, n)
        soc = active_ah / cap
        v = _ocv_curve(soc) + i_charge * (self.r0 + self.rp)
        # M4 kink：宽平台扭曲，幅度随析锂累积
        k_soc = self.cfg.get("synthetic", "kink_soc", default=0.72)
        k_w = self.cfg.get("synthetic", "kink_width", default=0.08)
        k_amp = self.cfg.get("synthetic", "kink_amp_v", default=0.05)
        v -= k_amp * prog * np.exp(-((soc - k_soc) / k_w) ** 2)
        return Segment("charge", t, np.full(n, i_charge),
                       self._noise(v), np.full(n, self.ambient_t), "charge")

    def _low_rate_charge(self, c: int, cap: float, prog: float) -> Segment:
        rate_c = self.cfg.get("synthetic", "low_rate_c", default=0.05)
        i_charge = rate_c * self.nominal_ah
        dt = 60.0  # 低倍率段用粗采样控制点数
        n = int((cap / i_charge * 3600) / dt)
        t = np.arange(n) * dt
        soc = np.linspace(0, 1, n)
        v = _ocv_curve(soc) + i_charge * (self.r0 + self.rp)
        # M2 峰移：在目标 staging 处用平滑阶跃使平台电压偏移
        m2_soc = self.cfg.get("synthetic", "m2_soc", default=0.35)
        m2_shift = self.cfg.get("synthetic", "m2_shift_v", default=0.045)
        v += 0.5 * m2_shift * prog * (1 + np.tanh((soc - m2_soc) / 0.02))
        return Segment("charge", t, np.full(n, i_charge),
                       self._noise(v), np.full(n, self.ambient_t), "charge")

    def _relax(self, c: int, relax_s: int, prog: float) -> Segment:
        rest_dt = self.cfg.get("synthetic", "rest_sample_interval_s",
                               default=self.dt)
        n = int(relax_s / rest_dt)
        t = np.arange(n) * rest_dt
        v0 = _ocv_curve(1.0)
        # 正常弛豫速率（-dV/dt），单调指数衰减
        r0_rate, tau = 1.2e-4, 800.0
        rate = r0_rate * np.exp(-t / tau)
        if prog > 0:
            # M1 肩部：在 t_s 处让速率出现局部谷
            depth = self.cfg.get("synthetic", "shoulder_depth", default=0.65)
            t_s = self.cfg.get("synthetic", "shoulder_time_s", default=600)
            w = self.cfg.get("synthetic", "shoulder_width_s", default=220)
            rate = rate * (1 - depth * prog * np.exp(-((t - t_s) / w) ** 2))
        # 积分得电压（减去累计弛豫量）
        drop = np.concatenate([[0], np.cumsum(0.5 * (rate[:-1] + rate[1:])
                                               * np.diff(t))])
        v = v0 - drop - self.rp
        return Segment("rest", t, np.zeros(n), self._noise(v),
                       np.full(n, self.ambient_t), "rest")

    def _discharge(self, c: int, cap: float) -> Segment:
        i_dis = self.nominal_ah * 1.0
        n = int((cap / i_dis * 3600) / self.dt)
        t = np.arange(n) * self.dt
        soc = np.linspace(1, 0, n)
        v = _ocv_curve(soc) - i_dis * (self.r0 + self.rp)
        return Segment("discharge", t, np.full(n, -i_dis),
                       self._noise(v), np.full(n, self.ambient_t), "discharge")


def generate_dataset(cfg: Config, n_cells: Optional[int] = None,
                     n_cycles: Optional[int] = None) -> List[CellData]:
    """生成合成数据集：约一半析锂、一半对照，真值写入 meta。"""
    n_cells = n_cells or cfg.get("synthetic", "n_cells", default=8)
    n_cycles = n_cycles or cfg.get("synthetic", "n_cycles", default=200)
    relax_s = cfg.get("synthetic", "relax_duration_s", default=3600)
    cells: List[CellData] = []
    for k in range(n_cells):
        plated = k >= n_cells // 2
        spec = _PlatingSpec(enabled=plated,
                            onset=cfg.get("synthetic", "plating_onset_cycle",
                                          default=80))
        cells.append(SyntheticBattery(cfg, f"Synth_{k:02d}", spec)
                     .build(n_cycles, relax_s))
    return cells


def generate_stress_dataset(cfg: Config, scenario_name: str,
                            n_cells: int = 8, n_cycles: int = 120) -> List[CellData]:
    """在指定强析锂工况（低温/高倍率）下生成一半析锂、一半对照的数据集。"""
    sc = cfg.get("stress_scenarios", scenario_name).as_dict()
    ambient_t = float(sc["ambient_temperature_c"])
    charge_c = float(sc["charge_c_rate"])
    onset = int(sc["onset_cycle"])
    fraction = float(sc["plating_fraction"])
    relax_s = cfg.get("synthetic", "relax_duration_s", default=3600)
    cells: List[CellData] = []
    for k in range(n_cells):
        plated = k >= n_cells // 2
        spec = _PlatingSpec(enabled=plated, onset=onset, fraction=fraction)
        cells.append(SyntheticBattery(
            cfg, f"{scenario_name}_{k:02d}", spec,
            ambient_t=ambient_t, charge_c=charge_c).build(n_cycles, relax_s))
    return cells


def ground_truth(cells: List[CellData]) -> Dict[str, Dict]:
    """返回每电芯析锂真值与起始循环。"""
    return {
        c.cell_id: {
            "plating": bool(c.meta.get("plating", False)),
            "onset": int(c.meta.get("plating_onset", -1)),
        }
        for c in cells
    }
