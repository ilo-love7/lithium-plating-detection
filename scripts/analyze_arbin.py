# -*- coding: utf-8 -*-
"""解析单个 Arbin 文件，统计 State、静置段时长、温度列、loop 结构。"""
import os
import sys

import numpy as np
import pandas as pd

f = r"D:\项目\02_数据集\01_国际公开集\MATR-10_DiagnosticFree\ResValData\ResVal_000013_000185.001"

# 第1行是日期信息，第2行是列名
df = pd.read_csv(f, sep="\t", skiprows=1, engine="python", on_bad_lines="skip")
print("行数", len(df))
print("列名:", list(df.columns))
# 清理列名空白
df.columns = [str(c).strip() for c in df.columns]

print("\nState 取值分布:")
print(df["State"].value_counts(dropna=False))

# 温度列排查：找含非零、量级像温度的 VAR 列
for c in df.columns:
    if c.startswith("VAR"):
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().sum() > 0 and vals.abs().max() > 0:
            print("  非零列", c, "min", vals.min(), "max", vals.max())

# 连续 State 段
st = df["State"].to_numpy()
sts = st.astype(str)
changes = np.where(sts[1:] != sts[:-1])[0]
bounds = np.concatenate([[0], changes + 1, [len(df)]])
print("\n连续段数:", len(bounds) - 1)
t = pd.to_numeric(df["Test (Sec)"], errors="coerce").to_numpy()
amps = pd.to_numeric(df["Amps"], errors="coerce").to_numpy()
rest_durations = []
for k in range(len(bounds) - 1):
    a, b = bounds[k], bounds[k + 1]
    dur = t[b - 1] - t[a]
    if str(st[a]) == "R":
        rest_durations.append(dur)
print("静置(R)段数:", len(rest_durations))
if rest_durations:
    rd = np.array(rest_durations)
    print("  静置时长 min/中位/max (s):", round(rd.min(), 1), round(np.median(rd), 1), round(rd.max(), 1))
    print("  >=1800s 的段数:", int(np.sum(rd >= 1800)))
    print("  >=600s 的段数:", int(np.sum(rd >= 600)))
# Cyc# / Step / Loop 范围
for c in ["Cyc#", "Step", "Loop1", "Loop2"]:
    if c in df.columns:
        num = pd.to_numeric(df[c], errors="coerce")
        print(c, "范围", num.min(), num.max())
