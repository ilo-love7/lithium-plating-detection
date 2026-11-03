# -*- coding: utf-8 -*-
"""按来源段统计 relaxation 事件，排查 20 vs 15。"""
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from plating.config import load_config
from plating.readers import load
from plating.cleaning import clean_cell
from plating.relaxation import detect_cell

f = r"D:\项目\02_数据集\01_国际公开集\MATR-10_DiagnosticFree\ResValData\ResVal_000013_000185.001"
cfg = load_config()
cell = load("arbin", f)[0]
clean, _ = clean_cell(cell, cfg)
events = detect_cell(clean, cfg)
print("总事件:", len(events))
print("按来源段:", Counter(e.source_segment for e in events))
print("按类型标记:", Counter("cv" if e.is_cv_suspect else "normal" for e in events))
# 列出非 rest 来源的事件
for e in events:
    if "rest" not in e.source_segment:
        print("  非rest事件:", e.as_row())

# 检查 charge 段电流最小值
print("\ncharge 段电流范围:")
for s in clean.all_segments("charge")[:4]:
    print("  ", s.name, "I min/max", round(float(s.current.min()), 4), round(float(s.current.max()), 4),
          "V std", round(float(s.voltage.std()), 5))
