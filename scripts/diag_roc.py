# -*- coding: utf-8 -*-
"""诊断 roc_curve：用已保存电芯分数复现，打印阈值与 TPR/FPR。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import numpy as np
import pandas as pd

from plating.roc import roc_curve, auc

df = pd.read_csv("results/roc_cell_scores.csv")
labels = df["truth"].to_numpy(dtype=bool)
for m in ["M1", "M2", "M3", "M4"]:
    scores = df[f"score_{m}"].to_numpy(dtype=float)
    fpr, tpr = roc_curve(scores, labels)
    print(f"===== {m} (dtype labels={labels.dtype}) =====")
    print("fpr:", np.round(fpr, 3))
    print("tpr:", np.round(tpr, 3))
    print("AUC=", auc(fpr, tpr))
