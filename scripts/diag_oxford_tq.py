# -*- coding: utf-8 -*-
"""诊断 Oxford 原始 t/q 单位。"""
import os
import sys

import numpy as np
import scipy.io as sio

path = r"D:\项目\02_数据集\01_国际公开集\Oxford_OBD1\Oxford_Battery_Degradation_Dataset_1.mat"
mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
node = mat["Cell1"].cyc0000.C1ch
t = np.asarray(node.t, float)
q = np.asarray(node.q, float)
print("t 前10:", t[:10])
print("q 前10:", q[:10])
print("t 差差分位:", np.diff(t)[:5], "量级", np.median(np.diff(t)))
print("q 差分:", np.diff(q)[:5])
print("t 最大:", t[-1], " q 最大:", q[-1])
# 若 t 单位是小时
dqdh = np.gradient(q, t)
print("gradient(q,t) 中位:", np.median(dqdh))
