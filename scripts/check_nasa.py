# -*- coding: utf-8 -*-
import scipy.io as sio
import numpy as np

p = r"D:\项目\02_数据集\01_国际公开集\NASA_PCoE\1. BatteryAgingARC-FY08Q4\B0005.mat"
m = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
cy = m["B0005"].cycle
types = [c.type for c in cy]
print("counts", {t: types.count(t) for t in set(types)})
d0 = cy[0].data
d1 = cy[1].data
print("charge I mean", round(float(np.mean(d0.Current_measured)), 3))
print("discharge I mean", round(float(np.mean(d1.Current_measured)), 3))
print("charge time range", round(float(d0.Time[0]), 1), round(float(d0.Time[-1]), 1))
dis = [c for c in cy if c.type == "discharge"]
print("cap first5", [round(float(c.data.Capacity), 3) for c in dis[:5]])
print("ambient", set(float(c.ambient_temperature) for c in cy))
# 检查是否有近零电流（潜在静置）
for i in range(6):
    d = cy[i].data
    print(i, cy[i].type, "I abs min=", round(float(np.min(np.abs(d.Current_measured))), 3))
