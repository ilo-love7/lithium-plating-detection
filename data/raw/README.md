# 数据集目录（不随仓库分发）

本目录用于存放真实电池公开数据集。**数据集体积较大且受各自许可约束，不随
本仓库分发**，请按下面结构自行下载后放入，或通过环境变量 `PLATING_DATA_ROOT`
指向你本地已有的数据根目录。

## 期望目录结构

```
data/raw/
├── Oxford_OBD1/
│   └── Oxford_Battery_Degradation_Dataset_1.mat
├── NASA_PCoE/
│   └── 1. BatteryAgingARC-FY08Q4/
│       ├── B0005.mat
│       ├── B0006.mat
│       └── B0007.mat
└── MATR-10_DiagnosticFree/
    └── ResValData/
        └── ResVal_*.001 ...
```

## 数据来源

- **Oxford OBD1**：Oxford Battery Degradation Dataset 1（8 电芯、0.74 Ah、40 ℃）。
- **NASA PCoE**：Battery Aging ARC-FY08Q4（B0005–B0007，2 Ah，CC-CV）。
- **MATR-10**：Diagnostic-Free ResVal Data（4.84 Ah，含真实静置段）。

## 使用环境变量替代

```powershell
# Windows PowerShell
$env:PLATING_DATA_ROOT = "C:\path\to\your\datasets"

# Linux / macOS
export PLATING_DATA_ROOT=/path/to/your/datasets
```

设置后，`scripts/run_cross_validation.py` 等工具会自动从该目录读取。
合成数据验证（`python cli.py synthetic`）不依赖任何外部数据，可直接运行。
