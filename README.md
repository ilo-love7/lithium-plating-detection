# 锂电池析锂（Lithium Plating）无损检测方法系统验证

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![CI](https://github.com/ilo-love7/lithium-plating-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/ilo-love7/lithium-plating-detection/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-72%25-yellow)](https://github.com/ilo-love7/lithium-plating-detection/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![No Deep Learning](https://img.shields.io/badge/deep%20learning-none-brightgreen)]()

*Lithium Plating Non-Destructive Detection — classical signal processing & physics-based statistics, no deep learning, no model training.*

纯经典信号处理与物理可解释统计的析锂无损检测工具链，**零深度学习、零模型训练**。
实现并系统验证四类公开文献中常见的无损检测方法，在合成真值数据与多个国际公开
真实数据集上进行交叉验证，并明确各方法的适用边界与可信度。

## 核心结论

- 合成真值 16 电芯（8 析锂 / 8 对照）：四方法灵敏度/特异度均为 1。
- 真实 16 电芯交叉验证：M1/M3/M4 全阴性；M2 仅在 Oxford 电芯深度老化后报峰移，
  结合 40 ℃ 高温抑制析锂的物理预期，该峰移更可能是正常老化而非析锂。
- 证据等级：M4 高、M1/M3 中高、M2 低（峰移非析锂特异且对阈值敏感）。

## 方法一览

| 方法 | 信号 | 核心算法 | 阳性判据 |
| --- | --- | --- | --- |
| M1 | 充电后弛豫电压 | SG 平滑 + 保序回归单调基线 + 残差肩部 | 连续 rest 残差肩部超阈值 |
| M2 | 低倍率 dQ/dV | 统一电压网格重采样 + 主峰提取 + 顺序匹配 | 峰位持续偏移/分裂 |
| M3 | 库仑效率 | 滚动 CE + ruptures（PELT）变点 | 持续 CE 下降（变点仅辅助） |
| M4 | 充电电压曲线 | CC 段 dV/dQ 跨循环基准对照 | 高 SOC 新出现负向凹陷 |

## 目录结构

```
.
├── cli.py                 # 命令行入口
├── requirements.txt
├── configs/config.yaml    # 全部阈值集中管理，禁止硬编码
├── src/plating/
│   ├── schema.py          # Segment/Cycle/CellData 统一结构
│   ├── cleaning.py        # 数据清洗
│   ├── relaxation.py      # 弛豫段检测
│   ├── synthetic.py       # 含析锂真值的合成电池
│   ├── analysis.py        # 交叉验证、一致性、工况对照
│   ├── readers/           # oxford/nasa/arbin/generic 读取器
│   └── methods/           # m1~m4
├── scripts/               # 诊断、测试、跑批、绘图
└── results/               # CSV、日志、figures
```

## 安装

```bash
# 方式一：作为包安装（提供 console 命令 plating）
python -m pip install .

# 方式二：开发模式（含测试/ruff 等开发依赖）
python -m pip install -e ".[dev]"

# 方式三：仅安装运行依赖（历史用法）
python -m pip install -r requirements.txt
```

安装后可直接使用 console 入口 `plating`（等价于 `python cli.py`），例如
`plating list-methods`、`plating synthetic --cells 16 --cycles 150`。

## 快速开始

```bash
# 列出方法
python cli.py list-methods

# 对数据运行检测（自动识别来源）
python cli.py detect --auto --path "<数据文件路径>"

# 指定方法与电芯，输出 JSON
python cli.py detect --source oxford --path "<.mat路径>" --method M2 --json

# 合成真值验证
python cli.py synthetic --cells 16 --cycles 150
```

## 复现实验

```bash
# 合成数据真值验证（16 电芯，四方法灵敏度/特异度）
python scripts/test_synthetic.py 16 150

# 真实数据多数据集交叉验证（16 电芯）
python scripts/run_cross_validation.py 5

# 阈值敏感性扫描
python scripts/threshold_sweep.py

# 合成闭环 ROC/AUC（fig5）
python scripts/run_roc_analysis.py

# 以 SOH 退化对齐的报警时机（fig6）
python scripts/run_degradation_alignment.py

# 强析锂工况（低温/高倍率，fig7）
python scripts/run_stress_scenarios.py

# 生成报告配图
python scripts/make_figures.py
```

## 方法学原则

1. **不做静态阈值调参**：所有判据基于跨循环/趋势的物理可解释对照，阈值集中在
   `configs/config.yaml`，并提供阈值敏感性扫描。
2. **无拆解真值不宣称准确率**：公开真实数据无析锂 ground truth，结论仅表述为
   "多方法一致性证据 + 工况物理预期对照"，不报告检测准确率。
3. **诚实标注局限**：M2 峰移非析锂特异（正常老化同样产生峰移），其阳性对阈值
   敏感；M3 周期性可恢复 CE 波动不应判析锂；M4 正常石墨 staging 须靠基准对照排除。
   强析锂工况验证进一步界定：M4 基于 dV/dQ 相对形态，在 3C 高倍率/极端低温下
   欧姆-极化压降主导、真实 kink 相对 prominence 低于阈值，灵敏度为 0，属方法适用
   边界而非缺陷；M1/M2/M3 在全部四工况下灵敏度/特异度均为 1，体现多方法互补价值。

## 数据获取

真实数据集不随仓库分发。请按 [`data/raw/README.md`](data/raw/README.md) 下载
Oxford / NASA / MATR-10 数据集并放入对应目录，或设置环境变量指向本地数据：

```bash
export PLATING_DATA_ROOT=/path/to/datasets   # Linux/macOS
$env:PLATING_DATA_ROOT="C:\path\to\datasets" # Windows PowerShell
```

合成验证（`python cli.py synthetic`）不依赖外部数据，可直接运行。

## 报告

完整方法卡、交叉验证结论与复现指南见
[`reports/析锂无损检测方法系统验证报告-v1.0.md`](reports/析锂无损检测方法系统验证报告-v1.0.md)。

## License

[MIT](LICENSE)
