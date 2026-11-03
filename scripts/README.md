# 脚本说明

## 核心复现脚本（推荐，路径自动适配）

这些脚本通过库内 `get_data_root()` 读取数据，支持环境变量
`PLATING_DATA_ROOT`，无需修改代码：

| 脚本 | 作用 |
| --- | --- |
| `test_synthetic.py` | 合成真值数据验证（灵敏度/特异度），不依赖外部数据 |
| `run_cross_validation.py` | 真实数据多数据集交叉验证 |
| `threshold_sweep.py` | 阈值敏感性扫描 |
| `make_figures.py` | 生成报告配图 |

## 开发诊断脚本（diag_* / check_* / debug_* / probe_*）

其余 `diag_`、`check_`、`debug_`、`probe_`、`test_data_layer`、`test_methods`
等脚本是开发过程中用于定位问题的一次性诊断记录，保留以体现调试过程。
其中部分脚本写有作者本机的绝对路径，**在其他机器上运行前需将路径改为
本地数据位置**（或设置 `PLATING_DATA_ROOT` 后参考其逻辑改写）。它们不是
发布工具链的必需部分，核心功能请使用项目根目录的 `cli.py` 与上表脚本。
