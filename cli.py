# -*- coding: utf-8 -*-
"""析锂无损检测工具链命令行入口。

用法示例：
  python cli.py list-methods
  python cli.py detect --source oxford --path "<mat路径>" --method M2
  python cli.py detect --auto --path "<数据路径>" --cell Oxford_Cell1
  python cli.py detect --source arbin --path "<ResVal文件>" --json
  python cli.py synthetic --cells 16 --cycles 150
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from plating.config import load_config
from plating.readers import load, auto_load
from plating.methods import run_method, run_all, METHOD_NAMES


def _result_payload(r):
    return {
        "method": r.method, "cell_id": r.cell_id, "cycle": r.cycle,
        "positive": bool(r.positive), "score": round(float(r.score), 6),
        "threshold": round(float(r.threshold), 6),
        "detail": {k: v for k, v in r.detail.items()
                   if isinstance(v, (int, float, bool, str)) or v is None},
    }


def cmd_detect(args):
    cfg = load_config(args.config)
    if args.auto:
        cells = auto_load(args.path)
    else:
        if not args.source:
            raise SystemExit("需指定 --source 或使用 --auto")
        cells = load(args.source, args.path)
    if args.cell:
        cells = [c for c in cells if c.cell_id == args.cell]
        if not cells:
            raise SystemExit(f"未找到电芯 {args.cell}")

    methods = [args.method] if args.method and args.method != "ALL" else METHOD_NAMES
    out = []
    for cell in cells:
        if len(methods) == len(METHOD_NAMES) and args.method == "ALL":
            res = run_all(cell, cfg)
            for m in METHOD_NAMES:
                out.extend(_result_payload(r) for r in res[m])
        else:
            for m in methods:
                out.extend(_result_payload(r) for r in run_method(m, cell, cfg))

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return
    # 人类可读输出
    for p in out:
        flag = "POSITIVE" if p["positive"] else "negative"
        print(f"[{p['method']}] {p['cell_id']} cycle={p['cycle']} "
              f"{flag} score={p['score']} threshold={p['threshold']}")
    n_pos = sum(1 for p in out if p["positive"])
    print(f"\n共 {len(out)} 条判定，阳性 {n_pos} 条")


def cmd_list_methods(args):
    desc = {
        "M1": "电压弛豫肩部（保序回归残差）",
        "M2": "dQ/dV 峰移/分裂（统一网格顺序匹配）",
        "M3": "库仑效率持续下降（ruptures 变点辅助）",
        "M4": "充电 kink（跨循环基准形态对照）",
    }
    for m in METHOD_NAMES:
        print(f"{m}: {desc[m]}")


def cmd_synthetic(args):
    from plating.synthetic import generate_dataset, ground_truth
    cfg = load_config(args.config)
    cells = generate_dataset(cfg, args.cells, args.cycles)
    gt = ground_truth(cells)
    stats = {m: {"TP": 0, "FN": 0, "FP": 0, "TN": 0} for m in METHOD_NAMES}
    for cell in cells:
        truth = gt[cell.cell_id]["plating"]
        res = run_all(cell, cfg)
        for m in METHOD_NAMES:
            pred = any(r.positive for r in res[m])
            key = ("TP" if pred else "FN") if truth else ("FP" if pred else "TN")
            stats[m][key] += 1
    for m in METHOD_NAMES:
        s = stats[m]
        sens = s["TP"] / max(s["TP"] + s["FN"], 1)
        spec = s["TN"] / max(s["TN"] + s["FP"], 1)
        print(f"{m}: 灵敏度={sens:.2f} 特异度={spec:.2f} {s}")


def build_parser():
    p = argparse.ArgumentParser(description="锂电池析锂无损检测工具链")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("detect", help="对数据运行检测方法")
    d.add_argument("--path", required=True, help="数据文件路径")
    d.add_argument("--source", help="来源：oxford/nasa/arbin/generic")
    d.add_argument("--auto", action="store_true", help="按路径自动识别来源")
    d.add_argument("--method", default="ALL", help="M1/M2/M3/M4，默认 ALL")
    d.add_argument("--cell", help="仅分析指定 cell_id")
    d.add_argument("--config", help="配置文件路径")
    d.add_argument("--json", action="store_true", help="输出 JSON")
    d.set_defaults(func=cmd_detect)

    lm = sub.add_parser("list-methods", help="列出检测方法")
    lm.add_argument("--config")
    lm.set_defaults(func=cmd_list_methods)

    sy = sub.add_parser("synthetic", help="合成数据真值验证")
    sy.add_argument("--cells", type=int, default=16)
    sy.add_argument("--cycles", type=int, default=150)
    sy.add_argument("--config")
    sy.set_defaults(func=cmd_synthetic)
    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
