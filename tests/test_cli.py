# -*- coding: utf-8 -*-
"""CLI 入口单元测试（subprocess）。"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(ROOT, "cli.py")


def _run(*args):
    return subprocess.run(
        [sys.executable, CLI, *args],
        capture_output=True, text=True, cwd=ROOT, timeout=180,
    )


def test_cli_list_methods():
    proc = _run("list-methods")
    assert proc.returncode == 0
    for m in ("M1", "M2", "M3", "M4"):
        assert m in proc.stdout


def test_cli_synthetic_small():
    proc = _run("synthetic", "--cells", "4", "--cycles", "40")
    assert proc.returncode == 0
    # 输出应包含方法名与结果
    assert "M1" in proc.stdout


def test_cli_help():
    proc = _run("--help")
    assert proc.returncode == 0
    assert "usage" in proc.stdout.lower()


def test_cli_bad_subcommand():
    proc = _run("not-a-command")
    assert proc.returncode != 0
