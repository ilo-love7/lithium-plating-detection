# -*- coding: utf-8 -*-
"""CLI 入口测试：直接 import 调用（含 main 调度）+ 薄包装 subprocess。"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import plating.cli as cli
from plating.cli import build_parser, cmd_list_methods, cmd_synthetic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_CLI = os.path.join(ROOT, "cli.py")


def _run(*args):
    return subprocess.run(
        [sys.executable, ROOT_CLI, *args],
        capture_output=True, text=True, cwd=ROOT, timeout=180)


def test_build_parser_and_main_dispatch(monkeypatch):
    parser = build_parser()
    assert parser.parse_args(["list-methods"]).command == "list-methods"

    # main 直接调度（monkeypatch 隔离真实检测）
    hit = {}
    monkeypatch.setattr(cli, "cmd_list_methods",
                        lambda a: hit.setdefault("called", True))
    monkeypatch.setattr(sys, "argv", ["plating", "list-methods"])
    cli.main()
    assert hit["called"]


def test_cmd_list_methods_direct(capsys):
    cmd_list_methods(argparse.Namespace())
    out = capsys.readouterr().out
    for m in ("M1", "M2", "M3", "M4"):
        assert m in out


def test_cmd_synthetic_direct(capsys):
    args = argparse.Namespace(cells=4, cycles=60, config=None)
    cmd_synthetic(args)
    out = capsys.readouterr().out
    for m in ("M1", "M2", "M3", "M4"):
        assert m in out


def test_cli_list_methods_subprocess():
    proc = _run("list-methods")
    assert proc.returncode == 0
    for m in ("M1", "M2", "M3", "M4"):
        assert m in proc.stdout


def test_cli_help_subprocess():
    proc = _run("--help")
    assert proc.returncode == 0
    assert "usage" in proc.stdout.lower()


def test_cli_bad_subcommand():
    proc = _run("not-a-command")
    assert proc.returncode != 0
