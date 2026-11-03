# -*- coding: utf-8 -*-
"""配置加载与 Config 包装测试。"""
from plating.config import Config, load_config


def test_load_config_returns_config():
    cfg = load_config()
    assert isinstance(cfg, Config)
    # 顶层应包含四个方法的配置键
    keys = cfg.as_dict().keys()
    assert {"m1_relaxation", "m2_dqdv", "m3_coulomb", "m4_kink"}.issubset(keys)


def test_config_attribute_and_get():
    cfg = Config({"a": 1, "b": {"c": 2}})
    assert cfg.a == 1
    assert cfg.b.c == 2
    assert cfg.get("b", "c") == 2
    assert cfg.get("x", default=99) == 99


def test_config_as_dict_deepcopy():
    cfg = Config({"a": {"k": 1}})
    d = cfg.as_dict()
    d["a"]["k"] = 999
    assert cfg.a.k == 1          # 修改副本不影响原配置
