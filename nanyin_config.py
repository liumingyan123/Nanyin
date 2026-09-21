# -*- coding: utf-8 -*-
"""
nanyin_config.py — 统一 YAML 配置加载器
=======================================

背景：
  此前 vocab / 量化 / 采样等影响结果的参数硬编码散落在各 py 文件中，
  不利于复现与对照实验。自 E1 阶段起，所有影响结果的参数统一写入
  ``configs/e1_compound_events.yaml``，各模块通过本加载器读取同一份配置。

配置分层（E2 起）：
  - 基础层：``configs/e1_compound_events.yaml``（BASE_CONFIG_PATH）
  - 覆盖层：后续阶段自己的 yaml（如 ``configs/e2_compound_window.yaml``），
    只写本阶段新增/改动的字段，未声明的字段一律回退基础层（深合并）。
    ⇒ 目的：codec / model 等跨阶段共用的参数只有一份真源，避免双份维护漂移。

用法：
    from nanyin_config import load_config
    cfg = load_config()
    unit = cfg["codec"]["time_unit_sec"]

    # 读取某阶段的覆盖层配置（未声明字段自动回退基础层）
    cfg2 = load_config("configs/e2_compound_window.yaml")
"""

import copy
import os
import threading
from typing import Any, Dict, Optional

import yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG_PATH: str = os.path.join(_SCRIPT_DIR, "configs", "e1_compound_events.yaml")
# 基础层：后续阶段 yaml 未声明字段的回退来源
BASE_CONFIG_PATH: str = DEFAULT_CONFIG_PATH
# E2 覆盖层（复合事件滑窗 + 窗口训练）
E2_CONFIG_PATH: str = os.path.join(_SCRIPT_DIR, "configs", "e2_compound_window.yaml")

_CACHE: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并：override 覆盖 base，未声明字段保留 base 的值（仅 dict 递归，其余整体覆盖）。"""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(cfg_path: str) -> Dict[str, Any]:
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"[nanyin_config] 配置文件不存在: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"[nanyin_config] 配置内容不是字典: {cfg_path}")
    return data


def load_config(path: Optional[str] = None, reload: bool = False) -> Dict[str, Any]:
    """加载 YAML 配置（支持覆盖层：未声明字段回退基础层）。

    参数:
        path:   配置文件路径；默认 configs/e1_compound_events.yaml（基础层）。
                传入其它路径（如 configs/e2_compound_window.yaml）时，
                先读基础层再与本文件做深合并，本文件字段优先。
        reload: True 时忽略缓存重新读取
    返回:
        配置字典（深拷贝，调用方修改不会影响缓存）
    异常:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError:    YAML 解析失败
    """
    cfg_path = os.path.abspath(path or DEFAULT_CONFIG_PATH)
    base_path = os.path.abspath(BASE_CONFIG_PATH)
    with _LOCK:
        if reload or cfg_path not in _CACHE:
            data = _read_yaml(cfg_path)
            if cfg_path != base_path:
                base = _CACHE.get(base_path)
                if base is None:
                    base = _read_yaml(base_path)
                    _CACHE[base_path] = base
                data = _deep_merge(base, data)
            _CACHE[cfg_path] = data
        return copy.deepcopy(_CACHE[cfg_path])


def get_section(name: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """读取配置的指定小节（如 "codec" / "model" / "generation" / "selftest"）。"""
    cfg = cfg if cfg is not None else load_config()
    if name not in cfg:
        raise KeyError(f"[nanyin_config] 配置缺少小节: {name}")
    return cfg[name]


if __name__ == "__main__":
    _cfg = load_config()
    print(f"配置文件: {DEFAULT_CONFIG_PATH}")
    for k, v in _cfg.items():
        print(f"  [{k}]")
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, list) and len(vv) > 4:
                    print(f"    {kk}: <list len={len(vv)}>")
                else:
                    print(f"    {kk}: {vv}")
        else:
            print(f"    {v}")
