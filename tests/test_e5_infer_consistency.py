# -*- coding: utf-8 -*-
"""
tests/test_e5_infer_consistency.py — E5 推理一致性单元测试
=========================================================

E5 任务书要求（**必须先于批量生成跑通**）：
  T1 可复现：固定随机种子、权重、采样参数，两次推理输出序列逐位完全一致；
  T2 prompt 对齐：生成序列前 16 个事件与对应窗口 GT 前 16 事件完全匹配
                 （与 E2/E3/E4 口径统一；且评估只统计第 16 个之后的续写片段）；
  T3 续写长度：可正确输出 target_length = 32 / 64 / 128 / 256 的序列。

附带自检（保证 E5 硬性约束不被违反）：
  T0 配置约束：postprocess.* 全 false；sampling 固定 T=1.0 / top_p=0.95 / top_k=0；
               prompt_len=16（总长度口径）；启用模型同时具备 best-val 权重。
  T4 命名与参数：E5 文件名规范 {song_id}_w{w}_{model}_seed{s}_len{L}.mid；
              换随即种子会得到不同序列（证明可复现不是"退化成贪心"）。

运行：
  venv/Scripts/python.exe tests/test_e5_infer_consistency.py
"""

import os
import re
import sys
import json
from typing import Dict, List, Any, Optional

import numpy as np
import torch

# ---- 项目根加入 sys.path ----
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TEST_DIR)
for p in (_ROOT, _TEST_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from generate_music import (                                    # noqa: E402
    CODEC, build_drift_midi_name, resolve_e5_models, load_e5_model,
    load_val_windows, generate_compound_autoregressive, _rng_seed,
)
from nanyin_config import load_config                            # noqa: E402

E5_CONFIG_PATH = os.path.join(_ROOT, "configs", "e5_model_drift.yaml")
RESULT_JSON = os.path.join(_ROOT, "results", "e5", "e5_infer_consistency_test.json")

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_SEP = "=" * 76


class TestRunner:
    def __init__(self) -> None:
        self.results: List[Dict[str, Any]] = []
        self.detail: Dict[str, Any] = {}

    def check(self, tid: str, name: str, passed: bool, info: str = "") -> bool:
        self.results.append({"id": tid, "name": name, "passed": bool(passed), "info": info})
        print(f"  [{'PASS' if passed else 'FAIL'}] {tid} {name}" + (f" — {info}" if info else ""))
        return passed

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r["passed"])

    @property
    def n_total(self) -> int:
        return len(self.results)


def _gen_once(model, win, target_len: int, rng_seed: int,
              cfg: dict) -> Dict[str, List[int]]:
    """按 E5 口径生成一条序列（采样参数全部取自固定配置）。"""
    scfg = cfg["sampling"]
    wcfg = cfg["window"]
    gen = torch.Generator(device=_DEVICE.type)
    gen.manual_seed(int(rng_seed))
    return generate_compound_autoregressive(
        model=model,
        prompt_pitch_ids=win["pitch_ids"][: int(wcfg["prompt_len_events"])],
        prompt_duration_ids=win["duration_ids"][: int(wcfg["prompt_len_events"])],
        prompt_shift_ids=win["shift_ids"][: int(wcfg["prompt_len_events"])],
        target_len=int(target_len),
        device=_DEVICE,
        codec=CODEC,
        temperature=float(scfg["temperature"]),
        top_p=float(scfg["top_p"]),
        top_k=dict(scfg["top_k"]),
        generator=gen,
        prepend_bos=bool(wcfg.get("prepend_bos", True)),
    )


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    tr = TestRunner()
    cfg = load_config(E5_CONFIG_PATH)
    wcfg = cfg["window"]
    scfg = cfg["sampling"]
    prompt_len = int(wcfg["prompt_len_events"])
    lens = list(wcfg["target_len_events_list"])

    print(_SEP)
    print("E5 推理一致性单元测试（必须在批量生成之前全部 PASS）")
    print(_SEP)
    print(f"  设备     : {_DEVICE}")
    print(f"  配置     : {E5_CONFIG_PATH}")
    print(f"  prompt   : {prompt_len} 个 GT 事件（总长度口径）")
    print(f"  续写长度 : {lens}")
    print(f"  采样固定 : T={scfg['temperature']}, p={scfg['top_p']}, k={scfg['top_k']}")

    # ------------------------------------------------------------------
    # T0 配置与权重约束自检
    # ------------------------------------------------------------------
    print(f"\n{_SEP}\nT0  配置与权重约束自检\n{_SEP}")
    pp = dict(cfg["postprocess"])
    tr.check("T0-1", "postprocess 全部关闭（评估使用原生输出）",
             all(not bool(v) for v in pp.values()), str(pp))
    tr.check("T0-2", "采样固定 T=1.0 / top_p=0.95 / top_k=0",
             float(scfg["temperature"]) == 1.0 and float(scfg["top_p"]) == 0.95
             and all(int(v) == 0 for v in dict(scfg["top_k"]).values()),
             f"T={scfg['temperature']}, p={scfg['top_p']}, k={dict(scfg['top_k'])}")
    tr.check("T0-3", "prompt_len = 16，窗口数/长度沿用 E2/E3/E4",
             prompt_len == 16 and int(wcfg["window_size_events"]) == 256
             and int(wcfg["eval_stride_events"]) == 256, "256/256/16")

    models = resolve_e5_models(cfg)
    tr.check("T0-4", "参与对比的模型 = boyatcn + musicmamba",
             sorted(n for n, _ in models) == ["boyatcn", "musicmamba"],
             str([n for n, _ in models]))

    missing: List[str] = []
    for name, mcfg in models:
        for seed in list(mcfg["model_seeds"]):
            path = os.path.join(_ROOT, str(mcfg["checkpoint_template"]).format(seed=seed))
            if not os.path.exists(path):
                missing.append(path)
    tr.check("T0-5", "两组模型的 best-val 早停权重齐备（每组 3 籽）",
             not missing, f"缺失 {missing}" if missing else
             f"{sum(len(m['model_seeds']) for _, m in models)} 个权重文件")

    windows = load_val_windows(cfg, verbose=False)
    tr.check("T0-6", "验证窗口 = 10（dapu14 × 4 + dapu15 × 6），窗口长度 256",
             len(windows) == 10 and all(len(w["pitch_ids"]) == 256 for w in windows),
             f"n={len(windows)}")

    if missing:
        print("\n【结论】权重缺失，**禁止**启动批量生成。")
        return 1

    win0 = windows[0]
    names_str = ", ".join(n for n, _ in models)

    # ------------------------------------------------------------------
    # T1 可复现（两组模型各验一次）
    # ------------------------------------------------------------------
    print(f"\n{_SEP}\nT1  可复现性校验（{names_str}）\n{_SEP}")
    reproducible = True
    for name, mcfg in models:
        seed0 = int(list(mcfg["model_seeds"])[0])
        ckpt = os.path.join(_ROOT, str(mcfg["checkpoint_template"]).format(seed=seed0))
        model = load_e5_model(name, ckpt, _DEVICE, strict=bool(mcfg.get("strict_load", True)))
        rng = 4242
        a1 = _gen_once(model, win0, 64, rng, cfg)
        a2 = _gen_once(model, win0, 64, rng, cfg)
        same = (a1["pitch_ids"] == a2["pitch_ids"]
                and a1["duration_ids"] == a2["duration_ids"]
                and a1["shift_ids"] == a2["shift_ids"])
        tr.check(f"T1-{name}", f"{name}: 同 (权重, rng, 采样参数) 两次推理逐位一致",
                 bool(same), f"len={len(a1['pitch_ids'])}")

        b = _gen_once(model, win0, 64, rng + 1, cfg)
        diff = not (a1["pitch_ids"] == b["pitch_ids"]
                    and a1["duration_ids"] == b["duration_ids"]
                    and a1["shift_ids"] == b["shift_ids"])
        tr.check(f"T4-{name}", f"{name}: 换 rng 种子后序列不同（非贪心退化）", bool(diff), "")
        reproducible = reproducible and bool(same)
        del model
        if _DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    tr.detail["T1_all_reproducible"] = bool(reproducible)

    # ------------------------------------------------------------------
    # T2 prompt 对齐（前 16 事件 == GT 前 16 事件，三路逐位）
    # ------------------------------------------------------------------
    print(f"\n{_SEP}\nT2  prompt 对齐校验（{names_str}）\n{_SEP}")
    for name, mcfg in models:
        seed0 = int(list(mcfg["model_seeds"])[0])
        ckpt = os.path.join(_ROOT, str(mcfg["checkpoint_template"]).format(seed=seed0))
        model = load_e5_model(name, ckpt, _DEVICE, strict=bool(mcfg.get("strict_load", True)))
        for wi in (0, 5, 9):
            win = windows[wi]
            out = _gen_once(model, win, 64, 777 + wi, cfg)
            ok = (out["pitch_ids"][:prompt_len] == win["pitch_ids"][:prompt_len]
                  and out["duration_ids"][:prompt_len] == win["duration_ids"][:prompt_len]
                  and out["shift_ids"][:prompt_len] == win["shift_ids"][:prompt_len])
            tr.check(f"T2-{name}-w{wi}",
                     f"{name}: 窗口 {wi} 生成序列前 {prompt_len} 事件 == GT 前 {prompt_len} 事件",
                     bool(ok), f"{win['song_id']}_w{win['window_index']}")
        eval_region_ok = len(_gen_once(model, windows[0], 64, 31, cfg)["pitch_ids"]) - prompt_len == 48
        tr.check(f"T2-{name}-region", f"{name}: 评估区域 = 续写片段（64 − 16 = 48 事件）",
                 bool(eval_region_ok), "")
        del model
        if _DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # T3 续写长度可正确输出 32 / 64 / 128 / 256
    # ------------------------------------------------------------------
    print(f"\n{_SEP}\nT3  续写长度校验（{names_str}）\n{_SEP}")
    for name, mcfg in models:
        seed0 = int(list(mcfg["model_seeds"])[0])
        ckpt = os.path.join(_ROOT, str(mcfg["checkpoint_template"]).format(seed=seed0))
        model = load_e5_model(name, ckpt, _DEVICE, strict=bool(mcfg.get("strict_load", True)))
        for L in lens:
            out = _gen_once(model, win0, L, 20260 + int(L), cfg)
            ok = (len(out["pitch_ids"]) == L and len(out["duration_ids"]) == L
                  and len(out["shift_ids"]) == L)
            tr.check(f"T3-{name}-{L}", f"{name}: target_len={L} → 三路序列长度均为 {L}",
                     bool(ok), f"p={len(out['pitch_ids'])}, d={len(out['duration_ids'])}, "
                               f"s={len(out['shift_ids'])}")
        del model
        if _DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # T5 命名规范
    # ------------------------------------------------------------------
    print(f"\n{_SEP}\nT5  产物命名规范\n{_SEP}")
    n1 = build_drift_midi_name("dapu14kouhuangtian", 2, "boyatcn", 2026, 256)
    tr.check("T5-1", "文件名 = {song_id}_w{w}_{model}_seed{s}_len{L}.mid",
             bool(re.match(r"^dapu14kouhuangtian_w2_boyatcn_seed2026_len256\.mid$", n1)), n1)
    n2 = build_drift_midi_name("dapu15wujinjiao", 0, "musicmamba", 2028, 32)
    tr.check("T5-2", "不同模型/长度组合文件名互不重复",
             n2 == "dapu15wujinjiao_w0_musicmamba_seed2028_len32.mid", n2)

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    print(f"\n{_SEP}")
    print(f"单元测试汇总: {tr.n_passed}/{tr.n_total} PASS")
    print(_SEP)
    for r in tr.results:
        print(f"  {r['id']:<16}{'PASS' if r['passed'] else 'FAIL':<6}{r['name']}")

    os.makedirs(os.path.dirname(RESULT_JSON), exist_ok=True)
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "stage": "E5", "passed": tr.n_passed, "total": tr.n_total,
            "device": str(_DEVICE), "config": E5_CONFIG_PATH,
            "models": [n for n, _ in models], "lengths": lens,
            "results": tr.results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n判定明细已写入: {RESULT_JSON}")

    ok = tr.n_passed == tr.n_total
    print("【结论】全部 PASS，可启动 E5 批量生成" if ok
          else "【结论】存在 FAIL 项，**禁止**启动批量生成")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
