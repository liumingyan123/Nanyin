# -*- coding: utf-8 -*-
"""
tests/test_sampling_consistency.py — E4 采样一致性与 prompt 口径单元测试
========================================================================

E4 任务书要求（**必须先于批量生成跑通**）：
  A. 固定随机种子下，同一套权重 + 同一套采样参数 → 生成序列可复现；
  B. prompt 截取逻辑与 E2/E3/exp3 完全对齐（每窗口前 16 个 GT 事件 → 续写到 256 事件）；
  C. E4 硬性约束自检：禁止 MIDI 后处理 / 时值钳位修正（配置开关必须全关）。

测试项
------
  T0 配置约束        : postprocess.* 全 false；prompt/target 长度；prepend_bos；skip_prompt
  T1 生成可复现      : 同一 (权重, T, p, rng_seed) 跑两次 → 三路序列逐位相同
  T2 随机流确实生效  : 换 rng_seed → 序列应当不同（证明可复现不是因为"退化成贪心"）
  T3 采样参数生效    : T=0.6 与 T=1.2 在同一 rng 下应产出不同序列
  T4 prompt 对齐     : 生成序列前 16 个事件 == 该窗口 GT 前 16 个事件（三路逐位）；总长 = 256
  T5 窗口口径        : 验证窗口数 = 10（dapu14 4 + dapu15 6），窗口长度 = 256，非重叠且覆盖顺序正确
  T6 命名规范        : {song_id}_w{w}_seed{s}_T{t}_p{p}.mid 格式正确
  T7 46 维特征       : 维数 = 46、确定性、finite

运行：
  venv/Scripts/python.exe tests/test_sampling_consistency.py
"""

import os
import re
import sys
import json
from typing import Dict, List, Any

import numpy as np
import torch

# ---- 项目根加入 sys.path ----
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TEST_DIR)
for p in (_ROOT, _TEST_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from dataset_nanyin import VAL_SPLIT_IDS, CompoundEventWindowDataset  # noqa: E402
from generate_music import (                                          # noqa: E402
    CODEC, build_midi_name, _rng_seed, load_e4_model,
    load_val_windows, generate_compound_autoregressive,
)
from nanyin_config import load_config                                  # noqa: E402
from symbolic_features import FEATURE_DIM, extract_symbolic_features    # noqa: E402

E4_CONFIG_PATH = os.path.join(_ROOT, "configs", "e4_sampling_search.yaml")
RESULT_JSON = os.path.join(_ROOT, "results", "e4", "e4_sampling_consistency_test.json")

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_SEP = "=" * 76


def _fmt(v, width: int = 10) -> str:
    return f"{v:>{width}}"


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


def main() -> int:
    print(_SEP)
    print("E4 采样一致性 / prompt 口径单元测试")
    print(_SEP)
    print(f"  分支约束: experiment/E4_sampling_search（不训练，只加载 E3-L3 best-val 权重）")
    print(f"  设备    : {_DEVICE}")
    print(f"  配置    : {E4_CONFIG_PATH}")

    cfg = load_config(E4_CONFIG_PATH)
    wcfg, scfg, pcfg, ecfg, acfg = (
        cfg["window"], cfg["sampling"], cfg["postprocess"], cfg["evaluation"], cfg["weights"]
    )
    tr = TestRunner()

    # ------------------------------------------------------------------
    # T0 配置约束自检（E4 硬性要求）
    # ------------------------------------------------------------------
    print(f"\n{_SEP}\nT0  E4 配置约束自检\n{_SEP}")
    pp_off = all(v is False for v in pcfg.values())
    tr.check("T0-1", "postprocess 全部开关为 False（禁止 MIDI 后处理/钳位）", pp_off,
             f"postprocess={pcfg}")
    tr.check("T0-2", "prompt_len = 16 个 GT 事件", int(wcfg["prompt_len_events"]) == 16,
             f"prompt_len_events={wcfg['prompt_len_events']}")
    tr.check("T0-3", "target_len = 256 事件（含 prompt）", int(wcfg["target_len_events"]) == 256,
             f"target_len_events={wcfg['target_len_events']}")
    tr.check("T0-4", "prepend_bos = True（输入格式与训练样本一致）",
             bool(wcfg.get("prepend_bos", False)) is True, f"prepend_bos={wcfg.get('prepend_bos')}")
    tr.check("T0-5", "评估 skip_prompt = 16（评估区域 = 续写的 240 事件）",
             int(ecfg["skip_prompt"]) == 16, f"skip_prompt={ecfg['skip_prompt']}")
    tr.check("T0-6", "top-k 全部关闭（0），仅网格变量 T / top-p 变化",
             all(int(v) == 0 for v in scfg["top_k"].values()), f"top_k={scfg['top_k']}")
    tr.check("T0-7", "权重来源 = E3-L3 best-val checkpoint",
             str(acfg["source_stage"]).upper().replace("-", "_") == "E3_L3"
             and acfg["checkpoint_template"].endswith("best_valloss_model.pt"),
             f"source_stage={acfg['source_stage']}, tpl={acfg['checkpoint_template']}")

    # ------------------------------------------------------------------
    # T5 窗口口径（先做，后面测试依赖 windows）
    # ------------------------------------------------------------------
    print(f"\n{_SEP}\nT5  验证窗口口径（E2/E3 同源）\n{_SEP}")
    windows = load_val_windows(cfg, verbose=False)
    counts: Dict[str, int] = {}
    for w in windows:
        counts[w["song_id"]] = counts.get(w["song_id"], 0) + 1
    tr.check("T5-1", "验证窗口数 = 10", len(windows) == 10, f"实际 {len(windows)} 窗口，明细 {counts}")
    tr.check("T5-2", "每首曲目窗口数与 e3_window_stats 一致（dapu14=4, dapu15=6）",
             counts.get("dapu14kouhuangtian", 0) == 4 and counts.get("dapu15wujinjiao", 0) == 6,
             str(counts))
    all_len_256 = all(
        len(w["pitch_ids"]) == len(w["duration_ids"]) == len(w["shift_ids"]) == 256
        for w in windows
    )
    tr.check("T5-3", "每个窗口三路长度 = 256 事件", all_len_256, "")

    # 独立复核：CompoundEventWindowDataset 的窗口编号与自然序一致
    val_ds = CompoundEventWindowDataset(song_ids=VAL_SPLIT_IDS, split="val", cfg=cfg)
    idx_ok = True
    for i, w in enumerate(windows):
        s = val_ds[i]
        same_song = s["song_id"] == w["song_id"] and int(s["window_index"]) == w["window_index"]
        same_start = int(s["start"]) == w["start"]
        idx_ok = idx_ok and same_song and same_start
    tr.check("T5-4", "窗口编号 / 起始事件号与数据集 records 一致", idx_ok, "")

    win0 = windows[0]
    p_ds = val_ds[0]["target_pitch_ids"][:256].tolist()
    tr.check("T5-5", "窗口 GT pitch 序列与数据集 target_pitch_ids 逐位一致",
             p_ds == win0["pitch_ids"], "")

    # ------------------------------------------------------------------
    # T1/T2/T3/T4 —— 载入 E3-L3 权重真跑
    # ------------------------------------------------------------------
    print(f"\n{_SEP}\nT1~T4  固定权重下的生成可复现 / 随机流 / prompt 对齐\n{_SEP}")
    model_seed = int(acfg["model_seeds"][0])
    ckpt = acfg["checkpoint_template"].format(seed=model_seed)
    if os.path.exists(ckpt):
        model = load_e4_model(ckpt, _DEVICE, strict=bool(acfg.get("strict_load", True)))
    else:
        print(f"  ⚠ 权重缺失: {ckpt} —— 该组测试降级（随机初始化的同结构模型仅校验机制，不复现指标）")
        from models import NanyinBoYaTCN  # noqa: E402
        model = NanyinBoYaTCN().to(_DEVICE).eval()

    rng0 = _rng_seed(int(scfg["rng_base_seed"]), model_seed, 0, 0)
    top_k_map = dict(scfg["top_k"])
    prompt_len = int(wcfg["prompt_len_events"])
    target_len = int(wcfg["target_len_events"])

    def _gen(seed_val: int, temp: float, topp: float) -> Dict[str, List[int]]:
        gen = torch.Generator(device=_DEVICE.type)
        gen.manual_seed(seed_val)
        return generate_compound_autoregressive(
            model=model,
            prompt_pitch_ids=win0["pitch_ids"][:prompt_len],
            prompt_duration_ids=win0["duration_ids"][:prompt_len],
            prompt_shift_ids=win0["shift_ids"][:prompt_len],
            target_len=target_len, device=_DEVICE, codec=CODEC,
            temperature=temp, top_p=topp, top_k=top_k_map,
            generator=gen, prepend_bos=bool(wcfg.get("prepend_bos", True)),
        )

    a1 = _gen(rng0, 0.8, 0.9)
    a2 = _gen(rng0, 0.8, 0.9)
    same = (a1["pitch_ids"] == a2["pitch_ids"]
            and a1["duration_ids"] == a2["duration_ids"]
            and a1["shift_ids"] == a2["shift_ids"])
    tr.check("T1-1", "同权重 + 同采样参数 + 同 rng_seed → 生成序列逐位可复现", same,
             f"rng_seed={rng0}, T=0.8, p=0.9")

    b1 = _gen(rng0 + 1, 0.8, 0.9)
    diff_seed = (a1["pitch_ids"] != b1["pitch_ids"])
    tr.check("T2-1", "换 rng_seed → 序列不同（证明确实是采样而非退化贪心）", diff_seed, "")

    c1 = _gen(rng0, 1.2, 0.9)
    diff_temp = (a1["pitch_ids"] != c1["pitch_ids"])
    tr.check("T3-1", "换 temperature（0.8 → 1.2）同 rng → 序列不同（采样参数确实生效）",
             diff_temp, "")

    d1 = _gen(rng0, 0.8, 0.7)
    diff_topp = (a1["pitch_ids"] != d1["pitch_ids"])
    tr.check("T3-2", "换 top-p（0.9 → 0.7）同 rng → 序列不同（采样参数确实生效）", diff_topp, "")

    # T4 prompt 逐位对齐
    prompt_ok = (
        a1["pitch_ids"][:prompt_len] == win0["pitch_ids"][:prompt_len]
        and a1["duration_ids"][:prompt_len] == win0["duration_ids"][:prompt_len]
        and a1["shift_ids"][:prompt_len] == win0["shift_ids"][:prompt_len]
    )
    tr.check("T4-1", "生成序列前 16 个事件 = 该窗口 GT 前 16 个事件（三路逐位）", prompt_ok, "")
    len_ok = len(a1["pitch_ids"]) == len(a1["duration_ids"]) == len(a1["shift_ids"]) == target_len
    tr.check("T4-2", "生成总长度 = 256 事件（16 prompt + 240 续写）", len_ok, "")

    # ------------------------------------------------------------------
    # T6 命名规范
    # ------------------------------------------------------------------
    print(f"\n{_SEP}\nT6  产物命名规范\n{_SEP}")
    name = build_midi_name("dapu14kouhuangtian", 2, 2026, 1.0, 0.95)
    pattern = re.compile(r"^dapu14kouhuangtian_w2_seed2026_T1_p095\.mid$")
    tr.check("T6-1", "文件名 = {song_id}_w{w}_seed{s}_T{t}_p{p}.mid", bool(pattern.match(name)), name)
    name2 = build_midi_name("dapu15wujinjiao", 0, 2028, 0.6, 0.7)
    tr.check("T6-2", "不同参数组合文件名互不重复",
             name2 == "dapu15wujinjiao_w0_seed2028_T06_p07.mid", name2)

    # ------------------------------------------------------------------
    # T7 46 维符号特征
    # ------------------------------------------------------------------
    print(f"\n{_SEP}\nT7  46 维符号统计特征\n{_SEP}")
    feats = extract_symbolic_features(
        a1["pitch_ids"][prompt_len:],
        [CODEC.index_to_duration(i) for i in a1["duration_ids"][prompt_len:]],
        None, mode_pcs=[0, 2, 4, 7, 9],
    )
    tr.check("T7-1", "特征维数 = 46", feats.shape == (FEATURE_DIM,), f"shape={feats.shape}")
    tr.check("T7-2", "特征值全部有限（无 NaN / Inf）", bool(np.all(np.isfinite(feats))), "")
    feats2 = extract_symbolic_features(
        a1["pitch_ids"][prompt_len:],
        [CODEC.index_to_duration(i) for i in a1["duration_ids"][prompt_len:]],
        None, mode_pcs=[0, 2, 4, 7, 9],
    )
    tr.check("T7-3", "同一输入两次提取完全相同（确定性）",
             bool(np.allclose(feats, feats2)), "")

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    print(f"\n{_SEP}")
    print(f"单元测试汇总: {tr.n_passed}/{tr.n_total} PASS")
    print(_SEP)
    for r in tr.results:
        print(f"  {r['id']:<6}{'PASS' if r['passed'] else 'FAIL':<6}{r['name']}")

    os.makedirs(os.path.dirname(RESULT_JSON), exist_ok=True)
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "passed": tr.n_passed, "total": tr.n_total,
            "device": str(_DEVICE), "results": tr.results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n判定明细已写入: {RESULT_JSON}")

    ok = tr.n_passed == tr.n_total
    print("【结论】全部 PASS，可启动 E4 网格批量生成" if ok
          else "【结论】存在 FAIL 项，**禁止**启动批量生成")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # pyright: ignore
    raise SystemExit(main())
