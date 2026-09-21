# -*- coding: utf-8 -*-
"""
tests/test_window_slice.py — E2 复合事件窗口切分单元自测
=========================================================

校验目标（对应 E2 的四条硬性约束与任务要求）：

  T1 窗口切分边界与数量公式（合成序列，长度已知）
  T2 窗口长度严格等于 window_size，不足长度的尾段被丢弃（不 padding）
  T3 曲目 ID 隔离：训练集 / 验证集曲目不相交，窗口不跨曲
  T4 BOS / EOS 标记与 attention_mask 布局正确
  T5 窗口末尾 time-shift 伪标签被 ignore_index 屏蔽（含损失/梯度层面的等价性验证）
  T6 窗口内容与整曲编码切片一致、解码后 onset 单调不回退
  T7 真实数据窗口统计（逐曲窗口数 / 尾部丢弃 / max_notes 截断清单）

运行：
    venv/Scripts/python.exe tests/test_window_slice.py

产物：
    results/e2_window_slice_test.json   （测试结果明细，被 .gitignore 排除）
"""

import json
import os
import sys
from typing import Any, Dict, List

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (_PROJECT_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn.functional as F

from nanyin_config import load_config, E2_CONFIG_PATH
from dataset_nanyin import (
    IGNORE_INDEX,
    TRAIN_SPLIT_IDS,
    VAL_SPLIT_IDS,
    CompoundEventWindowDataset,
    build_window_sample,
    get_compound_window_datasets,
    get_compound_codec,
    slice_event_windows,
)


# ==============================================================================
# 工具
# ==============================================================================

class Checker:
    """收集 PASS / FAIL 判定项。"""

    def __init__(self):
        self.items: List[Dict[str, Any]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.items.append({"name": name, "pass": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        return bool(ok)

    @property
    def all_pass(self) -> bool:
        return all(it["pass"] for it in self.items)

    def summary_line(self) -> str:
        n_pass = sum(1 for it in self.items if it["pass"])
        return f"{n_pass}/{len(self.items)} 项通过"


def make_synthetic_events(n: int) -> Dict[str, Any]:
    """构造长度为 n 的合成整曲事件序列（pitch 循环、duration/shift 取小档位）。"""
    return {
        "song_id": f"synthetic_{n}",
        "pitch_ids": [60 + (i % 12) for i in range(n)],
        "duration_ids": [10 + (i % 5) for i in range(n)],
        "shift_ids": [5 + (i % 7) for i in range(n)],
        "num_events": n,
        "stats": {
            "num_notes": n,
            "truncated_notes": 0,
            "duration_overflow": 0,
            "shift_overflow": 0,
            "negative_gap_clamped": 0,
        },
    }


# ==============================================================================
# T1 窗口切分边界与数量公式
# ==============================================================================

def test_t1_window_counts(cfg: dict, checker: Checker) -> Dict[str, Any]:
    win = int(cfg["window"]["window_size_events"])
    train_stride = int(cfg["window"]["train_stride_events"])
    eval_stride = int(cfg["window"]["eval_stride_events"])

    print(f"\n[T1] 窗口切分边界（window_size={win}, train_stride={train_stride}, eval_stride={eval_stride}）")

    cases = [
        # (序列长度, 步长, 期望窗口数, 说明)
        (255, train_stride, 0, "不足一个窗口 → 直接丢弃"),
        (256, train_stride, 1, "恰好一个窗口"),
        (257, train_stride, 1, "多 1 个事件仍只有 1 个窗口（尾部丢弃）"),
        (384, train_stride, 2, "起点 0/128（0+256=256、128+256=384 合法；256+256=512 越界）"),
        (600, train_stride, 3, "起点 0/128/256（384+256=640 越界）"),
        (600, eval_stride, 2, "非重叠：起点 0/256（512+256=768 越界）"),
        (1024, eval_stride, 4, "非重叠：起点 0/256/512/768"),
    ]

    rows = []
    all_ok = True
    for n, stride, expect, note in cases:
        wins = slice_event_windows(n, win, stride)
        ok = len(wins) == expect
        all_ok = all_ok and ok
        rows.append({"num_events": n, "stride": stride, "windows": len(wins),
                     "expected": expect, "note": note, "pass": ok})
        print(f"    n={n:<5} stride={stride:<4} → {len(wins)} 窗口 (期望 {expect})  {note}")

    checker.add("T1 窗口数量公式与边界（含不足窗口丢弃）", all_ok,
                f"{sum(1 for r in rows if r['pass'])}/{len(rows)} 个子用例一致")

    # 所有窗口等长
    wins = slice_event_windows(600, win, train_stride)
    lens = {e - s for s, e in wins}
    checker.add("T1 所有窗口长度恒等于 window_size", lens == {win}, f"实测长度集合 {sorted(lens)}")
    return {"cases": rows}


# ==============================================================================
# T2 窗口长度严格 / 不做 padding
# ==============================================================================

def test_t2_no_padding(cfg: dict, checker: Checker) -> Dict[str, Any]:
    win = int(cfg["window"]["window_size_events"])
    print(f"\n[T2] 窗口长度严格性（不做 padding，样本长度 = {win} + 1）")

    codec = get_compound_codec(cfg)
    ev = make_synthetic_events(win * 2 + 37)     # 故意让尾段不足一个窗口
    wins = slice_event_windows(ev["num_events"], win, int(cfg["window"]["train_stride_events"]))

    lengths = []
    for s, e in wins:
        sample = build_window_sample(ev, s, e, codec=codec)
        lengths.append(int(sample["input_pitch_ids"].shape[0]))
    ok_len = set(lengths) == {win + 1}
    checker.add("T2 合成序列：所有样本长度 = window_size + 1（BOS/EOS 侧）", ok_len,
                f"实测长度集合 {sorted(set(lengths))}")

    # 尾段不足窗口 → 不产生样本（padding 若存在则最后一个样本长度仍为 win+1，但事件数不足）
    last_end = wins[-1][1]
    discarded = ev["num_events"] - last_end
    checker.add("T2 不足 window_size 的尾段被丢弃（未 padding）",
                discarded > 0 and discarded < win,
                f"尾段丢弃 {discarded} 个事件（< {win}），未生成残缺窗口")

    # 目标侧：非 EOS 位置的真实事件数恰为 window_size
    sample = build_window_sample(ev, wins[0][0], wins[0][1], codec=codec)
    n_event_targets = int((sample["target_pitch_ids"] != codec.pitch_eos_id).sum())
    checker.add("T2 目标侧真实事件数 = window_size（无 padding 位）",
                n_event_targets == win, f"实测 {n_event_targets}")
    return {"lengths": lengths, "discarded_tail": int(discarded)}


# ==============================================================================
# T3 曲目 ID 隔离
# ==============================================================================

def test_t3_song_isolation(train_ds, val_ds, checker: Checker) -> Dict[str, Any]:
    print("\n[T3] 曲目 ID 隔离（同曲窗口不跨训练/验证集）")

    inter = set(TRAIN_SPLIT_IDS) & set(VAL_SPLIT_IDS)
    checker.add("T3 训练/验证曲目 ID 无交集", len(inter) == 0,
                f"交集 {sorted(inter)}" if inter else f"训练 {len(TRAIN_SPLIT_IDS)} 首 / 验证 {len(VAL_SPLIT_IDS)} 首，无交集")

    train_songs = {r["song_id"] for r in train_ds.records}
    val_songs = {r["song_id"] for r in val_ds.records}
    checker.add("T3 窗口所属曲目不跨集（窗口级再确认）",
                len(train_songs & val_songs) == 0,
                f"训练集出现曲目 {len(train_songs)} 首，验证集 {len(val_songs)} 首，交集 0")

    # 每个窗口只取自一首曲子（窗口记录自带 song_id，且 start/end 落在该曲事件数内）
    bad = []
    for ds in (train_ds, val_ds):
        for r in ds.records:
            n_events = ds._song_events[r["song_id"]]["num_events"]
            if not (0 <= r["start"] < r["end"] <= n_events):
                bad.append(r)
    checker.add("T3 所有窗口区间落在单曲事件范围内（窗口不跨曲拼接）", len(bad) == 0,
                f"越界窗口 {len(bad)} 个")
    return {"train_songs": sorted(train_songs), "val_songs": sorted(val_songs)}


# ==============================================================================
# T4 BOS / EOS 布局与 attention_mask
# ==============================================================================

def test_t4_bos_eos(train_ds, checker: Checker) -> Dict[str, Any]:
    print("\n[T4] BOS / EOS 标记与 attention_mask")

    codec = train_ds.codec
    sample = train_ds[0]
    L = int(sample["input_pitch_ids"].shape[0])

    ok_bos = (
        int(sample["input_pitch_ids"][0]) == codec.pitch_bos_id
        and int(sample["input_duration_ids"][0]) == codec.duration_bos_id
        and int(sample["input_shift_ids"][0]) == codec.shift_bos_id
    )
    checker.add("T4 输入首位 = BOS（三路一致）", ok_bos,
                f"pitch_bos={codec.pitch_bos_id}, duration_bos={codec.duration_bos_id}, shift_bos={codec.shift_bos_id}")

    ok_eos = (
        int(sample["target_pitch_ids"][-1]) == codec.pitch_eos_id
        and int(sample["target_duration_ids"][-1]) == codec.duration_eos_id
    )
    checker.add("T4 目标末位 = EOS（pitch/duration）", ok_eos,
                f"pitch_eos={codec.pitch_eos_id}, duration_eos={codec.duration_eos_id}")

    ok_shift = (
        int(sample["input_pitch_ids"].shape[0]) == int(sample["target_pitch_ids"].shape[0]) == L
    )
    checker.add("T4 输入/目标长度对齐（自回归 shift-by-one）", ok_shift, f"长度均为 {L}")

    ok_align_p = torch.equal(sample["target_pitch_ids"][:-1], sample["input_pitch_ids"][1:])
    ok_align_d = torch.equal(sample["target_duration_ids"][:-1], sample["input_duration_ids"][1:])
    checker.add("T4 pitch/duration 目标 = 输入右移一位", ok_align_p and ok_align_p and ok_align_d,
                f"pitch 对齐={ok_align_p}, duration 对齐={ok_align_d}")

    mask = sample["attention_mask"]
    ok_mask = (
        int(mask.shape[0]) == L and int(mask.sum()) == L and bool((mask == 1).all())
    )
    checker.add("T4 attention_mask 全 1（等长窗口无 padding，故无屏蔽位）", ok_mask,
                f"长度 {int(mask.shape[0])}，sum={int(mask.sum())}；"
                f"说明：本阶段窗口严格等长、不做 padding，掩码恒为全 1，"
                f"引入变长 batch 后该掩码才会出现 0")
    return {"seq_len": L}


# ==============================================================================
# T5 窗口末尾 time-shift 伪标签的 ignore_index 掩码
# ==============================================================================

def test_t5_shift_mask(train_ds, checker: Checker) -> Dict[str, Any]:
    print("\n[T5] 窗口末尾 time-shift 伪标签屏蔽（E1 硬性约束）")

    codec = train_ds.codec
    sample = train_ds[0]
    tgt_shift = sample["target_shift_ids"]
    L = int(tgt_shift.shape[0])

    # 1) 位置检查：最后一个事件位（索引 -2）与 EOS 位（索引 -1）被屏蔽
    ok_pos = int(tgt_shift[-1]) == IGNORE_INDEX and int(tgt_shift[-2]) == IGNORE_INDEX
    checker.add("T5 目标序列末两位（最后事件位 + EOS 位）被置 IGNORE_INDEX", ok_pos,
                f"target_shift[-2]={int(tgt_shift[-2])}, target_shift[-1]={int(tgt_shift[-1])}, IGNORE_INDEX={IGNORE_INDEX}")

    ok_rest = bool((tgt_shift[:-2] != IGNORE_INDEX).all())
    checker.add("T5 其余位置均保留真实 shift 标签", ok_rest,
                f"有效监督位置 {L - 2} / {L}")

    # 2) 伪标签值：输入侧最后事件的 shift 被置 0（未定义）
    ok_pad = int(sample["input_shift_ids"][-1]) == codec.pad_id
    checker.add("T5 输入侧窗口末尾 shift = 0（伪标签，非真实值）", ok_pad,
                f"input_shift[-1]={int(sample['input_shift_ids'][-1])}")

    # 3) 损失层面：ignore_index 与"手工只算有效位"完全等价
    torch.manual_seed(0)
    C = codec.shift_vocab_total
    logits = torch.randn(1, L, C, dtype=torch.float64, requires_grad=True)
    tgt = tgt_shift.unsqueeze(0)
    loss_masked = F.cross_entropy(logits.reshape(-1, C), tgt.reshape(-1), ignore_index=IGNORE_INDEX)

    valid = tgt.reshape(-1) != IGNORE_INDEX
    manual = F.cross_entropy(logits.reshape(-1, C)[valid], tgt.reshape(-1)[valid])
    ok_eq = float((loss_masked - manual).abs()) < 1e-9
    checker.add("T5 CE(ignore_index) == 手工仅对有效位求 CE", ok_eq,
                f"masked={float(loss_masked):.6f}, manual={float(manual):.6f}, 差={float((loss_masked-manual).abs()):.2e}")

    # 4) 掩码确实改变了结果：若不屏蔽（伪标签 0 参与）损失不同
    tgt_nomask = tgt.clone()
    tgt_nomask[tgt_nomask == IGNORE_INDEX] = 0
    loss_nomask = F.cross_entropy(logits.reshape(-1, C), tgt_nomask.reshape(-1))
    ok_diff = float((loss_masked - loss_nomask).abs()) > 1e-9
    checker.add("T5 不屏蔽时损失不同（证明掩码真实生效，非占位）", ok_diff,
                f"屏蔽后={float(loss_masked):.6f}, 未屏蔽={float(loss_nomask):.6f}")

    # 5) 梯度层面：被屏蔽位置的 logits 梯度恒为 0（伪标签不参与反向传播）
    logits2 = torch.randn(1, L, C, dtype=torch.float64, requires_grad=True)
    loss2 = F.cross_entropy(logits2.reshape(-1, C), tgt.reshape(-1), ignore_index=IGNORE_INDEX)
    loss2.backward()
    grad = logits2.grad[0]
    masked_idx = (tgt.reshape(-1) == IGNORE_INDEX).nonzero(as_tuple=True)[0]
    grad_at_masked = grad[masked_idx].abs().sum().item()
    grad_at_valid = grad[valid].abs().sum().item()
    ok_grad = grad_at_masked == 0.0 and grad_at_valid > 0.0
    checker.add("T5 被屏蔽位置梯度严格为 0（伪标签不回传）", ok_grad,
                f"屏蔽位梯度绝对值和={grad_at_masked:.1e}, 有效位={grad_at_valid:.3e}")

    return {
        "seq_len": L,
        "valid_shift_positions": int(L - 2),
        "masked_positions": 2,
        "loss_masked": float(loss_masked),
        "loss_without_mask": float(loss_nomask),
    }


# ==============================================================================
# T6 窗口内容与整曲切片一致 + 解码 onset 单调
# ==============================================================================

def test_t6_slice_consistency(train_ds, checker: Checker) -> Dict[str, Any]:
    print("\n[T6] 窗口内容与整曲编码切片一致 / 解码 onset 单调")

    codec = train_ds.codec
    rec = train_ds.records[0]
    song_events = train_ds._song_events[rec["song_id"]]
    s, e = rec["start"], rec["end"]

    exp_pitch = song_events["pitch_ids"][s:e]
    exp_duration = song_events["duration_ids"][s:e]
    exp_shift = song_events["shift_ids"][s:e]

    sample = train_ds[0]
    got_pitch = sample["input_pitch_ids"][1:].tolist()
    got_duration = sample["input_duration_ids"][1:].tolist()
    got_shift = sample["input_shift_ids"][1:].tolist()

    ok_p = got_pitch == exp_pitch
    ok_d = got_duration == exp_duration
    # shift：窗口末位被置 0（伪标签），其余与整曲切片一致
    ok_s = got_shift[:-1] == exp_shift[:-1] and got_shift[-1] == codec.pad_id
    checker.add("T6 窗口事件 = 整曲编码对应切片（shift 末位置 0）", ok_p and ok_d and ok_s,
                f"pitch 一致={ok_p}, duration 一致={ok_d}, shift 一致(末位伪标签除外)={ok_s}")

    # 解码：onset 单调不回退
    decoded = codec.decode_events(
        exp_pitch,
        exp_duration,
        exp_shift[:-1] + [codec.pad_id],
        start_offset_sec=0.0,
    )
    onsets = [float(n["start_time"]) for n in decoded]
    mono = all(onsets[i + 1] >= onsets[i] for i in range(len(onsets) - 1))
    checker.add("T6 解码 onset 单调不回退", mono,
                f"首 onset={onsets[0]:.3f}s, 末 onset={onsets[-1]:.3f}s, 事件数={len(onsets)}")
    return {"song_id": rec["song_id"], "start": int(s), "end": int(e),
            "onset_first": onsets[0], "onset_last": onsets[-1]}


# ==============================================================================
# T7 真实数据统计
# ==============================================================================

def test_t7_real_stats(train_ds, val_ds, checker: Checker) -> Dict[str, Any]:
    print("\n[T7] 真实数据窗口统计（训练集滑动 / 验证集非重叠）")

    rows = train_ds.song_stats_rows() + val_ds.song_stats_rows()
    print(f"    {'曲目':<28}{'划分':<7}{'事件数':>8}{'窗口数':>8}{'尾丢弃':>8}{'max_notes截断':>14}")
    for r in rows:
        print(f"    {r['song_id']:<28}{r['split']:<7}{r['num_events']:>8}{r['num_windows']:>8}"
              f"{r['discarded_tail_events']:>8}{r['truncated_notes']:>14}")

    total_train = len(train_ds)
    total_val = len(val_ds)
    print(f"    训练集窗口 {total_train}，验证集窗口 {total_val}")

    checker.add("T7 训练集 / 验证集窗口数均 > 0", total_train > 0 and total_val > 0,
                f"train={total_train}, val={total_val}")

    truncated = train_ds.truncated_songs() + val_ds.truncated_songs()
    if truncated:
        print(f"    ⚠ 被 max_notes 截断的曲目（如实记录，共 {len(truncated)} 首）：")
        for r in truncated:
            print(f"        - {r['song_id']} [{r['split']}]：丢弃 {r['truncated_notes']} 个音符")
    else:
        print("    无曲目被 max_notes 截断")

    return {
        "num_train_windows": int(total_train),
        "num_val_windows": int(total_val),
        "rows": rows,
        "truncated_songs": truncated,
    }


# ==============================================================================
# 主流程
# ==============================================================================

def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):           # type: ignore[attr-defined]
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

    cfg = load_config(E2_CONFIG_PATH)
    print("=" * 74)
    print("E2 单元自测：复合事件窗口切分 / BOS-EOS / 伪标签掩码")
    print("=" * 74)

    checker = Checker()

    t1 = test_t1_window_counts(cfg, checker)
    t2 = test_t2_no_padding(cfg, checker)

    print("\n[数据集] 构建真实训练/验证窗口数据集（11 首 BasicPitch 转录）...")
    train_ds, val_ds = get_compound_window_datasets(cfg=cfg, verbose=False)
    # 统计表由 T7 打印（此处不再重复）

    t3 = test_t3_song_isolation(train_ds, val_ds, checker)
    t4 = test_t4_bos_eos(train_ds, checker)
    t5 = test_t5_shift_mask(train_ds, checker)
    t6 = test_t6_slice_consistency(train_ds, checker)
    t7 = test_t7_real_stats(train_ds, val_ds, checker)

    # ---- 结果落盘 ----
    out_path = os.path.join(_PROJECT_ROOT, cfg["outputs"]["window_test_json"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "config": {
            "window_size_events": cfg["window"]["window_size_events"],
            "train_stride_events": cfg["window"]["train_stride_events"],
            "eval_stride_events": cfg["window"]["eval_stride_events"],
            "drop_tail": cfg["window"]["drop_tail"],
        },
        "checks": checker.items,
        "T1": t1, "T2": t2, "T3": t3, "T4": t4, "T5": t5, "T6": t6, "T7": t7,
        "all_pass": checker.all_pass,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 74)
    print(f"判定汇总：{checker.summary_line()}   结果明细: {out_path}")
    if checker.all_pass:
        print("结论：窗口切分 / BOS-EOS / 伪标签掩码 全部通过 → 自测 PASS")
        return 0
    print("结论：存在未通过项，详见上方 FAIL 条目与结果 JSON")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
