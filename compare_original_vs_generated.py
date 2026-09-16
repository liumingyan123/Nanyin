# -*- coding: utf-8 -*-
"""
compare_original_vs_generated.py — 原始南音 vs 模型生成南音 NOEF 对比评估
=========================================================================

目标：验证 NOEF 南音评估体系的合理性与保真性。

实验设计：
  Tier 1 (符号级, GT-参照): 将 Ground-Truth token 视作"天花板"，与
            BoYaTCN / MusicMamba 生成的 token 分别计算全部 9 项 NOEF 指标。
  Tier 2 (音频级, 盲评):    对原始 WAV 和生成 WAV 分别运行 evaluate_noef()，
            仅依据音频信号本身评估，不依赖任何 GT 标注。

预期结果：
  - 原始南音的 NOEF 得分应显著高于模型生成 → 证明指标体系的鉴别力
  - BoYaTCN 与 MusicMamba 在不同维度各有优劣 → 揭示架构差异

输出：
  results/comparison_original_vs_generated.csv  — Tier 1 + Tier 2 全表
  $stdout                                       — 格式化的对比报告

运行命令：
  venv/Scripts/python.exe compare_original_vs_generated.py
"""

import os
import sys
import csv
import json
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # pyright: ignore

from nanyin_metrics import (
    load_ground_truth,
    evaluate_single_song,
    evaluate_noef,
    compute_composite_score,
    NOEF_METRIC_NAMES,
    PROMPT_LEN,
)
from nanyin_metrics import _VAL_SONG_IDS as VAL_SONG_IDS
from nanyin_metrics import _SONG_CHINESE_NAMES as SONG_CN
from nanyin_metrics import _DATASET_BASE

# ============================================================
# 一、路径配置
# ============================================================

RESULTS_DIR   = os.path.join(_SCRIPT_DIR, "results")
MIDI_BASE     = os.path.join(RESULTS_DIR, "generated_midi")
AUDIO_BASE    = os.path.join(RESULTS_DIR, "generated_audio")
COMPARISON_CSV = os.path.join(RESULTS_DIR, "comparison_original_vs_generated.csv")

# 新流程（2026-09-02）：音频对比统一取自 v4 四声部合奏渲染中的琵琶轨 ens_pipa.wav
# 对比层面 = 主干旋律 midi：GT = 投入训练的 midi（BasicPitch 转录主干）经 v4 派生 → 琵琶轨
#  - GT 侧:  results/gt_ensemble_audio/{song}/ens_pipa.wav            (GT v4 四声部渲染, 琵琶轨)
#  - 生成侧: results/generated_ensemble_audio/{model}/{song}/ens_pipa.wav  (续写主干 v4 四声部渲染, 琵琶轨)
# 两侧渲染参数完全一致：derive_ensemble_v4.py --late-join 0 + ts_render_full.py --seed 2024 --ref refs/ref_pipa.wav
GT_AUDIO_BASE    = os.path.join(RESULTS_DIR, "gt_audio")
GEN_MAIN_BASE    = os.path.join(RESULTS_DIR, "gen_audio")
GT_ENSEMBLE_BASE  = os.path.join(RESULTS_DIR, "gt_ensemble_audio")
GEN_ENSEMBLE_BASE = os.path.join(RESULTS_DIR, "generated_ensemble_audio")


def gt_wav(song_id: str) -> str:
    """GT(原始南音) 琵琶参考 WAV：GT v4 四声部渲染中的琵琶轨 ens_pipa.wav（无晚入, seed 2024）。"""
    return os.path.join(GT_ENSEMBLE_BASE, song_id, "ens_pipa.wav")


def gen_main_wav(model_subdir: str, song_id: str) -> str:
    """生成侧琵琶 WAV：续写主干 v4 四声部渲染中的琵琶轨 ens_pipa.wav（无晚入, seed 2024）。"""
    return os.path.join(GEN_ENSEMBLE_BASE, model_subdir, song_id, "ens_pipa.wav")


def gt_ensemble_wav(song_id: str) -> str:
    """GT 四声部合奏 WAV（仅展示，不作评估参照）。"""
    return os.path.join(GT_ENSEMBLE_BASE, song_id, f"{song_id}_gt_ensemble.wav")


def gen_ensemble_wav(model_subdir: str, song_id: str) -> str:
    """生成四声部合奏 WAV（仅展示，不作评估参照）。"""
    return os.path.join(
        GEN_ENSEMBLE_BASE, model_subdir, song_id,
        f"{song_id}_{model_subdir}_ensemble.wav",
    )

MODELS = {
    "BoYaTCN":    "boyatcn",
    "MusicMamba": "musicmamba",
}

# CSV 完整列（Tier 1 + Tier 2）
CSV_HEADERS = [
    "song_id", "song_name", "source",
    # Tier 1 符号级（0-100）
    "t1_pitch_accuracy", "t1_pitch_stability",
    "t1_rhythm_consistency", "t1_tempo_accuracy",
    "t1_ornament_score", "t1_ornament_density",
    "t1_mode_matching_score",
    "t1_fad_distance", "t1_melody_cosine_similarity",
    "t1_composite_score",
    # Tier 2 音频级（0-1）
    "t2_pitch_accuracy", "t2_pitch_stability",
    "t2_rhythm_consistency", "t2_tempo_accuracy",
    "t2_ornament_score", "t2_ornament_density",
    "t2_mode_matching_score",
]

# ============================================================
# 二、前置步骤：生成 + MIDI→WAV
# ============================================================

def ensure_generated_outputs() -> bool:
    """
    确保产物就绪（新流程 2026-09-02，音频对比取四声部琵琶轨）：
      - 生成 MIDI: results/generated_midi/{model}/{song}_{model}.mid
      - 生成侧琵琶轨: results/generated_ensemble_audio/{model}/{song}/ens_pipa.wav
      - GT 侧琵琶轨:  results/gt_ensemble_audio/{song}/ens_pipa.wav
    渲染管线：derive_ensemble_v4.py --late-join 0 → ts_render_full.py --seed 2024 --ref refs/ref_pipa.wav
    """
    print("\n  ╔" + "═" * 68 + "╗")
    print("  ║  前置步骤：确保产物就绪 (v4 四声部琵琶轨)" + " " * 18 + "║")
    print("  ╚" + "═" * 68 + "╝")

    ok = True

    # Step 1: 生成 MIDI
    for model_name, subdir in MODELS.items():
        for sid in VAL_SONG_IDS:
            midi_file = os.path.join(MIDI_BASE, subdir, f"{sid}_{model_name.lower()}.mid")
            if not os.path.exists(midi_file):
                print(f"  ✗ 生成 MIDI 缺失: {midi_file}")
                print("    请先运行 generate_music.py")
                ok = False

    # Step 2: 生成侧琵琶轨 WAV（续写主干 v4 四声部渲染 → ens_pipa.wav）
    for model_name, subdir in MODELS.items():
        for sid in VAL_SONG_IDS:
            wav_file = gen_main_wav(subdir, sid)
            if not os.path.exists(wav_file):
                print(f"  ✗ 生成侧琵琶轨缺失: {wav_file}")
                print("    请先运行 render_ens_v4_newmidi.sh (WSL tmux renderv4)")
                ok = False

    # Step 3: GT 侧琵琶轨 WAV（GT 主干 v4 四声部渲染 → ens_pipa.wav，评估基线）
    for sid in VAL_SONG_IDS:
        wav_file = gt_wav(sid)
        if not os.path.exists(wav_file):
            print(f"  ✗ GT 琵琶轨缺失: {wav_file}")
            print("    请先运行 render_gt_ensemble_audio.sh")
            ok = False

    if ok:
        print("  ✓ 全部产物就绪 (MIDI + 四声部琵琶轨 WAV)")
    return ok


# ============================================================
# 三、Tier 1: 符号级 GT-参照评估
# ============================================================

def evaluate_tier1_original(
    song_id: str, gt: Dict[str, Any]
) -> Dict[str, float]:
    """
    将原始 GT token 自身送入 evaluate_single_song()。
    这给出理论天花板分数（GT vs GT，pitch_accuracy = 100 等）。
    """
    tokens = list(gt["tokens"])
    durations = list(gt["durations"])
    # FAD 参考侧：GT 主干渲染（投入训练的 midi → TokenSynth 琵琶，与生成侧主干同管线）
    audio_path = gt_wav(song_id)

    if len(tokens) < 2:
        return {k: 0.0 for k in NOEF_METRIC_NAMES}

    # GT vs GT 天花板评估：不需要跳过 prompt（没有生成环节）。
    # 传 gt["starts"] 保证 IOI 由真实 onset 计算（自比节奏应为满分）。
    gt_starts = gt.get("starts")
    metrics = evaluate_single_song(
        tokens, durations, gt, gen_audio_path=audio_path,
        skip_prompt=0, gen_starts=gt_starts,
    )
    return metrics


def evaluate_tier1_generated(
    song_id: str,
    model_name: str,
    gt: Dict[str, Any],
) -> Optional[Dict[str, float]]:
    """
    解析生成 MIDI 文件，送入 evaluate_single_song() 对比 GT。
    """
    midi_filename = f"{song_id}_{model_name.lower()}.mid"
    midi_subdir = MODELS[model_name]
    midi_path = os.path.join(MIDI_BASE, midi_subdir, midi_filename)

    if not os.path.exists(midi_path):
        print(f"    ⚠ 生成 MIDI 缺失: {midi_path}")
        return None

    gen_audio_path = gen_main_wav(midi_subdir, song_id)

    from nanyin_metrics import parse_midi_to_sequence
    gen_tokens, gen_durations, gen_starts, _ = parse_midi_to_sequence(midi_path)
    if not gen_tokens:
        print(f"    ⚠ MIDI 解析为空: {midi_path}")
        return None

    metrics = evaluate_single_song(
        gen_tokens, gen_durations, gt, gen_audio_path=gen_audio_path,
        skip_prompt=PROMPT_LEN, gen_starts=gen_starts,
    )
    return metrics


# ============================================================
# 四、Tier 2: 音频级盲评
# ============================================================

def evaluate_tier2(wav_path: str, mode_pcs: str = "gong") -> Dict[str, Optional[float]]:
    """对 WAV 音频运行纯音频 NOEF 分析。"""
    if not os.path.exists(wav_path):
        return {k: None for k in NOEF_METRIC_NAMES}

    return evaluate_noef(
        audio_path=wav_path,
        mode_pcs=mode_pcs,
    )


# ============================================================
# 五、格式化输出
# ============================================================

def print_comparison_table(all_rows: List[Dict[str, Any]]):
    """打印格式化的对比报告。"""
    print()
    print("  ╔" + "═" * 68 + "╗")
    print("  ║   NOEF 对比评估报告 — 原始南音 vs 模型生成" + " " * 17 + "║")
    print("  ╚" + "═" * 68 + "╝")
    print()

    # ── Tier 1 符号级汇总 ──
    print("  ┌─ Tier 1: 符号级 GT-参照评估 (0–100) ────────────────────────────┐")
    _print_tier_summary(all_rows, tier="t1", is_composite_range_100=True)

    # ── Tier 2 音频级汇总 ──
    print("  ┌─ Tier 2: 音频级盲评 (0–1) ──────────────────────────────────────┐")
    _print_tier_summary(all_rows, tier="t2", is_composite_range_100=False)

    # ── 逐曲目明细 ──
    print("  ┌─ 逐曲目 Tier 1 composite 明细 ───────────────────────────────────┐")
    sources = ["原始南音", "BoYaTCN", "MusicMamba"]
    for sid in VAL_SONG_IDS:
        cn = SONG_CN.get(sid, sid)
        vals = []
        for src in sources:
            row = _find_row(all_rows, sid, src)
            if row and row.get("t1_composite_score") is not None:
                vals.append(f"{row['t1_composite_score']:.1f}")
            else:
                vals.append("N/A")
        print(f"  │ {cn:6s}  │  原始: {vals[0]:>5s}  │  BoYaTCN: {vals[1]:>5s}  │  MusicMamba: {vals[2]:>5s}  │")
    print("  └──────────────────────────────────────────────────────────────────┘")

    # ── 关键结论 ──
    print()
    print("  ┌─ 结论验证 ───────────────────────────────────────────────────────┐")
    _print_conclusions(all_rows)
    print("  └──────────────────────────────────────────────────────────────────┘")
    print()


def _find_row(rows: List[Dict], song_id: str, source: str) -> Optional[Dict]:
    for r in rows:
        if r.get("song_id") == song_id and r.get("source") == source:
            return r
    return None


def _print_tier_summary(all_rows: List[Dict], tier: str, is_composite_range_100: bool):
    """打印单个 Tier 的按来源汇总。"""
    prefix = f"{tier}_"
    sources = ["原始南音", "BoYaTCN", "MusicMamba"]
    # 筛选出 tier 有数据的行
    key_metrics = [
        "pitch_accuracy", "pitch_stability", "rhythm_consistency",
        "ornament_score", "mode_matching_score",
    ]

    # 按来源汇总
    for src in sources:
        src_rows = [r for r in all_rows
                    if r.get("source") == src
                    and r.get(f"{prefix}pitch_stability") is not None]
        if not src_rows:
            print(f"  │ {src:8s}  (无数据)")
            continue

        # 取各指标均值
        parts = []
        for km in key_metrics:
            vals = [r.get(f"{prefix}{km}", 0) or 0 for r in src_rows]
            avg = float(np.mean(vals)) if vals else 0.0
            parts.append(f"{km}={avg:.3f}")

        line = f"  │ {src:8s}  " + " | ".join(parts)
        print(f"  │ {line} │")

        # composite
        if is_composite_range_100:
            composites = [r.get("t1_composite_score", 0) or 0 for r in src_rows]
            avg_c = float(np.mean(composites)) if composites else 0.0
            print(f"  │           ★ composite = {avg_c:.1f}/100")
    print("  └──────────────────────────────────────────────────────────────────┘")
    print()


def _print_conclusions(all_rows: List[Dict]):
    """打印关键结论。"""
    # 比较原始 vs 模型的 composite
    orig_rows = [r for r in all_rows if r.get("source") == "原始南音"
                 and r.get("t1_composite_score") is not None]
    boya_rows = [r for r in all_rows if r.get("source") == "BoYaTCN"
                 and r.get("t1_composite_score") is not None]
    mm_rows   = [r for r in all_rows if r.get("source") == "MusicMamba"
                 and r.get("t1_composite_score") is not None]

    if orig_rows and boya_rows:
        o = float(np.mean([r["t1_composite_score"] for r in orig_rows]))
        b = float(np.mean([r["t1_composite_score"] for r in boya_rows]))
        delta_b = o - b
        print(f"  │ 原始南音 Tier 1 均值: {o:.1f}/100")
        print(f"  │ BoYaTCN    Tier 1 均值: {b:.1f}/100  (差距: {delta_b:+.1f})")
        if delta_b > 0:
            print(f"  │ ✓ NOEF 成功捕获原始-生成差异 — 体系具备鉴别力")
        else:
            print(f"  │ ⚠ 原始分数未显著高于生成 — 需检查模型或指标")

    if orig_rows and mm_rows:
        m = float(np.mean([r["t1_composite_score"] for r in mm_rows]))
        delta_m = o - m
        print(f"  │ MusicMamba Tier 1 均值: {m:.1f}/100  (差距: {delta_m:+.1f})")

    if boya_rows and mm_rows:
        b = float(np.mean([r["t1_composite_score"] for r in boya_rows]))
        m = float(np.mean([r["t1_composite_score"] for r in mm_rows]))
        better = "BoYaTCN" if b > m else "MusicMamba"
        print(f"  │ 两模型对比: {better} 略优 (Δ={abs(b-m):.1f})")


# ============================================================
# 六、主流程
# ============================================================

def main():
    """主入口：运行完整对比评估。"""
    print("=" * 70)
    print("  原始南音 vs 模型生成 — NOEF 对比评估")
    print("=" * 70)
    print(f"  验证集曲目: {[SONG_CN.get(s, s) for s in VAL_SONG_IDS]}")
    print(f"  模型: {list(MODELS.keys())}")
    print()

    # ── 0. 确保生成产物就绪 ──
    if not ensure_generated_outputs():
        print("[错误] 前置步骤失败，终止")
        sys.exit(1)

    # ── 1. 逐曲目评估 ──
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 写 CSV 表头
    with open(COMPARISON_CSV, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADERS)

    all_rows: List[Dict[str, Any]] = []

    for song_id in VAL_SONG_IDS:
        cn_name = SONG_CN.get(song_id, song_id)
        print(f"\n{'─' * 60}")
        print(f"  ▶ 评估曲目: {cn_name} ({song_id})")
        print(f"{'─' * 60}")

        # 加载 GT
        gt = load_ground_truth(song_id)
        if gt is None:
            print(f"  ✗ GT 加载失败，跳过")
            continue

        # 音频参照统一为 v4 四声部琵琶轨（与生成侧 gen_audio_path 同管线，FAD 公平）
        gt_audio_path = gt_wav(song_id)
        if os.path.exists(gt_audio_path):
            gt["audio_path"] = gt_audio_path

        mode_key = gt.get("mode_key", "gong")
        print(f"  GT: seq_len={len(gt['tokens'])}, mode={mode_key}")

        # ── Tier 1: 符号级 ──
        print("\n  [Tier 1] 符号级 GT-参照评估...")

        # 原始天花板
        t1_orig = evaluate_tier1_original(song_id, gt)
        t1_orig_comp = compute_composite_score(t1_orig)
        print(f"    原始南音 (天花板): composite={t1_orig_comp:.1f}/100")

        # BoYaTCN
        t1_boya = evaluate_tier1_generated(song_id, "BoYaTCN", gt)
        t1_boya_comp = compute_composite_score(t1_boya) if t1_boya else 0.0
        print(f"    BoYaTCN:          composite={t1_boya_comp:.1f}/100")

        # MusicMamba
        t1_mm = evaluate_tier1_generated(song_id, "MusicMamba", gt)
        t1_mm_comp = compute_composite_score(t1_mm) if t1_mm else 0.0
        print(f"    MusicMamba:       composite={t1_mm_comp:.1f}/100")

        # ── Tier 2: 音频级盲评 ──
        print("\n  [Tier 2] 音频级盲评 (TokenSynth 主干渲染)...")

        # 原始侧：GT 主干渲染（投入训练的 midi → TokenSynth 琵琶，与生成侧主干同管线）
        orig_wav = gt_wav(song_id)
        t2_orig = evaluate_tier2(orig_wav, mode_pcs=mode_key) if os.path.exists(orig_wav) else {}
        ps = t2_orig.get("pitch_stability", 0) or 0
        print(f"    原始主干:  pitch_stability={ps:.3f}  ({os.path.basename(orig_wav)})")

        # BoYaTCN 主干
        boya_wav = gen_main_wav("boyatcn", song_id)
        t2_boya = evaluate_tier2(boya_wav, mode_pcs=mode_key) if os.path.exists(boya_wav) else {}
        ps_b = t2_boya.get("pitch_stability", 0) or 0
        print(f"    BoYaTCN 主干:   pitch_stability={ps_b:.3f}  ({os.path.basename(boya_wav)})")

        # MusicMamba 主干
        mm_wav = gen_main_wav("musicmamba", song_id)
        t2_mm = evaluate_tier2(mm_wav, mode_pcs=mode_key) if os.path.exists(mm_wav) else {}
        ps_m = t2_mm.get("pitch_stability", 0) or 0
        print(f"    MusicMamba 主干: pitch_stability={ps_m:.3f}  ({os.path.basename(mm_wav)})")

        # ── 汇总写入 ──
        def _build_row(source, t1, t1_comp, t2) -> Dict[str, Any]:
            row = {"song_id": song_id, "song_name": cn_name, "source": source}
            if t1:
                for k in NOEF_METRIC_NAMES:
                    row[f"t1_{k}"] = round(t1.get(k, 0), 2)
                row["t1_composite_score"] = round(t1_comp, 2)
            if t2:
                for k in NOEF_METRIC_NAMES:
                    val = t2.get(k)
                    row[f"t2_{k}"] = round(val, 4) if val is not None else ""
            return row

        rows_to_write = [
            _build_row("原始南音",    t1_orig, t1_orig_comp, t2_orig),
            _build_row("BoYaTCN",    t1_boya, t1_boya_comp, t2_boya),
            _build_row("MusicMamba", t1_mm,   t1_mm_comp,   t2_mm),
        ]

        with open(COMPARISON_CSV, "a", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            for r in rows_to_write:
                writer.writerow([r.get(k, "") for k in CSV_HEADERS])

        all_rows.extend(rows_to_write)

    # ── 2. 打印报告 ──
    print_comparison_table(all_rows)

    print(f"\n  完整结果已写入: {COMPARISON_CSV}")
    print(f"  {'='*70}")


if __name__ == "__main__":
    main()
