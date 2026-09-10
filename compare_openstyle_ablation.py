# -*- coding: utf-8 -*-
"""
compare_openstyle_ablation.py — 开放续写「无参考」消融评估（GT 作为南音特征参照）
==========================================================================

背景（2026-09-03 用户定义）:
  开放式南音旋律续写不再评估「生成旋律与原曲 GT 后半段有多像」。
  保留指标仍以该曲 GT 作为“真实南音特征”的参照（GT 分布=南音风格分布），
  但移除「只有当生成旋律与 GT 逐点相似才会得高分」的点对点指标：
      - t1_pitch_accuracy            (逐音位比对)
      - t1_melody_cosine_similarity  (整段旋律余弦相似)
      - t2_pitch_accuracy            (音频级恒 0.5 中性占位，无参考意义)

补充（2026-09-04）：节奏指标 t1_rhythm_consistency 已进一步完全无参考化，
  不再与 GT 的 IOI 分布做 KS 比对，改为评估生成片段自身的
  「IOI 稳定性 + 乐句内部撩拍网格规整度」（自动检测主导时值 u）。
  该指标不再依赖 GT；tempo_accuracy 仍以 GT 平均速度作南音速度参照。
  Tier2 音频级 rhythm_consistency 维持 beat-interval 自评估不变。

指标版次 t2（2026-09-04，本脚本当前口径）:
  - 删除   : t1_pitch_accuracy / t1_melody_cosine_similarity（点对点，已在上一版移除，
             本版起 evaluate_single_song 输出中仅作向后兼容遗留，不再进入任何报告列）
  - 移动   : t1_fad_distance → t2_fad_distance（FAD 属音频级风格距离，从符号级挪至 Tier2；
             参照音频 = 同曲 GT 渲染琵琶轨 ens_pipa.wav，生成/参照同域可比）
  - 新增(Tier1 符号级) : transition_smoothness（一阶相邻音程均值）/ melody_coherence
             （二阶差分即轮廓转折强度）——均无参考，只依赖生成 token 自身
  - 新增(Tier2 音频级) : pitch_contour_correlation（生成 f0 轮廓 vs 11 首南音
             test_audio 的 f0 风格画像 (μ,σ) 的相似度，画像首次构建后缓存 npz）

保留指标（评估生成片段自身是否具备南音音乐特征）:
  Tier1 符号级(0-100): pitch_stability / transition_smoothness / melody_coherence /
                       rhythm_consistency / tempo_accuracy / ornament_score /
                       ornament_density / mode_matching_score
  Tier2 音频级(0-1):   pitch_stability / pitch_contour_correlation /
                       rhythm_consistency / tempo_accuracy / ornament_score /
                       ornament_density / mode_matching_score / fad_distance

t1 与 t2 同名指标的计算方式区分（文档口径）:
  - t1_*（符号级）：作用在 parse_midi_to_sequence 得到的音高 token / IOI 序列上，
    按每个指标自身的定义直接计算（纯符号运算，见 nanyin_metrics 各函数 docstring）；
  - t2_*（音频级）：作用在渲染 WAV 波形上，由 librosa 提取 f0 / beats / MFCC 后计算，
    与 t1 同名指标共享"评分含义"但输入域与算法完全不同，两列数值独立，
    不是同一算法的复制，报告/表格中不得混用或相互推断。

t1_composite_score（t1_fad_distance 移至 Tier2 后，style 权重 0.10 按比例
分摊至其余四类并重归一化，总分仍为 0–100）:
      composite = 0.2778*pitch(stability, transition_smoothness, melody_coherence)
                + 0.2222*rhythm(rhythm_consistency, tempo_accuracy)
                + 0.2778*ornament(ornament_score, ornament_density)
                + 0.2222*mode(mode_matching_score)

消融组（架构对比）:
  - old_bidirectional      : 双向注意力原版模型  (checkpoint 消融前)
  - causal_autoregressive  : 因果自回归修改版模型
  每组 2 个模型: BoYaTCN / MusicMamba，验证曲 dapu14kouhuangtian / dapu15wujinjiao。
  另含一行 GT_reference：真实南音（GT 主干）在相同指标体系下的得分，
  作为“南音特征”的可达参照（GT vs GT 自洽分数）。

产物:
  results/generated_midi{_old}/{model}/{sid}_{model}.mid            — 主干旋律 MIDI
  results/generated_ensemble_audio{_old}/{model}/{sid}/ens_pipa.wav — v4 渲染琵琶轨
  results/gt_ensemble_audio/{sid}/ens_pipa.wav                      — GT 琵琶轨（南音参照）

输出:
  results/comparison_openstyle_ablation.csv
运行:
  venv/Scripts/python.exe compare_openstyle_ablation.py
"""

import os
import sys
import csv
from typing import Any, Dict, List, Optional

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # pyright: ignore

from nanyin_metrics import (  # noqa: E402
    load_ground_truth,
    evaluate_single_song,
    evaluate_noef,
    parse_midi_to_sequence,
    calculate_fad_score,
    PROMPT_LEN,
)
from nanyin_metrics import _VAL_SONG_IDS as VAL_SONG_IDS  # noqa: E402
from nanyin_metrics import _SONG_CHINESE_NAMES as SONG_CN  # noqa: E402

RESULTS_DIR = os.path.join(_SCRIPT_DIR, "results")
GT_ENS_BASE = os.path.join(RESULTS_DIR, "gt_ensemble_audio")

# ---------------------------------------------------------------
# 指标保留清单（不含任何「生成 vs GT 点对点」指标）
#  t1 指标版次 t2（2026-09-04）:
#    删除 t1_pitch_accuracy / t1_melody_cosine_similarity（点对点）
#    移动 t1_fad_distance → t2_fad_distance（音频级风格距离）
#    新增 t1 transition_smoothness / melody_coherence（符号级，无参考）
#    新增 t2 pitch_contour_correlation（音频级，11 首南音 f0 画像相似度）
# ---------------------------------------------------------------
T1_KEEP = [
    "pitch_stability", "transition_smoothness", "melody_coherence",
    "rhythm_consistency", "tempo_accuracy",
    "ornament_score", "ornament_density", "mode_matching_score",
]
T2_KEEP = [
    "pitch_stability", "pitch_contour_correlation",
    "rhythm_consistency", "tempo_accuracy",
    "ornament_score", "ornament_density", "mode_matching_score",
    "fad_distance",
]

# 类别权重（t1 指标版次 t2）：t1_fad_distance 移走后 style=0.10 按比例分摊，
# 使 pitch/rhythm/ornament/mode 的权重和为 1.0（0.25/0.90、0.20/0.90 重归一化）
_CW = {
    "pitch":    0.25 / 0.90,
    "rhythm":   0.20 / 0.90,
    "ornament": 0.25 / 0.90,
    "mode":     0.20 / 0.90,
}


def compute_openstyle_composite(t1: Dict[str, float]) -> float:
    """重算 t1_composite_score：仅用保留指标加权（style 类已移除并重归一化）。"""
    def _mean(keys: List[str]) -> float:
        vals = [float(t1.get(k, 0.0) or 0.0) for k in keys]
        return float(np.mean(vals)) if vals else 0.0

    # 音高/旋律线类：稳定性 + 过渡平滑 + 轮廓连贯（三者同源相邻音高关系，取均值）
    pitch = _mean(["pitch_stability", "transition_smoothness", "melody_coherence"])
    rhythm = _mean(["rhythm_consistency", "tempo_accuracy"])
    ornament = _mean(["ornament_score", "ornament_density"])
    mode = float(t1.get("mode_matching_score", 0.0) or 0.0)
    return (
        _CW["pitch"] * pitch
        + _CW["rhythm"] * rhythm
        + _CW["ornament"] * ornament
        + _CW["mode"] * mode
    )


# 架构消融配置: (architecture 名, 标签, MIDI 目录, 音频目录)
ARCHES = [
    ("old_bidirectional", "双向注意力原版",
     os.path.join(RESULTS_DIR, "generated_midi_old"),
     os.path.join(RESULTS_DIR, "generated_ensemble_audio_old")),
    ("causal_autoregressive", "因果自回归修改版",
     os.path.join(RESULTS_DIR, "generated_midi"),
     os.path.join(RESULTS_DIR, "generated_ensemble_audio")),
]
MODELS = {"BoYaTCN": "boyatcn", "MusicMamba": "musicmamba"}

CSV_HEADERS = (
    ["song_id", "song_name", "architecture", "architecture_label", "model_name"]
    + [f"t1_{k}" for k in T1_KEEP]
    + ["t1_composite_score"]
    + [f"t2_{k}" for k in T2_KEEP]
)


def gt_wav(song_id: str) -> str:
    """GT(真实南音) 琵琶参照音频：GT 主干 v4 渲染琵琶轨。"""
    return os.path.join(GT_ENS_BASE, song_id, "ens_pipa.wav")


def _gt_ceiling_tier1(song_id: str, gt: Dict[str, Any]) -> Dict[str, float]:
    """GT 自洽天花板：GT token 自身过 evaluate_single_song()，skip_prompt=0。"""
    tokens = list(gt["tokens"])
    durations = list(gt["durations"])
    if len(tokens) < 2:
        return {k: 0.0 for k in T1_KEEP}
    m = evaluate_single_song(
        tokens, durations, gt,
        gen_audio_path=gt_wav(song_id),
        skip_prompt=0, gen_starts=gt.get("starts"),
    )
    return {k: m.get(k, 0.0) for k in T1_KEEP}


def _run_tier2(wav_path: str, gt: Dict[str, Any]) -> Dict[str, float]:
    """
    音频级无参考评估；GT 的 tempo_bpm/mode_key 作为南音特征参照。

    - 常规指标：evaluate_noef() 直接从 WAV 波形提取 f0/beats 计算；
    - t2_fad_distance：FAD 参照 = 同曲 GT 渲染琵琶轨（gt["audio_path"]，
      由 main() 提前写入），生成与参照同域（均 v4 渲染），GT 参照行因
      自相似得分趋近 1.0，构成可达上限。
    """
    out = evaluate_noef(
        audio_path=wav_path,
        target_bpm=gt.get("tempo_bpm"),
        mode_pcs=gt.get("mode_key", "gong"),
    )
    res = {k: float(out.get(k) or 0.0) for k in T2_KEEP if k != "fad_distance"}
    # t2_fad_distance（音频级，0–1）：仅当参照与自身都真实存在时计算
    ref = gt.get("audio_path") or gt.get("ref_audio_path") or ""
    fad100 = 0.0
    if ref and os.path.exists(ref) and os.path.exists(wav_path):
        try:
            fad100 = calculate_fad_score(ref, wav_path)  # 内部返回 0–100
        except Exception:  # noqa: BLE001
            fad100 = 0.0
    res["fad_distance"] = max(0.0, min(1.0, float(fad100) / 100.0))
    return res


def _parse_midi(midi_path: str):
    toks, durs, starts, _ = parse_midi_to_sequence(midi_path)
    return list(toks), list(durs), list(starts)


def main() -> None:
    print("=" * 78)
    print("  开放续写 · 无参考消融评估 (GT=南音特征参照, 剔除点对点相似指标)")
    print("=" * 78)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(RESULTS_DIR, "comparison_openstyle_ablation.csv")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as fh:
        csv.writer(fh).writerow(CSV_HEADERS)

    all_rows: List[Dict[str, Any]] = []
    missing: List[str] = []

    for song_id in VAL_SONG_IDS:
        cn = SONG_CN.get(song_id, song_id)
        print(f"\n{'─' * 78}")
        print(f"  曲目: {cn} ({song_id})")
        gt = load_ground_truth(song_id)
        if gt is None:
            print("  ✗ GT 加载失败，跳过")
            continue

        gw = gt_wav(song_id)
        if os.path.exists(gw):
            gt["audio_path"] = gw
        mode_key = gt.get("mode_key", "gong")
        print(f"  GT: tokens={len(gt['tokens'])}, mode={mode_key}, "
              f"tempo_bpm={gt.get('tempo_bpm')}")

        # ---- GT_reference 行（真实南音在相同指标体系的得分 = 南音特征可达参照）----
        t1_gt = _gt_ceiling_tier1(song_id, gt)
        t2_gt = _run_tier2(gw, gt) if os.path.exists(gw) else {}
        row_gt = {
            "song_id": song_id, "song_name": cn,
            "architecture": "gt_reference", "architecture_label": "真实南音GT(参照)",
            "model_name": "-",
        }
        for k in T1_KEEP:
            row_gt[f"t1_{k}"] = round(t1_gt.get(k, 0.0), 2)
        row_gt["t1_composite_score"] = round(compute_openstyle_composite(t1_gt), 2)
        for k in T2_KEEP:
            row_gt[f"t2_{k}"] = round(t2_gt.get(k, 0.0), 4)
        all_rows.append(row_gt)
        print(f"  [GT参照] composite={row_gt['t1_composite_score']:.1f}")

        # ---- 架构 × 模型消融行 ----
        for arch, arch_label, midi_base, audio_base in ARCHES:
            for model_name, subdir in MODELS.items():
                midi_path = os.path.join(midi_base, subdir, f"{song_id}_{subdir}.mid")
                wav_path = os.path.join(audio_base, subdir, song_id, "ens_pipa.wav")
                if not os.path.exists(midi_path):
                    missing.append(midi_path)
                    print(f"  ⚠ 缺 MIDI: {midi_path}")
                    continue
                g_tok, g_dur, g_sta = _parse_midi(midi_path)
                if not g_tok:
                    print(f"  ⚠ 空 MIDI: {midi_path}")
                    continue
                m = evaluate_single_song(
                    g_tok, g_dur, gt,
                    gen_audio_path=wav_path if os.path.exists(wav_path) else "",
                    skip_prompt=PROMPT_LEN, gen_starts=g_sta,
                )
                t1 = {k: m.get(k, 0.0) for k in T1_KEEP}
                comp = compute_openstyle_composite(t1)
                t2 = _run_tier2(wav_path, gt) if os.path.exists(wav_path) else {}
                row = {
                    "song_id": song_id, "song_name": cn,
                    "architecture": arch, "architecture_label": arch_label,
                    "model_name": model_name,
                }
                for k in T1_KEEP:
                    row[f"t1_{k}"] = round(t1.get(k, 0.0), 2)
                row["t1_composite_score"] = round(comp, 2)
                for k in T2_KEEP:
                    row[f"t2_{k}"] = round(t2.get(k, 0.0), 4)
                all_rows.append(row)
                tag = "✔" if os.path.exists(wav_path) else "✘(无音频)"
                print(f"  [{arch_label}/{model_name}] t1_comp={comp:.1f} "
                      f"t2_stab={t2.get('pitch_stability', 0):.3f} {tag}")

    # ---- 写入 CSV ----
    with open(out_csv, "a", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        for r in all_rows:
            w.writerow([r.get(k, "") for k in CSV_HEADERS])

    # ---- 汇总报告 ----
    print("\n" + "=" * 78)
    print("  汇总（各曲均值）")
    print("=" * 78)
    groups = {
        ("gt_reference", "-"): "GT 南音参照",
        ("old_bidirectional", "BoYaTCN"): "双向原版/BoYaTCN",
        ("old_bidirectional", "MusicMamba"): "双向原版/MusicMamba",
        ("causal_autoregressive", "BoYaTCN"): "因果版/BoYaTCN",
        ("causal_autoregressive", "MusicMamba"): "因果版/MusicMamba",
    }
    print(f"{'组':<22}{'T1_comp':>9}{'t1_stab':>9}{'t1_trans':>9}{'t1_coher':>9}"
          f"{'t1_rhythm':>10}{'t1_orn':>9}{'t1_mode':>9} | "
          f"{'t2_stab':>8}{'t2_pcc':>8}{'t2_rhythm':>9}{'t2_orn':>8}"
          f"{'t2_mode':>8}{'t2_fad':>8}")
    means: Dict[Any, Dict[str, float]] = {}
    for (a, mn), _ in groups.items():
        rows = [r for r in all_rows
                if r.get("architecture") == a and r.get("model_name") == mn]
        if not rows:
            continue
        agg: Dict[str, float] = {}
        for key in ["t1_composite_score"] + [f"t1_{k}" for k in T1_KEEP] + [f"t2_{k}" for k in T2_KEEP]:
            agg[key] = float(np.mean([r.get(key, 0) or 0 for r in rows]))
        means[(a, mn)] = agg
        print(f"{groups[(a, mn)]:<22}"
              f"{agg['t1_composite_score']:>9.2f}"
              f"{agg['t1_pitch_stability']:>9.2f}{agg['t1_transition_smoothness']:>9.2f}"
              f"{agg['t1_melody_coherence']:>9.2f}{agg['t1_rhythm_consistency']:>10.2f}"
              f"{agg['t1_ornament_score']:>9.2f}{agg['t1_mode_matching_score']:>9.2f} | "
              f"{agg['t2_pitch_stability']:>8.3f}{agg['t2_pitch_contour_correlation']:>8.3f}"
              f"{agg['t2_rhythm_consistency']:>9.3f}{agg['t2_ornament_score']:>8.3f}"
              f"{agg['t2_mode_matching_score']:>8.3f}{agg['t2_fad_distance']:>8.3f}")

    # 消融 Δ（因果版 - 双向版）
    print("\n  架构消融 Δ = 因果自回归 − 双向注意力 (Tier1 composite, 各曲)")
    for mn in MODELS:
        old = [r for r in all_rows if r.get("architecture") == "old_bidirectional"
               and r.get("model_name") == mn]
        new = [r for r in all_rows if r.get("architecture") == "causal_autoregressive"
               and r.get("model_name") == mn]
        if old and new:
            o = float(np.mean([r["t1_composite_score"] for r in old]))
            n = float(np.mean([r["t1_composite_score"] for r in new]))
            print(f"    {mn:<12} 双向={o:6.2f}  因果={n:6.2f}  Δ={n - o:+6.2f}")

    if missing:
        print("\n  ⚠ 缺失产物:")
        for p in missing:
            print("   ", p)
    print(f"\n  已写出: {out_csv}")

    print("=" * 78)
    print("  说明: GT(南音参照) 行 = 真实南音主干过相同指标的自洽得分；")
    print("  指标版次 t2（2026-09-04）: 删除 t1_pitch_accuracy / t1_melody_cosine_similarity;")
    print("  t1_fad_distance → t2_fad_distance; 新增 t1 transition_smoothness/melody_coherence")
    print("  与 t2 pitch_contour_correlation(11 首南音 f0 画像相似度, 画像缓存见 output/)。")
    print("=" * 78)


if __name__ == "__main__":
    main()
