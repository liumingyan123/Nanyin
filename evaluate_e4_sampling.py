# -*- coding: utf-8 -*-
"""
evaluate_e4_sampling.py — E4 采样配置的批量Tier-1评估
=====================================================

输入：E4 网格生成的**原始三路 token 序列**（results/e4/e4_events.jsonl，由
      generate_music.py --mode sampling_search 落盘）+ 同窗口 GT。
输出：
  1) results/e4/e4_eval_rows.csv       每条生成序列一行（9 项 NOEF + composite + 非法率
                                        + 重复率 + 46 维特征）
  2) results/e4/e4_eval_summary.csv    每个 (T, top-p) 配置一行的聚合（mean/std）
  3) results/e4/e4_eval_summary.json   同上 + GT 天花板 + 热力图矩阵 + 最优配置选择
  4) reports/e4/e4_heatmap_composite.png / e4_heatmap_illegal.png / e4_heatmap_diversity.png

⚠ E4 硬性约束（本脚本不得违反）：
  - 不做任何训练/微调/权重修改；
  - **评估使用模型原生输出**：不开启任何 MIDI 规则后处理、不做时值钳位修正；
  - 非法 token（pitch >= 128 的特殊 token、duration/shift 越界或 PAD）如实计数，不清洗；
  - 所有指标基于真实推理输出，不编造。

运行：
  venv/Scripts/python.exe evaluate_e4_sampling.py --config configs/e4_sampling_search.yaml
"""

import os
import csv
import json
import sys
import argparse
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from dataset_nanyin import CompoundEventCodec, get_compound_codec  # noqa: E402
from generate_music import load_val_windows                        # noqa: E402
from nanyin_config import load_config                             # noqa: E402
from nanyin_metrics import (                                      # noqa: E402
    evaluate_single_song, compute_composite_score, resolve_mode_pcs,
)
from symbolic_features import (                                   # noqa: E402
    FEATURE_DIM, FEATURE_NAMES, extract_symbolic_features, pairwise_distance_matrix,
)

E4_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "configs", "e4_sampling_search.yaml")
E5_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "configs", "e5_model_drift.yaml")

# 窗口 GT 的调式 pitch-class 集合缓存：{(song_id, window_index): List[int]}
_MODE_PCS_CACHE: Dict[Tuple[str, int], List[int]] = {}


# ------------------------------------------------------------------------------
# 工具
# ------------------------------------------------------------------------------

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[E4-eval] 原始事件文件不存在: {path}")
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _cumulative_starts(shift_ids: Sequence[int], codec: CompoundEventCodec) -> List[float]:
    """由前向 IOI(shift) 累加得到 onset：onset_{i+1} = onset_i + index_to_shift(shift_i)。"""
    starts: List[float] = [0.0]
    t = 0.0
    for s in shift_ids[:-1]:
        t += codec.index_to_shift(int(s))
        starts.append(t)
    if len(starts) < len(shift_ids):        # 极端兜底
        starts += [starts[-1]] * (len(shift_ids) - len(starts))
    return starts


def _mode_key_of_tokens(tokens: Sequence[int]) -> str:
    """窗口 GT 的调式键名（与 nanyin_metrics.load_ground_truth 同源规则）。"""
    from nanyin_metrics import _estimate_mode_from_pitch  # noqa: PLC0415（私有函数，口径同源）
    return _estimate_mode_from_pitch([float(t) for t in tokens])[0]


def _ngram_repetition_rate(tokens: Sequence[int], n: int = 4) -> float:
    """n-gram 重复率 = 1 - 不同 n-gram 数 / n-gram 总数（0 = 完全无重复）。"""
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - len(set(grams)) / float(len(grams))


def _illegal_stats(
    pitch_ids: Sequence[int],
    duration_ids: Sequence[int],
    shift_ids: Sequence[int],
    icfg: Dict[str, Any],
) -> Dict[str, float]:
    """非法事件统计（**评估区域**；最后一个事件的 shift 是伪标签，不计入）。

    非法定义：
      pitch    : id >= pitch_special_threshold（128）→ BOS/EOS 特殊 token
      duration : id >= duration_valid_max（402，含 EOS/BOS）或 id == pad_value（0）
      shift    : id >= shift_valid_max（1002）或 id == pad_value（0）
    """
    p_thr = int(icfg["pitch_special_threshold"])
    d_max = int(icfg["duration_valid_max"])
    s_max = int(icfg["shift_valid_max"])
    pad = int(icfg["pad_value"])

    p = np.asarray(pitch_ids, dtype=np.int64)
    d = np.asarray(duration_ids, dtype=np.int64)
    s = np.asarray(shift_ids[:-1], dtype=np.int64)      # 去掉伪标签位

    ill_p = int(np.sum(p >= p_thr))
    ill_d = int(np.sum((d >= d_max) | (d == pad)))
    ill_s = int(np.sum((s >= s_max) | (s == pad)))
    total = float(len(p))
    return {
        "illegal_pitch_count": ill_p,
        "illegal_duration_count": ill_d,
        "illegal_shift_count": ill_s,
        "illegal_pitch_rate": ill_p / total,
        "illegal_duration_rate": ill_d / total,
        "illegal_shift_rate": ill_s / float(len(s)) if len(s) > 0 else 0.0,
        # 综合非法率：三路非法事件数 / 三路可统计总位数
        "illegal_total_rate": (ill_p + ill_d + ill_s) / (2.0 * total + float(len(s))),
    }


# ------------------------------------------------------------------------------
# GT 窗口构造
# ------------------------------------------------------------------------------

def build_window_gt(cfg: dict) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """返回 {(song_id, window_index): 窗口 GT 字典}，字典结构兼容 evaluate_single_song。"""
    codec = get_compound_codec(cfg)
    gt_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for win in load_val_windows(cfg, verbose=False):
        tokens = list(win["pitch_ids"])
        durations = [codec.index_to_duration(int(i)) for i in win["duration_ids"]]
        starts = _cumulative_starts(win["shift_ids"], codec)
        mode_key = _mode_key_of_tokens(tokens)
        gt_map[(win["song_id"], win["window_index"])] = {
            "tokens": tokens,
            "durations": durations,
            "starts": starts,
            "pitches": [float(t) for t in tokens],
            "mode_key": mode_key,
            "mode_name": mode_key,
            "audio_path": "",          # 不做音频级 FAD（E4 只做 Tier-1 符号级）
            "num_events": len(tokens),
        }
    return gt_map


# ------------------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------------------

def evaluate_all(cfg: dict) -> Dict[str, Any]:
    outcfg = cfg["outputs"]
    ecfg = cfg["evaluation"]
    skip_prompt = int(ecfg["skip_prompt"])
    ngram = int(ecfg["repetition"]["ngram"])
    codec = get_compound_codec(cfg)

    rows_csv = os.path.join(_SCRIPT_DIR, str(outcfg["eval_rows_csv"]))
    summary_csv = os.path.join(_SCRIPT_DIR, str(outcfg["eval_summary_csv"]))
    summary_json = os.path.join(_SCRIPT_DIR, str(outcfg["eval_summary_json"]))
    for p in (rows_csv, summary_csv, summary_json):
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)

    events_path = os.path.join(_SCRIPT_DIR, str(outcfg["events_jsonl"]))
    records = _read_jsonl(events_path)
    gt_map = build_window_gt(cfg)
    # 每个窗口的调式 pitch-class 集合（供 46 维特征的调式分量使用）
    _MODE_PCS_CACHE.clear()
    for key, gt in gt_map.items():
        _MODE_PCS_CACHE[key] = resolve_mode_pcs(gt["mode_key"])
    print(f"[E4-eval] 原始序列 {len(records)} 条；窗口 GT {len(gt_map)} 个")

    # ---- GT 天花板（GT vs 自身，不跳过 prompt）----
    gt_ceiling: List[Dict[str, Any]] = []
    for key, gt in gt_map.items():
        m = evaluate_single_song(
            gt["tokens"], gt["durations"], gt,
            skip_prompt=int(ecfg["gt_ceiling_skip_prompt"]),
            gen_starts=gt["starts"],
        )
        m["composite"] = compute_composite_score(m)
        gt_ceiling.append({"song_id": key[0], "window_index": key[1], **m})
    gt_ceiling_vals = [r["composite"] for r in gt_ceiling]

    # ---- 逐条评估 ----
    rows: List[Dict[str, Any]] = []
    for rec in records:
        key = (rec["song_id"], int(rec["window_index"]))
        gt = gt_map[key]
        p_ids = list(rec["pitch_ids"])
        d_ids = list(rec["duration_ids"])
        s_ids = list(rec["shift_ids"])

        gen_tokens_full = [int(x) for x in p_ids]
        gen_durations_full = [codec.index_to_duration(int(x)) for x in d_ids]
        gen_starts_full = _cumulative_starts(s_ids, codec)

        # 原生输出直评：**不做任何后处理 / 钳位**
        metrics = evaluate_single_song(
            gen_tokens_full, gen_durations_full, gt,
            skip_prompt=skip_prompt, gen_starts=gen_starts_full,
        )
        composite = compute_composite_score(metrics)

        # 非法统计 / 重复率：评估区域 = 续写的 240 个事件
        region_p = p_ids[skip_prompt:]
        region_d = d_ids[skip_prompt:]
        region_s = s_ids[skip_prompt:]
        ill = _illegal_stats(region_p, region_d, region_s, ecfg["illegal"])
        rep = _ngram_repetition_rate(region_p, ngram)

        # 46 维特征（评估区域；调式分量以该窗口 GT 调式为参照）
        region_starts = gen_starts_full[skip_prompt:]
        feat = extract_symbolic_features(
            [int(x) for x in region_p],
            [codec.index_to_duration(int(x)) for x in region_d],
            region_starts,
            mode_pcs=_MODE_PCS_CACHE[key],
        )

        row = {
            "song_id": rec["song_id"],
            "window_index": int(rec["window_index"]),
            "model_seed": int(rec["model_seed"]),
            "temperature": float(rec["temperature"]),
            "top_p": float(rec["top_p"]),
            "rng_seed": int(rec["rng_seed"]),
            "composite": composite,
            "pitch_accuracy": metrics.get("pitch_accuracy", 0.0),
            "pitch_stability": metrics.get("pitch_stability", 0.0),
            "transition_smoothness": metrics.get("transition_smoothness", 0.0),
            "melody_coherence": metrics.get("melody_coherence", 0.0),
            "rhythm_consistency": metrics.get("rhythm_consistency", 0.0),
            "tempo_accuracy": metrics.get("tempo_accuracy", 0.0),
            "ornament_score": metrics.get("ornament_score", 0.0),
            "ornament_density": metrics.get("ornament_density", 0.0),
            "mode_matching_score": metrics.get("mode_matching_score", 0.0),
            "melody_cosine_similarity": metrics.get("melody_cosine_similarity", 0.0),
            **ill,
            "repetition_rate": rep,
            "midi_path": rec.get("midi_path", ""),
        }
        for i, v in enumerate(feat):
            row[f"feat_{i:02d}"] = float(v)
        rows.append(row)

    # ---- 写明细 CSV ----
    feat_cols = [f"feat_{i:02d}" for i in range(FEATURE_DIM)]
    base_cols = [
        "song_id", "window_index", "model_seed", "temperature", "top_p", "rng_seed",
        "composite", "pitch_accuracy", "pitch_stability", "transition_smoothness",
        "melody_coherence", "rhythm_consistency", "tempo_accuracy",
        "ornament_score", "ornament_density", "mode_matching_score",
        "melody_cosine_similarity",
        "illegal_pitch_count", "illegal_duration_count", "illegal_shift_count",
        "illegal_pitch_rate", "illegal_duration_rate", "illegal_shift_rate",
        "illegal_total_rate", "repetition_rate", "midi_path",
    ]
    with open(rows_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=base_cols + feat_cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- 按 (T, top-p) 聚合 ----
    temps = sorted({float(r["temperature"]) for r in rows})
    topps = sorted({float(r["top_p"]) for r in rows})
    metric_cols = [c for c in base_cols if c not in (
        "song_id", "window_index", "model_seed", "temperature", "top_p", "rng_seed", "midi_path",
    )]

    summary_rows: List[Dict[str, Any]] = []
    for t in temps:
        for p in topps:
            sub = [r for r in rows if r["temperature"] == t and r["top_p"] == p]
            if not sub:
                continue
            agg: Dict[str, Any] = {"temperature": t, "top_p": p, "n_samples": len(sub)}
            for c in metric_cols:
                vals = np.asarray([r[c] for r in sub], dtype=np.float64)
                agg[f"{c}_mean"] = float(vals.mean())
                agg[f"{c}_std"] = float(vals.std(ddof=0))
            summary_rows.append(agg)

    # ---- 多样性：同一窗口 + 同一模型种子下，16 个采样参数的 46 维特征两两距离 ----
    feats = {i: np.asarray([r[f"feat_{j:02d}"] for j in range(FEATURE_DIM)], dtype=np.float64)
             for i, r in enumerate(rows)}
    groups: Dict[Tuple[str, int, int], List[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault((r["song_id"], r["window_index"], r["model_seed"]), []).append(i)

    div_sum: Dict[Tuple[float, float], List[float]] = {}
    gt_dist_sum: Dict[Tuple[float, float], List[float]] = {}
    seed_div_sum: Dict[Tuple[float, float], List[float]] = {}
    # 同配置**跨训练种子**的多样性（同一窗口、同一采样参数，换随机种子能产生多少变异）
    seed_groups: Dict[Tuple[str, int, float, float], List[int]] = {}
    for i, r in enumerate(rows):
        seed_groups.setdefault(
            (r["song_id"], r["window_index"], r["temperature"], r["top_p"]), []
        ).append(i)
    for _, idxs in seed_groups.items():
        if len(idxs) < 2:
            continue
        mat = pairwise_distance_matrix([feats[i] for i in idxs])
        vals = [mat[a, b] for a in range(len(idxs)) for b in range(a + 1, len(idxs))]
        k0 = (rows[idxs[0]]["temperature"], rows[idxs[0]]["top_p"])
        seed_div_sum.setdefault(k0, []).append(float(np.mean(vals)))

    for key, idxs in groups.items():
        mat = pairwise_distance_matrix([feats[i] for i in idxs])
        n = len(idxs)
        for a in range(n):
            ra = rows[idxs[a]]
            k = (ra["temperature"], ra["top_p"])
            others = [mat[a, b] for b in range(n) if b != a]
            if others:
                div_sum.setdefault(k, []).append(float(np.mean(others)))

    # 到同窗口 GT 46 维特征的距离（越小越接近 GT 分布，仅作参考）
    gt_feat_cache: Dict[Tuple[str, int], np.ndarray] = {}
    for i, r in enumerate(rows):
        key = (r["song_id"], r["window_index"])
        if key not in gt_feat_cache:
            gt = gt_map[key]
            sp = int(ecfg["skip_prompt"])
            gt_feat_cache[key] = extract_symbolic_features(
                gt["tokens"][sp:], gt["durations"][sp:], gt["starts"][sp:],
                mode_pcs=_MODE_PCS_CACHE[key],
            )
        d = float(np.linalg.norm(feats[i] - gt_feat_cache[key]))
        gt_dist_sum.setdefault((r["temperature"], r["top_p"]), []).append(d)

    for agg in summary_rows:
        k = (agg["temperature"], agg["top_p"])
        agg["diversity_inter_config_mean"] = float(np.mean(div_sum[k])) if k in div_sum else 0.0
        agg["diversity_within_seed_mean"] = float(np.mean(seed_div_sum[k])) if k in seed_div_sum else 0.0
        agg["gt_feature_distance_mean"] = float(np.mean(gt_dist_sum[k])) if k in gt_dist_sum else 0.0

    summary_cols = ["temperature", "top_p", "n_samples"] + [
        f"{c}_{m}" for c in metric_cols for m in ("mean", "std")
    ] + ["diversity_inter_config_mean", "diversity_within_seed_mean", "gt_feature_distance_mean"]
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=summary_cols)
        w.writeheader()
        for agg in summary_rows:
            w.writerow(agg)

    result = {
        "config_paths": {
            "rows_csv": rows_csv, "summary_csv": summary_csv, "summary_json": summary_json,
        },
        "num_records": len(records),
        "temps": temps, "topps": topps,
        "gt_ceiling": {
            "composite_mean": float(np.mean(gt_ceiling_vals)),
            "composite_std": float(np.std(gt_ceiling_vals)),
            "per_window": gt_ceiling,
        },
        "summary": summary_rows,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n[E4-eval] GT 天花板 composite = {np.mean(gt_ceiling_vals):.2f} ± {np.std(gt_ceiling_vals):.2f}")
    print(f"[E4-eval] 明细: {rows_csv}")
    print(f"[E4-eval] 聚合: {summary_csv}")
    return result


# ==============================================================================
# E5：任务A/B/C（模型公平对比 × 自回归漂移 × 46 维降维）
# ==============================================================================
# ⚠ 说明：以下为 E5 新增段落，E4 的 `evaluate_all` / `_heatmap` / 主流程 e4 分支
#   逻辑逐字未改（通过 `--stage` 选择分支），保证 E4 可复现性不受影响。
# ⚠ E5 硬性约束：评估使用模型**原生输出**（无后处理 / 无钳位）；不训练不改权重；
#   所有指标基于真实推理输出，异常现象如实记录。
# ==============================================================================

def _gt_region_features(gt: Dict[str, Any], skip_prompt: int, target_len: int,
                        mode_pcs: Sequence[int]) -> np.ndarray:
    """GT 的**等长评估区域** 46 维特征：tokens[skip_prompt:target_len]。

    与生成片段同起点、同长度 → 不同续写长度下的 gtDist 可比。
    """
    end = int(target_len)
    return extract_symbolic_features(
        gt["tokens"][skip_prompt:end], gt["durations"][skip_prompt:end],
        gt["starts"][skip_prompt:end], mode_pcs=mode_pcs,
    )


def _eval_one_record(
    rec: Dict[str, Any], gt: Dict[str, Any], cfg: dict, codec: CompoundEventCodec,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """对一条 E5 生成记录做 Tier-1 评估，返回 (row, feat46)。"""
    ecfg = cfg["evaluation"]
    skip_prompt = int(ecfg["skip_prompt"])
    ngram = int(ecfg["repetition"]["ngram"])

    p_ids = list(rec["pitch_ids"])
    d_ids = list(rec["duration_ids"])
    s_ids = list(rec["shift_ids"])

    gen_tokens = [int(x) for x in p_ids]
    gen_durations = [codec.index_to_duration(int(x)) for x in d_ids]
    gen_starts = _cumulative_starts(s_ids, codec)

    # 原生输出直评：**不做任何后处理 / 钳位**
    metrics = evaluate_single_song(
        gen_tokens, gen_durations, gt,
        skip_prompt=skip_prompt, gen_starts=gen_starts,
    )
    composite = compute_composite_score(metrics)

    ill = _illegal_stats(p_ids[skip_prompt:], d_ids[skip_prompt:], s_ids[skip_prompt:],
                         ecfg["illegal"])
    rep = _ngram_repetition_rate(p_ids[skip_prompt:], ngram)

    feat = extract_symbolic_features(
        [int(x) for x in p_ids[skip_prompt:]],
        [codec.index_to_duration(int(x)) for x in d_ids[skip_prompt:]],
        gen_starts[skip_prompt:],
        mode_pcs=_MODE_PCS_CACHE[(rec["song_id"], int(rec["window_index"]))],
    )

    row = {
        "song_id": rec["song_id"],
        "window_index": int(rec["window_index"]),
        "model_name": rec["model_name"],
        "model_seed": int(rec["model_seed"]),
        "target_len": int(rec["target_len"]),
        "rng_seed": int(rec.get("rng_seed", -1)),
        "composite": composite,
        "pitch_accuracy": metrics.get("pitch_accuracy", 0.0),
        "pitch_stability": metrics.get("pitch_stability", 0.0),
        "transition_smoothness": metrics.get("transition_smoothness", 0.0),
        "melody_coherence": metrics.get("melody_coherence", 0.0),
        "rhythm_consistency": metrics.get("rhythm_consistency", 0.0),
        "tempo_accuracy": metrics.get("tempo_accuracy", 0.0),
        "ornament_score": metrics.get("ornament_score", 0.0),
        "ornament_density": metrics.get("ornament_density", 0.0),
        "mode_matching_score": metrics.get("mode_matching_score", 0.0),
        "melody_cosine_similarity": metrics.get("melody_cosine_similarity", 0.0),
        **ill,
        "repetition_rate": rep,
        "midi_path": rec.get("midi_path", ""),
    }
    return row, feat


def evaluate_all_e5(cfg: dict) -> Dict[str, Any]:
    """E5 主评估：逐条 Tier-1 → 明细 CSV → (模型 × 续写长度) 聚合 → 任务C 样本集合。

    返回 dict（含内存中的 `_samples` 供降维使用，写 JSON 前会剔除）。
    """
    outcfg = cfg["outputs"]
    ecfg = cfg["evaluation"]
    skip_prompt = int(ecfg["skip_prompt"])
    codec = get_compound_codec(cfg)

    rows_csv = os.path.join(_SCRIPT_DIR, str(outcfg["eval_rows_csv"]))
    summary_csv = os.path.join(_SCRIPT_DIR, str(outcfg["eval_summary_csv"]))
    summary_json = os.path.join(_SCRIPT_DIR, str(outcfg["eval_summary_json"]))
    for p in (rows_csv, summary_csv, summary_json):
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)

    events_path = os.path.join(_SCRIPT_DIR, str(outcfg["events_jsonl"]))
    records = _read_jsonl(events_path)
    gt_map = build_window_gt(cfg)

    _MODE_PCS_CACHE.clear()
    for key, gt in gt_map.items():
        _MODE_PCS_CACHE[key] = resolve_mode_pcs(gt["mode_key"])
    print(f"[E5-eval] 原始序列 {len(records)} 条；窗口 GT {len(gt_map)} 个")

    # ---- GT 天花板（GT vs 自身，不跳过 prompt）----
    gt_ceiling = []
    for key, gt in gt_map.items():
        m = evaluate_single_song(
            gt["tokens"], gt["durations"], gt,
            skip_prompt=int(ecfg["gt_ceiling_skip_prompt"]), gen_starts=gt["starts"],
        )
        m["composite"] = compute_composite_score(m)
        gt_ceiling.append({"song_id": key[0], "window_index": key[1], **m})
    gt_ceiling_vals = [r["composite"] for r in gt_ceiling]

    # ---- 逐条评估 ----
    rows: List[Dict[str, Any]] = []
    feats: List[np.ndarray] = []
    for rec in records:
        key = (rec["song_id"], int(rec["window_index"]))
        gt = gt_map[key]
        row, feat = _eval_one_record(rec, gt, cfg, codec)
        # gtDist：与生成片段**等长、同起点**的 GT 区域
        gt_feat = _gt_region_features(gt, skip_prompt, int(rec["target_len"]),
                                      _MODE_PCS_CACHE[key])
        row["gt_feature_distance"] = float(np.linalg.norm(feat - gt_feat))
        rows.append(row)
        feats.append(feat)

    print(f"[E5-eval] GT 天花板 composite = {np.mean(gt_ceiling_vals):.2f} "
          f"± {np.std(gt_ceiling_vals):.2f}")

    # ---- 明细 CSV ----
    feat_cols = [f"feat_{i:02d}" for i in range(FEATURE_DIM)]
    base_cols = [
        "song_id", "window_index", "model_name", "model_seed", "target_len", "rng_seed",
        "composite", "pitch_accuracy", "pitch_stability", "transition_smoothness",
        "melody_coherence", "rhythm_consistency", "tempo_accuracy",
        "ornament_score", "ornament_density", "mode_matching_score",
        "melody_cosine_similarity",
        "illegal_pitch_count", "illegal_duration_count", "illegal_shift_count",
        "illegal_pitch_rate", "illegal_duration_rate", "illegal_shift_rate",
        "illegal_total_rate", "repetition_rate", "gt_feature_distance", "midi_path",
    ]
    for r, feat in zip(rows, feats):
        for i, v in enumerate(feat):
            r[f"feat_{i:02d}"] = float(v)
    with open(rows_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=base_cols + feat_cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[E5-eval] 明细: {rows_csv}")

    # ---- 按 (模型, 续写长度) 聚合 ----
    model_names = sorted({r["model_name"] for r in rows})
    lens = sorted({int(r["target_len"]) for r in rows})
    metric_cols = [c for c in base_cols if c not in (
        "song_id", "window_index", "model_name", "model_seed", "target_len",
        "rng_seed", "midi_path",
    )]

    summary_rows: List[Dict[str, Any]] = []
    for m in model_names:
        for L in lens:
            sub = [r for r in rows if r["model_name"] == m and int(r["target_len"]) == L]
            if not sub:
                continue
            agg: Dict[str, Any] = {"model_name": m, "target_len": L, "n_samples": len(sub)}
            for c in metric_cols:
                vals = np.asarray([r[c] for r in sub], dtype=np.float64)
                agg[f"{c}_mean"] = float(vals.mean())
                agg[f"{c}_std"] = float(vals.std(ddof=0))
            # 跨训练种子多样性：同 (模型, 长度, 窗口) 下 3 个种子的 46 维特征两两距离
            seed_div: List[float] = []
            for r_i, row_i in enumerate(rows):
                if row_i["model_name"] != m or int(row_i["target_len"]) != L:
                    continue
                peers = [j for j, r_j in enumerate(rows)
                         if j != r_i and r_j["model_name"] == m
                         and int(r_j["target_len"]) == L
                         and r_j["song_id"] == row_i["song_id"]
                         and int(r_j["window_index"]) == int(row_i["window_index"])]
                if peers:
                    mat = pairwise_distance_matrix([feats[r_i]] + [feats[j] for j in peers])
                    seed_div.append(float(np.mean(mat[0, 1:])))
            agg["cross_seed_diversity_mean"] = float(np.mean(seed_div)) if seed_div else 0.0
            summary_rows.append(agg)

    summary_cols = ["model_name", "target_len", "n_samples"] + [
        f"{c}_{k}" for c in metric_cols for k in ("mean", "std")
    ] + ["cross_seed_diversity_mean"]
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=summary_cols)
        w.writeheader()
        for agg in summary_rows:
            w.writerow(agg)
    print(f"[E5-eval] 聚合: {summary_csv}")

    # ---- 任务C 降维样本集合（仅续写片段的 46 维特征）----
    samples: List[Dict[str, Any]] = []
    for r, feat in zip(rows, feats):
        samples.append({
            "feat": feat, "source": r["model_name"], "group": str(int(r["target_len"])),
            "seed": int(r["model_seed"]), "song_id": r["song_id"],
            "window_index": int(r["window_index"]),
        })
    for key, gt in gt_map.items():
        for L in lens:
            samples.append({
                "feat": _gt_region_features(gt, skip_prompt, int(L), _MODE_PCS_CACHE[key]),
                "source": "GT", "group": str(int(L)), "seed": -1,
                "song_id": key[0], "window_index": key[1],
            })

    result: Dict[str, Any] = {
        "config_paths": {
            "rows_csv": rows_csv, "summary_csv": summary_csv, "summary_json": summary_json,
        },
        "num_records": len(records),
        "models": model_names,
        "lengths": lens,
        "gt_ceiling": {
            "composite_mean": float(np.mean(gt_ceiling_vals)),
            "composite_std": float(np.std(gt_ceiling_vals)),
            "per_window": gt_ceiling,
        },
        "summary": summary_rows,
        "_samples": samples,      # 内存数据，写 JSON 前剔除
    }
    return result


def _plot_drift_lines(result: Dict[str, Any], cfg: dict) -> None:
    """任务B：随续写长度 32→64→128→256 的指标漂移折线图（4 面板）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = cfg["outputs"]
    lens = [int(x) for x in result["lengths"]]
    models = list(result["models"])
    rows = result["summary"]

    specs = [
        ("composite_mean", "composite_std", "Composite score (Tier-1)"),
        ("gt_feature_distance_mean", "gt_feature_distance_std",
         "46-dim feature distance to same-length GT"),
        ("repetition_rate_mean", "repetition_rate_std", "4-gram repetition rate"),
        ("illegal_total_rate_mean", "illegal_total_rate_std", "Illegal token rate"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), dpi=160)
    for ax, (mkey, skey, title) in zip(axes.ravel(), specs):
        for model_name in models:
            xs, ys, es = [], [], []
            for L in lens:
                hit = [r for r in rows if r["model_name"] == model_name
                       and int(r["target_len"]) == L]
                if not hit:
                    continue
                xs.append(L)
                ys.append(float(hit[0][mkey]))
                es.append(float(hit[0][skey]))
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=model_name)
        ax.set_xscale("log", base=2)
        ax.set_xticks(lens, [str(l) for l in lens])
        ax.set_xlabel("target sequence length (events, incl. 16 prompt)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("E5 Task B - autoregressive drift vs continuation length "
                 "(mean +/- std over 3 seeds x 10 val windows)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(_SCRIPT_DIR, str(out["drift_line_png"]))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"[E5-eval] 漂移折线图: {path}")


def _cluster_info(samples: List[Dict[str, Any]], xy: np.ndarray,
                  tag: str) -> List[Dict[str, Any]]:
    """各类簇中心 + 到**同长度 GT 簇中心**的距离（降维空间，仅作定性参考）。"""
    info: List[Dict[str, Any]] = []
    gt_center: Dict[str, np.ndarray] = {}
    for grp in sorted({s["group"] for s in samples}):
        idx = [i for i, s in enumerate(samples) if s["source"] == "GT" and s["group"] == grp]
        if idx:
            gt_center[grp] = xy[idx].mean(axis=0)
    for source in sorted({s["source"] for s in samples if s["source"] != "GT"}):
        for grp in sorted({s["group"] for s in samples if s["source"] == source}):
            idx = [i for i, s in enumerate(samples)
                   if s["source"] == source and s["group"] == grp]
            if not idx:
                continue
            center = xy[idx].mean(axis=0)
            dist = (float(np.linalg.norm(center - gt_center[grp]))
                    if grp in gt_center else float("nan"))
            spread = float(np.mean(np.linalg.norm(xy[idx] - center, axis=1)))
            info.append({
                "method": tag, "source": source, "length": int(grp), "n_samples": len(idx),
                "center_x": float(center[0]), "center_y": float(center[1]),
                "dist_to_gt_center": dist, "within_cluster_spread": spread,
            })
    return info


def _plot_reduction(result: Dict[str, Any], cfg: dict) -> List[Dict[str, Any]]:
    """任务C：46 维符号特征 → StandardScaler → UMAP / t-SNE。

    超参严格按 exp3 口径：UMAP(n_neighbors=8, min_dist=0.1, random_state=42)、
    t-SNE(perplexity=5, max_iter=2000, init=pca, random_state=42)。
    样本：10 个验证窗口 GT（同起点等长截取）+ 两组模型 × 3 种子 × 4 长度 × 10 窗口。
    ⚠ 禁止混入 exp3 旧 NOEF 单 token 样本（事件表示不同，特征空间不兼容）。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    from umap import UMAP

    out = cfg["outputs"]
    rcfg = cfg["reduction"]
    samples: List[Dict[str, Any]] = result["_samples"]
    X = np.asarray([np.asarray(s["feat"], dtype=np.float64) for s in samples])
    Xs = StandardScaler().fit_transform(X)
    print(f"[E5-eval] 任务C 样本数 = {len(samples)}（含同起点等长 GT 区域）")

    # 视觉编码：**颜色 = 续写长度**，**形状 = 模型/来源**（GT 用星形并加黑边区分）
    groups = sorted({s["group"] for s in samples}, key=lambda g: int(g))
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3"]
    colors = {g: palette[i % len(palette)] for i, g in enumerate(groups)}
    markers = {"GT": "*", "boyatcn": "o", "musicmamba": "^"}
    sizes = {"GT": 130, "boyatcn": 32, "musicmamba": 38}
    sources = ("GT", "boyatcn", "musicmamba")
    u = rcfg["umap"]
    t = rcfg["tsne"]

    uu = UMAP(n_neighbors=int(u["n_neighbors"]), min_dist=float(u["min_dist"]),
              n_components=2, random_state=int(u["random_state"]), n_jobs=1)
    xy_u = uu.fit_transform(Xs)
    tt = TSNE(n_components=2, perplexity=float(t["perplexity"]),
              max_iter=int(t["max_iter"]), init=str(t["init"]),
              random_state=int(t["random_state"]))
    xy_t = tt.fit_transform(Xs)

    def _draw(xy: np.ndarray, title: str, path: str) -> None:
        from matplotlib.lines import Line2D

        fig, ax = plt.subplots(figsize=(8.6, 6.4), dpi=160)
        for source in sources:
            for grp in groups:
                idx = [i for i, s in enumerate(samples)
                       if s["source"] == source and s["group"] == grp]
                if not idx:
                    continue
                ax.scatter(xy[idx, 0], xy[idx, 1],
                           s=sizes[source], c=colors[grp], marker=markers[source],
                           alpha=0.85 if source != "GT" else 0.95,
                           edgecolors="black" if source == "GT" else "white",
                           linewidths=0.5, zorder=3 if source == "GT" else 2)
        # 双图例：颜色=续写长度，形状=来源
        h_color = [Line2D([], [], marker="o", ls="", color=colors[g], markersize=8,
                          label=f"len={g}") for g in groups]
        h_marker = [Line2D([], [], marker=markers[s], ls="", color="#333333",
                           markersize=11 if s == "GT" else 8,
                           label=("GT (ground truth)" if s == "GT" else s)) for s in sources]
        leg1 = ax.legend(handles=h_color, title="Length (color)", fontsize=8,
                         title_fontsize=8, loc="upper left", framealpha=0.9)
        ax.add_artist(leg1)
        ax.legend(handles=h_marker, title="Source (marker)", fontsize=8,
                  title_fontsize=8, loc="lower left", framealpha=0.9)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"[E5-eval] 降维图: {path}")

    umap_png = os.path.join(_SCRIPT_DIR, str(out["umap_png"]))
    tsne_png = os.path.join(_SCRIPT_DIR, str(out["tsne_png"]))
    _draw(xy_u, "UMAP of 46-dim features (color = length, marker = source)",
          umap_png)
    _draw(xy_t, "t-SNE of 46-dim features (color = length, marker = source)",
          tsne_png)

    return _cluster_info(samples, xy_u, "umap") + _cluster_info(samples, xy_t, "tsne")


def _heatmap(result: Dict[str, Any], cfg: dict) -> None:
    """绘制三张热力图：composite / 非法率 / 多样性（行=temperature，列=top-p）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    temps = result["temps"]
    topps = result["topps"]
    rows = result["summary"]
    out = cfg["outputs"]
    os.makedirs(os.path.join(_SCRIPT_DIR, "reports", "e4"), exist_ok=True)

    def _matrix(key: str) -> np.ndarray:
        m = np.zeros((len(temps), len(topps)), dtype=np.float64)
        for i, t in enumerate(temps):
            for j, p in enumerate(topps):
                hit = [r for r in rows if r["temperature"] == t and r["top_p"] == p]
                m[i, j] = float(hit[0][key]) if hit else np.nan
        return m

    specs = [
        ("composite_mean", "Composite score", out["heatmap_composite_png"], "viridis"),
        ("illegal_total_rate_mean", "Illegal event rate", out["heatmap_illegal_png"], "magma"),
        ("diversity_inter_config_mean", "Diversity (46-dim pairwise dist)",
         out["heatmap_diversity_png"], "cividis"),
    ]
    for key, title, rel_path, cmap in specs:
        mat = _matrix(key)
        fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=160)
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(topps)), [f"{p:g}" for p in topps])
        ax.set_yticks(range(len(temps)), [f"{t:g}" for t in temps])
        ax.set_xlabel("top-p")
        ax.set_ylabel("temperature")
        ax.set_title(f"E4 {title} (mean over 3 seeds × 10 windows)")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                        fontsize=8, color="white" if np.nanmax(mat) > 0 and
                        (mat[i, j] - np.nanmin(mat)) / max(1e-9, (np.nanmax(mat) - np.nanmin(mat))) > 0.5
                        else "black")
        fig.colorbar(im, ax=ax, shrink=0.85)
        fig.tight_layout()
        path = os.path.join(_SCRIPT_DIR, str(rel_path))
        fig.savefig(path)
        plt.close(fig)
        print(f"[E4-eval] 热力图: {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="E4 / E5 批量 Tier-1 评估")
    ap.add_argument("--stage", default="e4", choices=["e4", "e5"],
                    help="e4 = E4 采样网格评估（默认）；e5 = E5 模型对比与自回归漂移")
    ap.add_argument("--config", default=None, help="配置 YAML（默认按 stage 选择）")
    ap.add_argument("--no-plot", action="store_true", help="跳过绘图")
    args = ap.parse_args()

    cfg = load_config(args.config or (E4_CONFIG_PATH if args.stage == "e4" else E5_CONFIG_PATH))

    if args.stage == "e5":
        result = evaluate_all_e5(cfg)
        if not args.no_plot:
            _plot_drift_lines(result, cfg)
            result["cluster_distance"] = _plot_reduction(result, cfg)
        result.pop("_samples", None)          # 内存特征矩阵不落盘
        json_path = os.path.join(_SCRIPT_DIR, str(cfg["outputs"]["eval_summary_json"]))
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[E5-eval] 聚合 JSON: {json_path}")
        return 0

    result = evaluate_all(cfg)
    if not args.no_plot:
        _heatmap(result, cfg)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # pyright: ignore
    raise SystemExit(main())
