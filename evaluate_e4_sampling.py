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
    ap = argparse.ArgumentParser(description="E4 采样配置批量评估")
    ap.add_argument("--config", default=E4_CONFIG_PATH, help="E4 配置 YAML")
    ap.add_argument("--no-plot", action="store_true", help="跳过热力图绘制")
    args = ap.parse_args()

    cfg = load_config(args.config)
    result = evaluate_all(cfg)
    if not args.no_plot:
        _heatmap(result, cfg)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # pyright: ignore
    raise SystemExit(main())
