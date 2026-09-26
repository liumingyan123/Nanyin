#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E5 补充校验：滑窗粒度 46 维符号特征降维可视化（后处理，只读）。

分支：feature/e5-fullsong-vis_check（从 experiment/E5_model_compare_drift HEAD 新建）
用途：在**256 事件非重叠滑窗粒度**上补充定性校验图，与主实验窗口粒度任务C 互为参照。

【本脚本的硬性边界】
1. 不训练、不微调、不重新推理生成：生成样本**只读取** results/e5/e5_events.jsonl（只读，不回写）。
2. 不修改任何既有业务脚本（generate_music.py / evaluate_e4_sampling.py / visualize_embedding.py 等一律不动）。
3. 不改动 E5 主实验图 e5_umap_46d.png / e5_tsne_46d.png，本脚本只输出新增的两张补充校验图。
4. 特征计算全部复用仓库既有 symbolic_features.extract_symbolic_features（46 维，直方图内部已做概率归一化）。

【样本口径】
- GT 真值（**256 事件非重叠滑窗**，与 E5 主实验 / 训练阶段滑窗规则一致，
  stride = window_size = 256，不足 256 的尾段直接丢弃）：
    南音训练集 9 首（output/BasicPitch/*.mid）+ 南音验证集 2 首
    + 外部非南音音乐 2 首（目录 古典曲/*.mid）
    每首曲目先经 CompoundEventCodec 编码为全曲事件序列，再做非重叠滑窗；
    **1 个滑窗窗口 = 1 条 46 维特征 = 图上 1 个 GT 点**。
- 生成样本（聚合逻辑不变）：丢弃前 16 个 prompt 事件，整条续写全部事件
  聚合为**一组** 46 维特征；**一次续写只输出 1 个点**，禁止续写内部分段产生多个点。
- 多长度绘图：target_len ∈ [32, 64, 128, 256]，每个长度独立出一对 UMAP / t-SNE 图。

【降维规则（关键）】
- StandardScaler / UMAP / t-SNE 均在**全部滑窗 GT 样本**上 fit；
- 生成样本特征向量用 transform 投射进已拟合完成的空间，禁止全体样本混在一起重新 fit；
- 四个 target_len 循环复用**同一套**已 fit 好的 scaler / 降维模型，保证跨长度可比。
- 超参严格沿用 configs/e5_model_drift.yaml：
    UMAP: n_neighbors=8, min_dist=0.1, random_state=42
    t-SNE: perplexity=5, max_iter=2000, init=pca, random_state=42
  ⚠ sklearn 的 TSNE 不提供 out-of-sample transform，故 t-SNE 侧使用 openTSNE
  （TSNEEmbedding.transform）完成"GT 拟合 → 生成样本 transform"，与 UMAP 侧口径一致。

【客观局限性（脚本不做任何主观解读，仅出图 + 打印客观数值）】
- GT 为 256 事件非重叠滑窗窗口特征；生成样本为"单 prompt 引出的一条续写"整体聚合特征，
  各 target_len 下续写事件数为 16 / 48 / 112 / 240，与 GT 窗口的 256 事件**不等长**，
  且短长度（32/64）统计噪声高；本图仅作定性辅助校验，不能替代 E5 窗口粒度定量评估。
- 本分支仅做后处理可视化，E5 原有全部实验结论保持不变。

输出（reports/e5/，每个 target_len 一对）：
    e5_sliding_umap_len{32,64,128,256}_check.png
    e5_sliding_tsne_len{32,64,128,256}_check.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402
from sklearn.preprocessing import StandardScaler       # noqa: E402
from umap import UMAP                                  # noqa: E402
from openTSNE import TSNE as OpenTSNE                  # noqa: E402

from dataset_nanyin import (                           # noqa: E402
    TRAIN_SPLIT_IDS, VAL_SPLIT_IDS, _DATASET_BASE,
    encode_song_events, estimate_mode_from_pitch, get_compound_codec,
    slice_event_windows,
)
from nanyin_metrics import resolve_mode_pcs            # noqa: E402
from nanyin_config import load_config                  # noqa: E402
from symbolic_features import (                        # noqa: E402
    FEATURE_DIM, NUM_PITCH_CLASS_BINS, extract_symbolic_features,
)

DEFAULT_CONFIG = os.path.join(_SCRIPT_DIR, "configs", "e5_model_drift.yaml")
DEFAULT_EXTERNAL_DIR = os.path.join(_SCRIPT_DIR, "古典曲")

# 类别 → 视觉编码（颜色区分数据类别/模型，形状区分 GT 与生成模型）
CATEGORY_ORDER = ["GT_train", "GT_val", "GT_external", "boyatcn", "musicmamba"]
STYLE = {
    "GT_train":    {"color": "#2E7D32", "marker": "o", "size": 78,  "edge": "black"},
    "GT_val":      {"color": "#7CB342", "marker": "o", "size": 110, "edge": "black"},
    "GT_external": {"color": "#8C8C8C", "marker": "o", "size": 78,  "edge": "black"},
    "boyatcn":     {"color": "#1F77B4", "marker": "^", "size": 34,  "edge": "white"},
    "musicmamba":  {"color": "#FF7F0E", "marker": "s", "size": 34,  "edge": "white"},
}
LABEL = {
    "GT_train":    "GT 南音训练集（256 事件滑窗）",
    "GT_val":      "GT 南音验证集（256 事件滑窗）",
    "GT_external": "GT 外部非南音（256 事件滑窗）",
    "boyatcn":     "BoYaTCN 续写整体聚合（1 条续写 = 1 点）",
    "musicmamba":  "MusicMamba 续写整体聚合（1 条续写 = 1 点）",
}

LIMITATION_NOTE = (
    "注：GT = 256 事件非重叠滑窗窗口全局统计（与 E5 / 训练同滑窗规则）；"
    "生成样本 = 单 prompt 引出的一条续写整体聚合统计（续写内不分段，1 条 = 1 点）。"
    "各 target_len 下续写事件数与 GT 窗口 256 事件不等长，本图仅供定性辅助校验，"
    "不能替代 E5 窗口粒度定量评估。"
)


# ------------------------------------------------------------------------------
# 基础工具
# ------------------------------------------------------------------------------

def _use_cjk_font() -> None:
    """Windows 下显式指定中文字体（DejaVu Sans 缺 CJK 字形会渲染成方块）。"""
    try:
        from matplotlib import font_manager
        names = {f.name for f in font_manager.fontManager.ttflist}
        for cand in ("Microsoft YaHei", "SimHei", "DengXian", "KaiTi", "SimSun"):
            if cand in names:
                plt.rcParams["font.sans-serif"] = [cand]
                plt.rcParams["axes.unicode_minus"] = False
                return
    except Exception:
        pass


def _read_jsonl(path: str) -> list:
    """只读读取 E5 事件 JSONL（绝不回写）。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"[fullsong-vis] 事件文件不存在: {path}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _cumulative_starts(shift_ids, codec) -> list:
    """前向 IOI 累加得到 onset：onset_{i+1} = onset_i + index_to_shift(shift_i)。

    与 evaluate_e4_sampling._cumulative_starts 口径完全一致。
    """
    starts = [0.0]
    t = 0.0
    for s in shift_ids[:-1]:
        t += codec.index_to_shift(int(s))
        starts.append(t)
    if len(starts) < len(shift_ids):
        starts += [starts[-1]] * (len(shift_ids) - len(starts))
    return starts


def _mode_pcs_of(pitch_ids) -> list:
    """曲目级调式 pitch-class 集合（与 E5 一致：estimate_mode_from_pitch → resolve_mode_pcs）。"""
    return resolve_mode_pcs(estimate_mode_from_pitch(list(pitch_ids)))


def _notes_from_midi(midi_path: str) -> list:
    """用 pretty_midi 解析任意 MIDI（含外部非南音曲目）→ 音符列表。

    与 dataset_nanyin.load_song_annotations 同口径：合并全部轨道音符，按 (start, pitch) 升序。
    """
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for inst in pm.instruments:
        for n in inst.notes:
            notes.append({
                "note": int(n.pitch),
                "duration": float(n.end - n.start),
                "start_time": float(n.start),
                "velocity": int(n.velocity),
            })
    notes.sort(key=lambda x: (x["start_time"], x["note"]))
    return notes


def _ids_to_feature(pitch_ids, duration_ids, shift_ids, codec, mode_pcs):
    """三路事件 ID → 46 维符号特征（复用仓库既有函数）。"""
    tokens = [int(x) for x in pitch_ids]
    durations = [codec.index_to_duration(int(x)) for x in duration_ids]
    starts = _cumulative_starts(shift_ids, codec)
    return extract_symbolic_features(tokens, durations, starts, mode_pcs=mode_pcs)


# ------------------------------------------------------------------------------
# 样本构建
# ------------------------------------------------------------------------------

def build_gt_samples(codec, external_dir: str, window_size: int, stride: int) -> list:
    """GT 样本：**256 事件非重叠滑窗**（与 E5 主实验 / 训练阶段滑窗规则一致）。

    南音训练 9 首 + 南音验证 2 首 + 外部非南音 2 首，每首曲目先编码为全曲事件序列，
    再按 [start, end) 非重叠切窗（不足 window_size 的尾段丢弃）；
    1 个滑窗窗口 = 1 条 46 维特征 = 图上 1 个 GT 点。
    """
    samples = []
    song_mode: dict = {}                    # 曲目级调式（生成样本与窗口级回退均用）

    def _add_windows(category, song_id, pitch_ids, duration_ids, shift_ids,
                     song_mode_name):
        n = len(pitch_ids)
        wins = slice_event_windows(n, window_size, stride)
        if not wins:
            print(f"[fullsong-vis] 曲目事件数 {n} < 窗口 {window_size}，无滑窗: {song_id}")
            return
        for wi, (s, e) in enumerate(wins):
            wp, wd, ws = pitch_ids[s:e], duration_ids[s:e], shift_ids[s:e]
            w_mode = estimate_mode_from_pitch(list(wp)) or song_mode_name
            samples.append({
                "category": category,
                "song_id": song_id,
                "window_index": wi,
                "window_range": (s, e),
                "num_events": len(wp),
                "mode": str(w_mode),
                "feat": _ids_to_feature(wp, wd, ws, codec, resolve_mode_pcs(w_mode)),
            })

    # 1) 南音：训练集 9 首 / 验证集 2 首（沿用 dataset_nanyin.encode_song_events）
    for category, song_ids in (("GT_train", TRAIN_SPLIT_IDS), ("GT_val", VAL_SPLIT_IDS)):
        for song_id in song_ids:
            ev = encode_song_events(song_id, base_dir=_DATASET_BASE, codec=codec)
            if ev is None:
                print(f"[fullsong-vis] 南音曲目加载失败: {song_id}")
                continue
            song_mode[song_id] = str(estimate_mode_from_pitch(list(ev["pitch_ids"])))
            _add_windows(category, song_id, ev["pitch_ids"], ev["duration_ids"],
                         ev["shift_ids"], song_mode[song_id])

    # 2) 外部非南音：目录内全部 MIDI（同一 codec 编码 → 与生成样本特征空间一致）
    ext_paths = sorted(glob.glob(os.path.join(external_dir, "*.mid")) +
                       glob.glob(os.path.join(external_dir, "*.midi")))
    if not ext_paths:
        print(f"[fullsong-vis] 警告：外部曲目目录未找到 MIDI: {external_dir}")
    for p in ext_paths:
        song_id = os.path.splitext(os.path.basename(p))[0]
        try:
            notes = _notes_from_midi(p)
        except Exception as e:
            print(f"[fullsong-vis] 外部曲目解析失败，跳过: {p} ({type(e).__name__}: {e})")
            continue
        if not notes:
            print(f"[fullsong-vis] 外部曲目无音符，跳过: {p}")
            continue
        ev = codec.encode_notes(notes)
        song_mode[song_id] = str(estimate_mode_from_pitch(list(ev["pitch_ids"])))
        _add_windows("GT_external", song_id, ev["pitch_ids"], ev["duration_ids"],
                     ev["shift_ids"], song_mode[song_id])

    return samples, song_mode


def build_gen_samples(records, codec, gt_mode_map: dict, skip_prompt: int,
                      target_len: int) -> list:
    """生成样本：仅 target_len == 目标长度的记录，丢弃前 skip_prompt 个 prompt 事件。"""
    samples = []
    for rec in records:
        if int(rec.get("target_len", -1)) != int(target_len):
            continue
        p_ids = list(rec["pitch_ids"])[skip_prompt:]
        d_ids = list(rec["duration_ids"])[skip_prompt:]
        s_ids = list(rec["shift_ids"])[skip_prompt:]
        if not p_ids:
            continue
        song_id = str(rec.get("song_id", ""))
        mode_pcs = gt_mode_map.get(song_id, resolve_mode_pcs(None))
        samples.append({
            "category": str(rec.get("model_name", "unknown")),
            "song_id": song_id,
            "seed": int(rec.get("model_seed", -1)),
            "window_index": int(rec.get("window_index", -1)),
            "num_events": len(p_ids),
            "feat": _ids_to_feature(p_ids, d_ids, s_ids, codec, mode_pcs),
        })
    return samples


# ------------------------------------------------------------------------------
# 降维：仅用 GT 拟合 → 生成样本 transform
# ------------------------------------------------------------------------------

def fit_reducers(X_gt: np.ndarray, rcfg: dict) -> dict:
    """StandardScaler / UMAP / t-SNE 均**只在滑窗 GT 样本上 fit**，返回可复用的拟合对象。"""
    scaler = StandardScaler().fit(X_gt)          # 只用 GT 拟合
    Xg = scaler.transform(X_gt)

    u = rcfg["umap"]
    um = UMAP(n_components=2,
              n_neighbors=int(u["n_neighbors"]),
              min_dist=float(u["min_dist"]),
              random_state=int(u["random_state"]),
              n_jobs=1)
    um.fit(Xg)                                   # 只用 GT 拟合

    t = rcfg["tsne"]
    # sklearn.manifold.TSNE 无 out-of-sample transform → 用 openTSNE 保持"GT 拟合 + 生成 transform"同口径
    emb = OpenTSNE(n_components=2,
                   perplexity=float(t["perplexity"]),
                   initialization=str(t["init"]),
                   random_state=int(t["random_state"]),
                   n_jobs=1)
    emb_gt = emb.fit(Xg)                         # 只用 GT 拟合
    return {"scaler": scaler, "umap": um, "tsne": emb_gt, "gt_scaled": Xg,
            "tsne_perplexity": float(t["perplexity"]), "tsne_n_iter": int(t["max_iter"])}


def project(fitted: dict, X: np.ndarray) -> dict:
    """把任意样本（GT 自身 / 生成样本）用已拟合模型 transform 投射，不重新 fit。"""
    Xs = fitted["scaler"].transform(X)
    return {
        "umap": fitted["umap"].transform(Xs),
        "tsne": fitted["tsne"].transform(Xs,
                                         perplexity=fitted["tsne_perplexity"],
                                         n_iter=fitted["tsne_n_iter"]),
    }


# ------------------------------------------------------------------------------
# 绘图
# ------------------------------------------------------------------------------

def _draw(xy_gt: np.ndarray, xy_gen: np.ndarray, gt: list, gen: list,
          method: str, rcfg: dict, path: str, target_len: int,
          skip_prompt: int, window_size: int) -> None:
    _use_cjk_font()
    fig, ax = plt.subplots(figsize=(9.6, 7.2), dpi=160)

    by_cat = {}
    for i, s in enumerate(gt):
        by_cat.setdefault(s["category"], []).append(("gt", i, s))
    for i, s in enumerate(gen):
        by_cat.setdefault(s["category"], []).append(("gen", i, s))

    handles = []
    for cat in CATEGORY_ORDER:
        items = by_cat.get(cat, [])
        if not items:
            continue
        st = STYLE[cat]
        pts = [xy_gt[i] if kind == "gt" else xy_gen[i] for kind, i, _ in items]
        xy = np.asarray(pts)
        ax.scatter(xy[:, 0], xy[:, 1], s=st["size"], c=st["color"],
                   marker=st["marker"], alpha=0.9 if cat.startswith("GT") else 0.75,
                   edgecolors=st["edge"], linewidths=0.8 if cat.startswith("GT") else 0.5,
                   zorder=3 if cat.startswith("GT") else 2)
        handles.append(Line2D([], [], marker=st["marker"], ls="", color=st["color"],
                              markersize=9 if cat.startswith("GT") else 7,
                              label=f"{LABEL[cat]}  n={len(items)}"))

        # 仅标注验证集 GT 滑窗（生成样本的来源曲目），避免图上文字过密
        if cat == "GT_val":
            for kind, i, s in items:
                ax.annotate(f"{s['song_id'][:12]}#w{s['window_index']}",
                            (xy_gt[i, 0], xy_gt[i, 1]),
                            fontsize=5.8, color=st["color"],
                            xytext=(3, 3), textcoords="offset points", zorder=4)

    n_cont = int(target_len) - int(skip_prompt)
    ax.set_title(f"滑窗粒度 46 维符号特征 — {method}（target_len={target_len}）\n"
                 f"（fit：全部 {len(gt)} 个 GT 滑窗窗口 = {window_size} 事件非重叠；"
                 f"生成样本 transform 投射，续写 {n_cont} 事件整体聚合）\n"
                 f"GT 点数 {len(gt)} / 生成点数 {len(gen)}", fontsize=10.0)
    ax.set_xlabel(f"{method} dim 1")
    ax.set_ylabel(f"{method} dim 2")
    ax.grid(alpha=0.25)
    ax.legend(handles=handles, fontsize=8, loc="best", framealpha=0.92)
    fig.text(0.5, 0.005, LIMITATION_NOTE, ha="center", fontsize=7.2, color="#555555")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fullsong-vis] 输出: {path}")


# ------------------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="E5 补充校验：滑窗粒度 46 维特征 UMAP / t-SNE（只读后处理，不训练不重生成）")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="E5 配置（沿用其 reduction 超参 / 滑窗规则）")
    ap.add_argument("--events-jsonl", default=None, help="默认取配置 outputs.events_jsonl（只读）")
    ap.add_argument("--external-dir", default=DEFAULT_EXTERNAL_DIR, help="外部非南音 MIDI 目录")
    ap.add_argument("--outdir", default=None, help="默认 reports/e5")
    ap.add_argument("--target-lens", default=None,
                    help="续写长度列表，逗号分隔；默认取配置 window.target_len_events_list")
    ap.add_argument("--window-size", type=int, default=None,
                    help="GT 滑窗窗口事件数，默认取配置 window.window_size_events（256）")
    ap.add_argument("--stride", type=int, default=None,
                    help="GT 滑窗步长，默认取配置 window.eval_stride_events（256，非重叠）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    wcfg = cfg.get("window", {})
    ecfg = cfg.get("evaluation", {})
    rcfg = cfg.get("reduction", {})
    skip_prompt = int(ecfg.get("skip_prompt", 16))
    window_size = int(args.window_size or wcfg.get("window_size_events", 256))
    stride = int(args.stride or wcfg.get("eval_stride_events", window_size))
    target_lens = ([int(x) for x in args.target_lens.split(",") if x.strip()]
                   if args.target_lens
                   else [int(x) for x in wcfg.get("target_len_events_list", [32, 64, 128, 256])])
    events_path = args.events_jsonl or os.path.join(
        _SCRIPT_DIR, str(cfg["outputs"]["events_jsonl"]))
    outdir = args.outdir or os.path.join(_SCRIPT_DIR, "reports", "e5")

    codec = get_compound_codec(cfg)
    print("[fullsong-vis] 分支用途：仅新增后处理校验可视化（无训练 / 无重生成）")
    print(f"[fullsong-vis] 只读事件文件: {events_path}")
    print(f"[fullsong-vis] GT 滑窗规则: window_size={window_size}, stride={stride}（非重叠，尾段丢弃）")
    print(f"[fullsong-vis] 续写长度循环: {target_lens}")

    # ---- GT：256 事件非重叠滑窗（训练集 / 验证集 / 外部非南音全部滑窗）----
    gt, song_mode = build_gt_samples(codec, args.external_dir, window_size, stride)
    if not gt:
        raise RuntimeError("[fullsong-vis] GT 滑窗样本为空，终止")
    print(f"[fullsong-vis] GT 滑窗样本 {len(gt)} 个")
    for cat in CATEGORY_ORDER:
        n_gt = sum(1 for s in gt if s["category"] == cat)
        if n_gt:
            print(f"    {LABEL[cat]}: {n_gt} 个窗口")

    # ---- 概率归一化校验（直方图分量应为概率分布，消除长度带来的计数偏差）----
    sums = [float(np.asarray(s["feat"])[:NUM_PITCH_CLASS_BINS].sum()) for s in gt]
    print(f"[fullsong-vis] pitch-class 直方图分量求和 = "
          f"{min(sums):.6f} ~ {max(sums):.6f}（应为 1.0，概率归一化）")

    X_gt = np.asarray([np.asarray(s["feat"], dtype=np.float64) for s in gt])
    if X_gt.shape[1] != FEATURE_DIM:
        raise RuntimeError(f"[fullsong-vis] 特征维数异常，应为 {FEATURE_DIM}")

    # ---- 只用 GT 滑窗 fit；四个长度复用同一套模型 ----
    fitted = fit_reducers(X_gt, rcfg)
    emb_gt = project(fitted, X_gt)

    # ---- 生成样本：只读 jsonl，聚合逻辑不变（1 条续写 = 1 个点）----
    records = _read_jsonl(events_path)
    mode_of_song = {sid: resolve_mode_pcs(m) for sid, m in song_mode.items()}
    nanyin_gt = np.asarray([s["feat"] for s in gt if s["category"] in ("GT_train", "GT_val")])
    centroid = nanyin_gt.mean(axis=0)

    os.makedirs(outdir, exist_ok=True)
    for L in target_lens:
        gen = build_gen_samples(records, codec, mode_of_song, skip_prompt, int(L))
        if not gen:
            print(f"[fullsong-vis] target_len={L}: 无匹配记录，跳过")
            continue
        X_gen = np.asarray([np.asarray(s["feat"], dtype=np.float64) for s in gen])
        emb_gen = project(fitted, X_gen)

        print(f"[fullsong-vis] target_len={L}: 生成样本 {len(gen)} 条"
              f"（续写 {int(L) - skip_prompt} 事件整体聚合）")
        # 种子/组构成留痕：保证每张图的样本来源可追溯
        comp: dict = {}
        for s in gen:
            comp[(s["category"], s["seed"])] = comp.get((s["category"], s["seed"]), 0) + 1
        print(f"    训练种子: {sorted({sd for _, sd in comp})}"
              f"；模型×种子组数 {len(comp)}，每组点数 {sorted(set(comp.values()))}")
        for (mdl, sd) in sorted(comp):
            pair_sel = [(str(r.get("song_id")), int(r.get("window_index", -1)))
                        for r in records
                        if int(r.get("target_len", -1)) == int(L)
                        and str(r.get("model_name")) == mdl
                        and int(r.get("model_seed", -1)) == sd]
            n_win = len(set(pair_sel))          # 验证窗口按 (曲目, 窗口号) 去重
            n_rng = len({r.get("rng_seed") for r in records
                         if int(r.get("target_len", -1)) == int(L)
                         and str(r.get("model_name")) == mdl
                         and int(r.get("model_seed", -1)) == sd})
            print(f"      {mdl} seed={sd}: {comp[(mdl, sd)]} 点"
                  f"（去重验证窗口 {n_win} 个 / 采样 rng_seed 去重 {n_rng} 个）")
        print(f"    原始 46 维空间 · 到南音 GT 质心欧氏距离（客观数值，不含解读）：")
        for cat in CATEGORY_ORDER:
            arr = np.asarray([s["feat"] for s in gen if s["category"] == cat])
            if arr.size == 0:
                continue
            d = np.linalg.norm(arr - centroid, axis=1)
            print(f"    {LABEL[cat]}: mean={d.mean():.4f}  min={d.min():.4f} "
                  f"max={d.max():.4f}  n={len(d)}")

        _draw(emb_gt["umap"], emb_gen["umap"], gt, gen, "UMAP", rcfg,
              os.path.join(outdir, f"e5_sliding_umap_len{int(L)}_check.png"),
              int(L), skip_prompt, window_size)
        _draw(emb_gt["tsne"], emb_gen["tsne"], gt, gen, "t-SNE", rcfg,
              os.path.join(outdir, f"e5_sliding_tsne_len{int(L)}_check.png"),
              int(L), skip_prompt, window_size)

    print("[fullsong-vis] 完成。本分支不改变 E5 任何既有结论。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
