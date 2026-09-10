# -*- coding: utf-8 -*-
"""
visualize_embedding.py — 符号统计特征 embedding 降维可视化 (UMAP / t-SNE)
==========================================================================

目标：观察四类样本的 embedding 是否聚集，佐证生成片段的南音风格相似性：
  1. 真实南音样本    : 11 首 dapu GT (BasicPitch 转录主干 midi) 的滑窗片段
  2. 续写前置片段    : 验证曲 GT 头部 PROMPT_LEN=16 token（模型实际喂入的前缀）
  3. 双向原版生成片段: results/generated_midi_old  (双向注意力原版模型产物)
  4. 因果自回归生成片段: results/generated_midi     (因果自回归修改版产物)

表征：符号统计特征向量（本地计算、零模型依赖）
  每段 = pitch-class 直方图(12) + 相邻音程直方图(12) + log-时值直方图(8)
        + log-IOI 直方图(8) + 音高统计(2) + 音符密度(1) + 调式符合度(1)
        + 装饰音占比(2)，共 46 维，StandardScaler 标准化后降维。

输出:
  results/embedding_symbolic_umap.png
  results/embedding_symbolic_tsne.png
运行:
  venv/Scripts/python.exe visualize_embedding.py
"""

import os
import sys
import glob

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # pyright: ignore

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from nanyin_metrics import (  # noqa: E402
    parse_midi_to_sequence,
    load_ground_truth,
    resolve_mode_pcs,
    PROMPT_LEN,
)
from nanyin_metrics import _VAL_SONG_IDS as VAL_SONG_IDS  # noqa: E402
from nanyin_metrics import _DATASET_BASE  # noqa: E402

RESULTS_DIR = os.path.join(_SCRIPT_DIR, "results")

# ---------------------------------------------------------------------------
# 中文字体（Windows 微软雅黑；找不到则退回默认英文标签）
# ---------------------------------------------------------------------------
_CN_FONT = None
for _fp in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc",
            r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(_fp):
        try:
            font_manager.fontManager.addfont(_fp)
            _CN_FONT = font_manager.FontProperties(fname=_fp).get_name()
            break
        except Exception:
            pass
if _CN_FONT:
    plt.rcParams["font.family"] = _CN_FONT
    plt.rcParams["axes.unicode_minus"] = False

# 南音五空管 gong 音阶（模式参照，与 nanyin_metrics 一致）
_GONG_PCS = sorted(resolve_mode_pcs("gong"))

_WIN, _STRIDE, _MINLEN, _SKIP = 256, 160, 64, PROMPT_LEN  # 16


# ---------------------------------------------------------------------------
# 特征工程：每段序列 → 46 维符号统计向量
# ---------------------------------------------------------------------------
def _pcs(tokens):
    return (np.asarray(tokens) % 12).astype(int)


def _log_hist(vals, lo, hi, nbins):
    v = np.asarray([float(x) for x in vals if x is not None], dtype=float)
    if v.size == 0:
        return np.zeros(nbins)
    edges = np.logspace(lo, hi, nbins + 1)
    h, _ = np.histogram(np.clip(v, edges[0], edges[-1]), bins=edges)
    return h.astype(float) / max(v.size, 1)


def extract_features(tokens, durations, starts):
    """一段 (tokens/durations/starts) → 46 维特征。"""
    t = np.asarray(tokens, dtype=float)
    d = np.asarray(durations, dtype=float)
    s = np.asarray(starts, dtype=float)
    n = len(t)
    f = np.zeros(46)

    # 1. pitch-class 直方图 (12)
    pc = np.bincount(_pcs(t), minlength=12).astype(float) / n
    f[0:12] = pc

    # 2. 相邻音程 |diff| 直方图 (12, clip 0..11)
    if n >= 2:
        ad = np.abs(np.diff(t)).clip(0, 11).astype(int)
        f[12:24] = np.bincount(ad, minlength=12).astype(float) / (n - 1)

    # 3. log-时值直方图 (8): 0.05s ~ 2s
    f[24:32] = _log_hist(d, -1.30, 0.301, 8)

    # 4. log-IOI 直方图 (8): 0.03s ~ 2s（去同时/非正 onset）
    if n >= 2:
        ioi = np.sort(s)[1:] - np.sort(s)[:-1]
        ioi = ioi[ioi > 0]
        if ioi.size:
            f[32:40] = _log_hist(ioi, -1.52, 0.301, 8)

    # 5. 音高统计 (2): 均值/标准差（归一）
    f[40] = (float(np.mean(t)) if n else 60.0) / 100.0
    f[41] = (float(np.std(t)) if n else 0.0) / 20.0

    # 6. 音符密度 (1): 每秒音符数（log1p 归一）
    span = max(float(s[-1] - s[0]), 1e-6) if n >= 2 else 1e-6
    f[42] = min(np.log1p(n / span) / 3.0, 1.0)

    # 7. 调式符合度 (1): 落在 gong 五空管 pcs 内的占比
    f[43] = float(np.mean(np.isin(_pcs(t), _GONG_PCS)))

    # 8. 装饰音占比 (2): 快速邻跳比率 |diff|∈[1,6] & 前一音≤0.3s；短音比率 dur≤0.25s
    if n >= 2:
        f[44] = float(np.mean(
            (np.abs(np.diff(t)) >= 1) & (np.abs(np.diff(t)) <= 6)
            & (d[:-1] <= 0.30)
        ))
    f[45] = float(np.mean(d <= 0.25))
    return f


def slice_windows(tokens, durations, starts, skip=_SKIP,
                  win=_WIN, stride=_STRIDE, minlen=_MINLEN):
    """从 skip 起的滑窗切片；不足一段则取整段。"""
    n = len(tokens)
    if n <= skip:
        yield 0, n
        return
    starts_idx = list(range(skip, n - minlen + 1, stride))
    if not starts_idx:
        starts_idx = [skip]
    for st in starts_idx:
        yield st, min(st + win, n)


# ---------------------------------------------------------------------------
# 收集样本
# ---------------------------------------------------------------------------
def collect_samples():
    samples = []  # (类别, 模型/曲信息, 特征向量, meta)
    feat_rows = []

    # ---- 1. 真实南音样本：11 首 GT BasicPitch 主干 midi 滑窗 ----
    bp_files = sorted(glob.glob(os.path.join(_DATASET_BASE, "*.mid")))
    print(f"[真实南音] 找到 {len(bp_files)} 首 GT midi")
    n_real = 0
    for mp in bp_files:
        toks, durs, starts, _ = parse_midi_to_sequence(mp)
        if not toks:
            continue
        for st, en in slice_windows(toks, durs, starts):
            if en - st < _MINLEN:
                continue
            v = extract_features(toks[st:en], durs[st:en], starts[st:en])
            samples.append(("真实南音", os.path.splitext(os.path.basename(mp))[0], v, st))
            n_real += 1
    print(f"   → {n_real} 个窗口样本")

    # ---- 2. 续写前置片段：验证曲 GT 头部 PROMPT_LEN token ----
    n_prompt = 0
    for sid in VAL_SONG_IDS:
        gt = load_ground_truth(sid)
        if gt is None:
            continue
        p = int(min(PROMPT_LEN, len(gt["tokens"])))
        if p < 4:
            continue
        v = extract_features(gt["tokens"][:p], gt["durations"][:p],
                             np.asarray(gt["starts"])[:p])
        samples.append(("续写前置片段", sid, v, 0))
        n_prompt += 1
    print(f"[续写前置片段] {n_prompt} 个样本 (PROMPT_LEN={PROMPT_LEN})")

    # ---- 3/4. 两架构生成样本：old / causal midi 滑窗 ----
    arch_conf = [
        ("双向原版生成", os.path.join(RESULTS_DIR, "generated_midi_old")),
        ("因果自回归生成", os.path.join(RESULTS_DIR, "generated_midi")),
    ]
    for arch, base in arch_conf:
        n_arch = 0
        for subdir in ("boyatcn", "musicmamba"):
            for sid in VAL_SONG_IDS:
                mp = os.path.join(base, subdir, f"{sid}_{subdir}.mid")
                if not os.path.exists(mp):
                    print(f"  ⚠ 缺失: {mp}")
                    continue
                toks, durs, starts, _ = parse_midi_to_sequence(mp)
                if not toks:
                    continue
                for st, en in slice_windows(toks, durs, starts):
                    if en - st < _MINLEN:
                        continue
                    v = extract_features(toks[st:en], durs[st:en], starts[st:en])
                    samples.append((arch, f"{subdir}|{sid}", v, st))
                    n_arch += 1
        print(f"[{arch}] {n_arch} 个窗口样本")
    return samples


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
CAT_COLOR = {
    "真实南音": "#3b82c4",
    "续写前置片段": "#2e9e5b",
    "双向原版生成": "#e0913a",
    "因果自回归生成": "#d8503f",
}
MODEL_MARKER = {
    "boyatcn": "o",
    "musicmamba": "^",
}


def _gt_palette(samples):
    """给真实南音每首曲分配稳定可区分的颜色。
    颜色按曲名排序映射到 tab20/tab20b，并用隔一取一方式让相邻曲色差更大；
    跨两张图保持一致（同曲同色）。"""
    names = sorted({s[1] for s in samples if s[0] == "真实南音"})
    # 隔一个取一个，避免 tab20 深浅相邻色过于相似
    allc = (list(plt.get_cmap("tab20").colors)[::2]
            + list(plt.get_cmap("tab20b").colors)[::2])
    return {nm: allc[i % len(allc)] for i, nm in enumerate(names)}


def _plot_scatter(ax, xy, samples, title):
    for cat in CAT_COLOR:
        idx = [i for i, s in enumerate(samples) if s[0] == cat]
        if not idx:
            continue
        xs = xy[idx, 0]
        ys = xy[idx, 1]
        if cat == "真实南音":
            # 按曲目分色：观察同一首曲的滑窗片段是否聚在一起
            palette = _gt_palette(samples)
            for nm in sorted(palette):
                sub_idx = [i for i in idx if samples[i][1] == nm]
                if not sub_idx:
                    continue
                ax.scatter(xy[sub_idx, 0], xy[sub_idx, 1], s=16,
                           c=[palette[nm]], alpha=0.5,
                           edgecolors="none", zorder=1,
                           label=f"真实南音 · {nm}")
        elif cat == "续写前置片段":
            ax.scatter(xs, ys, s=150, c=CAT_COLOR[cat], marker="*",
                       edgecolors="k", linewidths=0.6, zorder=5, label=cat)
        else:  # 生成样本：颜色=架构组, marker=子模型 (boyatcn=o / musicmamba=^)
            for sub, mk in MODEL_MARKER.items():
                sub_idx = [i for i in idx if sub in samples[i][1]]
                if sub_idx:
                    ax.scatter(xy[sub_idx, 0], xy[sub_idx, 1], s=42,
                               c=CAT_COLOR[cat], marker=mk,
                               edgecolors="k", linewidths=0.4,
                               alpha=0.85, zorder=3,
                               label=f"{cat} · {sub}")
    ax.set_title(title)
    # 图例条目较多，放到图外右侧避免遮挡
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=8, framealpha=0.9)


def main() -> None:
    print("=" * 70)
    print("  符号统计特征 embedding 降维可视化 (UMAP / t-SNE)")
    print("=" * 70)

    samples = collect_samples()
    X = np.vstack([s[2] for s in samples])
    print(f"\n  样本总数 = {len(samples)}，特征维度 = {X.shape[1]}")
    for cat in CAT_COLOR:
        print(f"    {cat:<10} {sum(1 for s in samples if s[0] == cat)}")

    Xs = StandardScaler().fit_transform(X)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    umap_out = os.path.join(RESULTS_DIR, "embedding_symbolic_umap.png")
    tsne_out = os.path.join(RESULTS_DIR, "embedding_symbolic_tsne.png")

    # ---- UMAP ----
    print("\n[UMAP] 计算中 ...")
    from umap import UMAP
    xy_u = UMAP(n_neighbors=15, min_dist=0.15, n_components=2,
                random_state=42, n_jobs=1).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=150)
    _plot_scatter(ax, xy_u, samples,
                  "符号特征 embedding · UMAP (GT=按曲分色 · prompt=星 · 生成: o=boyatcn / ^=musicmamba)")
    fig.tight_layout()
    fig.savefig(umap_out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {umap_out}")

    # ---- t-SNE ----
    print("[t-SNE] 计算中 ...")
    xy_t = TSNE(n_components=2, perplexity=min(22, max(5, len(samples) // 5 - 1)),
                init="pca", random_state=42, max_iter=2000).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=150)
    _plot_scatter(ax, xy_t, samples,
                  "符号特征 embedding · t-SNE (GT=按曲分色 · prompt=星 · 生成: o=boyatcn / ^=musicmamba)")
    fig.tight_layout()
    fig.savefig(tsne_out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {tsne_out}")

    print("=" * 70)


if __name__ == "__main__":
    main()
