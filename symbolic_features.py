# -*- coding: utf-8 -*-
"""
symbolic_features.py — E4 新增：46 维符号统计特征
=================================================

用途
----
对**窗口级生成结果**（或对应 GT 窗口）计算一个固定 46 维的符号统计量，
用于：
  1) 同一窗口不同采样参数之间的生成多样性度量（两两欧氏距离）；
  2) 后续可选的降维可视化（PCA / UMAP / t-SNE）。

⚠ 历史说明：EXPERIMENT_LOG 中记录的 exp3 阶段 46 维特征由其临时脚本
  （`exp3_recipe/_exp3_eval_windows.py`，未入库）产生，本工作区的
  `feature_extractor.py` 实为**音频**特征提取（librosa F0/MFCC），
  并不包含该 46 维符号特征。E4 按其文档口径重新实现，维数定义如下，
  保证与 exp3 记录的分块结构一致（若未来找回原脚本，可对齐替换）。

46 维构成（**评估区域 = 跳过 16 个 prompt 事件后的 240 个事件**）
--------------------------------------------------------------
| 分块                | 维数 | 定义                                                    |
|---------------------|------|---------------------------------------------------------|
| pitch-class 直方图  | 12   | 音高 % 12 的归一化频次                                  |
| 有向音程直方图      | 12   | 相邻音高差（半音）落入 12 个有向桶的归一化频次            |
| log-时值直方图      | 8    | log2(duration) 落在 8 个对数分箱的归一化频次             |
| log-IOI 直方图      | 8    | log2(IOI)（相邻 onset 间隔）落在 8 个对数分箱的归一化频次 |
| 音高统计            | 2    | 音高均值 /127、音高标准差 /24                            |
| 音符密度            | 1    | 每秒音符数（/10 后截断到 [0,1]）                          |
| 调式符合度          | 1    | 落在目标调式五声音级内的音符占比                          |
| 装饰音占比          | 2    | 短音占比（<0.15s）、短音邻音跳进占比                      |
| **合计**            | 46   |                                                          |

有向音程 12 桶划分（_interval_bin）：
    idx 0      : 同音（interval == 0）
    idx 1..5   : +1 .. +5 半音
    idx 6      : >= +6 半音（大跳上行）
    idx 7..11  : -5 .. -1 半音；idx 11 同时吸收 <= -6 半音？—— **不**，见下方精确定义：
        idx 7  : -1
        idx 8  : -2
        idx 9  : -3
        idx 10 : -4
        idx 11 : <= -5（含 -5 及更大下行）

所有分块内部归一化（和为 1），统计量类压缩到 [0, 1]；序列为空时返回全零向量，
保证距离计算不产生 NaN。
"""

from typing import List, Optional, Sequence

import numpy as np

# ------------------------------------------------------------------------------
# 常量
# ------------------------------------------------------------------------------

NUM_PITCH_CLASS_BINS: int = 12
NUM_INTERVAL_BINS: int = 12
NUM_LOG_DURATION_BINS: int = 8
NUM_LOG_IOI_BINS: int = 8
NUM_PITCH_STATS: int = 2
NUM_DENSITY: int = 1
NUM_MODE: int = 1
NUM_ORNAMENT: int = 2

FEATURE_DIM: int = (
    NUM_PITCH_CLASS_BINS + NUM_INTERVAL_BINS + NUM_LOG_DURATION_BINS
    + NUM_LOG_IOI_BINS + NUM_PITCH_STATS + NUM_DENSITY + NUM_MODE + NUM_ORNAMENT
)  # = 46

# log 分箱边界（秒）：覆盖 2^-7 ~ 2^3.5 ≈ 0.0078 s ~ 11.3 s
_LOG_MIN_EXP: float = -7.0
_LOG_MAX_EXP: float = 3.5

ORNAMENT_SHORT_SEC: float = 0.15       # 短音阈值（与 dataset_nanyin / nanyin_metrics 一致）
ORNAMENT_LEAP_SEMI: float = 3.0        # 邻音跳进阈值（半音）

FEATURE_NAMES: List[str] = (
    [f"pc_{i:02d}" for i in range(NUM_PITCH_CLASS_BINS)]
    + [f"interval_{i:02d}" for i in range(NUM_INTERVAL_BINS)]
    + [f"log_dur_{i:02d}" for i in range(NUM_LOG_DURATION_BINS)]
    + [f"log_ioi_{i:02d}" for i in range(NUM_LOG_IOI_BINS)]
    + ["pitch_mean", "pitch_std"]
    + ["note_density"]
    + ["mode_in_scale_ratio"]
    + ["short_note_ratio", "ornament_ratio"]
)


# ------------------------------------------------------------------------------
# 内部工具
# ------------------------------------------------------------------------------

def _safe_hist(values: np.ndarray, num_bins: int, lo: float, hi: float) -> np.ndarray:
    """把 value 序列按 [lo, hi] 等宽分箱后归一化（含越界吸收到端点箱）。"""
    if values.size == 0:
        return np.zeros(num_bins, dtype=np.float64)
    idx = np.floor((values - lo) / (hi - lo) * num_bins).astype(int)
    idx = np.clip(idx, 0, num_bins - 1)
    hist = np.bincount(idx, minlength=num_bins).astype(np.float64)
    total = hist.sum()
    return hist / total if total > 0 else hist


def _interval_bin(delta: int) -> int:
    """有向音程（半音）→ 0..11 桶。"""
    if delta == 0:
        return 0
    if delta > 0:
        return int(min(delta, 6))                 # 1..5 → 1..5；>=6 → 6
    return int(7 + min(-delta - 1, 4))            # -1..-5 → 7..11；<=-5 → 11


def interval_bin_hist(tokens: Sequence[int]) -> np.ndarray:
    """相邻音高 → 12 桶直方图（归一化）。长度 < 2 时返回全零。"""
    hist = np.zeros(NUM_INTERVAL_BINS, dtype=np.float64)
    arr = np.asarray(tokens, dtype=int)
    if arr.size < 2:
        return hist
    deltas = np.diff(arr)
    for d in deltas:
        hist[_interval_bin(int(d))] += 1.0
    total = hist.sum()
    return hist / total if total > 0 else hist


def _log_hist(values: np.ndarray, num_bins: int) -> np.ndarray:
    """正值序列 → log2 均匀分箱直方图（归一化）；非正值忽略。"""
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[vals > 0]
    if vals.size == 0:
        return np.zeros(num_bins, dtype=np.float64)
    logs = np.log2(vals)
    return _safe_hist(logs, num_bins, _LOG_MIN_EXP, _LOG_MAX_EXP)


def _ornament_ratios(
    durations: Sequence[float],
    tokens: Sequence[int],
) -> tuple:
    """(短音占比, 短音邻音跳进占比)，规则与 nanyin_metrics._derive_ornament_flags 一致。"""
    d = np.asarray(durations, dtype=np.float64)
    p = np.asarray(tokens, dtype=np.float64)
    if d.size == 0:
        return 0.0, 0.0
    short_mask = d < ORNAMENT_SHORT_SEC
    short_ratio = float(short_mask.mean())
    if int(short_mask.sum()) == 0:
        return short_ratio, 0.0
    idx = np.nonzero(short_mask)[0]
    flags = 0
    for i in idx:
        prev_gap = abs(p[i] - p[i - 1]) if i > 0 else 99.0
        next_gap = abs(p[i] - p[i + 1]) if i + 1 < len(p) else 99.0
        if prev_gap >= ORNAMENT_LEAP_SEMI or next_gap >= ORNAMENT_LEAP_SEMI:
            flags += 1
    return short_ratio, float(flags) / float(len(idx))


# ------------------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------------------

def extract_symbolic_features(
    tokens: Sequence[int],
    durations: Sequence[float],
    starts: Optional[Sequence[float]] = None,
    mode_pcs: Optional[Sequence[int]] = None,
    total_duration_sec: Optional[float] = None,
) -> np.ndarray:
    """计算 46 维符号统计特征（全部落在 [0, 1] 且有限）。

    参数:
        tokens:     MIDI 音高序列（评估区域，通常 = 240 个续写事件）
        durations:  对应时值（秒）
        starts:     对应 onset（秒）；缺省时退化为 durations 累加（无法还原休止）
        mode_pcs:   目标调式的 pitch-class 集合（None → 调式符合度记 0）
        total_duration_sec: 评估区域总时长（秒）；缺省用最后一个 onset+duration 推算
    返回:
        np.ndarray, shape = (46,), dtype = float64
    """
    tok = np.asarray(tokens, dtype=int)
    dur = np.asarray(durations, dtype=np.float64)
    n = int(tok.size)
    if dur.size != n:
        dur = np.resize(dur, n)

    if n == 0:
        return np.zeros(FEATURE_DIM, dtype=np.float64)

    # ---- onset ----
    if starts is not None and len(starts) == n:
        ons = np.asarray(starts, dtype=np.float64)
    else:
        ons = np.cumsum(np.maximum(dur, 0.0))

    # ---- 12 维 pitch-class 直方图 ----
    pc_hist = np.zeros(NUM_PITCH_CLASS_BINS, dtype=np.float64)
    pc = tok % 12
    pc_hist = np.bincount(pc, minlength=NUM_PITCH_CLASS_BINS).astype(np.float64)
    if pc_hist.sum() > 0:
        pc_hist = pc_hist / pc_hist.sum()

    # ---- 12 维有向音程直方图 ----
    iv_hist = interval_bin_hist(tok)

    # ---- 8 维 log-时值 / 8 维 log-IOI 直方图 ----
    dur_hist = _log_hist(dur, NUM_LOG_DURATION_BINS)
    if n >= 2:
        ioi = np.diff(ons)
    else:
        ioi = np.zeros(0, dtype=np.float64)
    ioi_hist = _log_hist(ioi, NUM_LOG_IOI_BINS)

    # ---- 2 维音高统计 ----
    pitch_mean = float(np.clip(tok.mean() / 127.0, 0.0, 1.0))
    pitch_std = float(np.clip(tok.std() / 24.0, 0.0, 1.0)) if n > 1 else 0.0

    # ---- 1 维音符密度 ----
    if total_duration_sec is None:
        span = float(ons[-1] + max(dur[-1], 0.0)) if n > 0 else 0.0
    else:
        span = float(total_duration_sec)
    density = float(np.clip((n / span / 10.0), 0.0, 1.0)) if span > 0 else 0.0

    # ---- 1 维调式符合度 ----
    if mode_pcs is not None:
        in_scale = float(np.isin(pc, np.asarray(list(mode_pcs), dtype=int)).mean())
    else:
        in_scale = 0.0

    # ---- 2 维装饰音占比 ----
    short_ratio, orn_ratio = _ornament_ratios(dur, tok)

    feat = np.concatenate([
        pc_hist,                                   # 12
        iv_hist,                                   # 12
        dur_hist,                                  # 8
        ioi_hist,                                  # 8
        np.array([pitch_mean, pitch_std]),         # 2
        np.array([density]),                       # 1
        np.array([in_scale]),                      # 1
        np.array([short_ratio, orn_ratio]),        # 2
    ]).astype(np.float64)

    if feat.shape[0] != FEATURE_DIM:
        raise RuntimeError(f"[symbolic_features] 特征维数异常: {feat.shape[0]} != {FEATURE_DIM}")
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


def pairwise_distance_matrix(features: List[np.ndarray]) -> np.ndarray:
    """N 条 46 维特征 → (N, N) 欧氏距离矩阵。"""
    mat = np.asarray(features, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float64)
    sq = (mat ** 2).sum(axis=1, keepdims=True)
    dist_sq = sq + sq.T - 2.0 * (mat @ mat.T)
    dist_sq = np.clip(dist_sq, 0.0, None)
    return np.sqrt(dist_sq)


__all__ = [
    "FEATURE_DIM", "FEATURE_NAMES",
    "extract_symbolic_features", "pairwise_distance_matrix", "interval_bin_hist",
]
