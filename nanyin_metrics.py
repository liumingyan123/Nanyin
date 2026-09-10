# -*- coding: utf-8 -*-
"""
NOEF 南音客观评估指标库（Nanyin Objective Evaluation Framework）
================================================================

本模块提供南音模型生成旋律的全套客观评估指标，涵盖五大类：
  1. 音高类：pitch_accuracy、pitch_stability
  2. 节奏类：rhythm_consistency、tempo_accuracy
  3. 装饰音类：ornament_score、ornament_density
  4. 调式类：mode_matching_score
  5. 整体类：fad_distance、melody_cosine_similarity

设计原则：
  - 主体评估在**符号级**进行：解析生成 MIDI 音符序列，与 ground-truth
    token/duration 序列逐项比对。
  - FAD 使用**音频级**评估：加载合成 WAV 与参考 WAV 计算 MFCC 分布距离。
  - 所有指标归一化至 **0~100**（越高越好）。
  - 独立工具类，无训练代码依赖（仅需 numpy/scipy/librosa/mido/sklearn）。

用法示例：
    python nanyin_metrics.py              # 默认批量评估 generate_music.py 输出

    from nanyin_metrics import batch_evaluate_generated
    batch_evaluate_generated()            # 遍历全部生成 MIDI + GT → CSV

参考文献总览：
  [Huang+18]   Music Transformer           arXiv:1809.04281
  [Maman+24]   MusicMamba (KAD)            arXiv:2502.15602
  [Kilgour+19] Fréchet Audio Distance      arXiv:1812.08466
  [Dixon01]    Beat Tracking Evaluation    JAES 49(9), 2001
  [Chordia07]  Ornament Detection          ISMIR 2007
  [McKinney06] Tempo Extraction            JNMR 35(1), 2006
  [Umbert13]   Singing Voice Eval          ISMIR 2013
  [Wiggins93]  Tonal Center Detection      Computing in Musicology, 1993
  [Serra11]    Melodic Similarity          IEEE TASLP 19(1), 2011
  [Yang12]     Music Similarity            ACM Computing Surveys 44(2), 2012
  [MelodyGLM]  Symbolic Melody Generation  arXiv:2309.10738
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
from scipy.linalg import sqrtm

# 南音自回归生成的 prompt 长度（与 generate_music.py 的 MIN_PROMPT_LEN 保持一致）。
# 评估 GT-参照指标时默认跳过前 PROMPT_LEN 个 token，避免 prompt 区域污染评估分数。
PROMPT_LEN: int = 16

# 可选依赖：librosa 用于 FAD 音频评估，mido 用于 MIDI 解析
try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

try:
    import mido
    _HAS_MIDO = True
except ImportError:
    _HAS_MIDO = False

try:
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ==============================================================================
# 一、全局常量与配置
# ==============================================================================

# 南音四种专属调式对应的 12 音级 pitch-class 集合（0=C）
#  参考文献：五声音阶宫商角徵羽体系，半音偏移对齐南音管门
NANYIN_MODES: Dict[str, List[int]] = {
    "gong":  [0, 2, 4, 7, 9],       # 五空管（宫调式）
    "shang": [2, 4, 7, 9, 11],      # 四空管（商调式）
    "jue":   [4, 7, 9, 11, 1],      # 倍思管（角调式，1=1 mod 12）
    "zhi":   [7, 9, 11, 1, 3],      # 尺调（徵调式，1,3=1,3 mod 12）
}

# 传统管门中文名 → 调式键名
TRADITIONAL_MODE_MAP: Dict[str, str] = {
    "五空管":     "gong",
    "四空管":     "shang",
    "倍思管":     "jue",
    "尺调":       "zhi",
    "尺管":       "zhi",
    "五空四仪管": "gong",  # 变体映射
    "五空四乂管": "gong",  # 异体字变体
}

# 南音四种调式中文名列表
NANYIN_FOUR_MODES_CHINESE: List[str] = ["五空管", "五空四仪管", "倍思管", "四空管"]

# NOEF 五大类权重（用于综合加权得分）
NOEF_CATEGORY_WEIGHTS: Dict[str, float] = {
    "pitch":    0.25,   # 音高类
    "rhythm":   0.20,   # 节奏类
    "ornament": 0.25,   # 装饰音类
    "mode":     0.20,   # 调式类
    "style":    0.10,   # 整体风格类
}

# ORS 装饰音综合得分内部权重
ORS_WEIGHTS: Dict[str, float] = {
    "melodic_blend": 0.40,
    "rhythm_fit":    0.25,
    "density":       0.20,
    "evenness":      0.15,
}

# 全部 NOEF 指标名称（有序）—— 注意：这是【符号级】(token/MIDI) 指标名集合，
# 被 batch_evaluate_generated/_print_model_summary 等符号级流程遍历使用。
# 音频级指标(如 pitch_contour_correlation) 不在此集合，仅在 evaluate_noef 中产出。
NOEF_METRIC_NAMES: Tuple[str, ...] = (
    "pitch_accuracy",
    "pitch_stability",
    "rhythm_consistency",
    "tempo_accuracy",
    "ornament_score",
    "ornament_density",
    "mode_matching_score",
    "fad_distance",
    "melody_cosine_similarity",
)

# CSV 列名（英文 + 中文）
NOEF_CSV_HEADER: List[str] = [
    "song_id", "song_name", "model_name",
    "pitch_accuracy", "pitch_stability",
    "rhythm_consistency", "tempo_accuracy",
    "ornament_score", "ornament_density",
    "mode_matching_score",
    "fad_distance",
    "melody_cosine_similarity",
    "composite_score",
]

# 脚本路径
_SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR: str = os.path.join(_SCRIPT_DIR, "results")
_MIDI_BASE: str = os.path.join(_RESULTS_DIR, "generated_midi")
_AUDIO_BASE: str = os.path.join(_RESULTS_DIR, "generated_audio")
# GT(原始南音) 参照数据源 = BasicPitch AMT 转录 midi（output/BasicPitch/*.mid）。
# 2026-08-31 数据源规范：训练与评估的唯一 GT 数据源 = BasicPitch 对 dapu 大谱
# 音频的自动转录；structured_data/*_midi.json 为后处理产物，判定不准确，整体弃用。
_DATASET_BASE: str = os.path.join(_SCRIPT_DIR, "output", "BasicPitch")

# 两套模型子目录名
_MODEL_SUBDIRS: Dict[str, str] = {
    "BoYaTCN":    "boyatcn",
    "MusicMamba": "musicmamba",
}

# 2 首验证集曲目（dapu 大谱）→ 中文名
_VAL_SONG_IDS: List[str] = ["dapu14kouhuangtian", "dapu15wujinjiao"]
_SONG_CHINESE_NAMES: Dict[str, str] = {
    "dapu14kouhuangtian": "叩皇天",
    "dapu15wujinjiao":    "舞金蛟",
}

# 输出 CSV 路径
_SCORES_CSV: str = os.path.join(_RESULTS_DIR, "generate_scores.csv")


# ==============================================================================
# 二、MIDI 解析 — 生成 .mid → 符号序列
# ==============================================================================

def parse_midi_to_sequence(
    midi_path: str,
) -> Tuple[List[int], List[float], List[float], float]:
    """
    解析生成 MIDI 文件，提取音符序列、时值与真实 onset 时间。

    参数:
        midi_path: 生成的 .mid 文件路径
    返回:
        (tokens, durations, starts, total_sec)
        - tokens:      MIDI pitch 序列 (0~127)
        - durations:   时值序列 (秒)
        - starts:      note_on 起始时间序列 (秒，真实 onset)
        - total_sec:   音频总时长 (秒)

    说明: starts 用于计算 IOI（onset 间隔）。南音原曲中音符之间常含
    休止/撩拍间隙，仅凭 duration 无法还原真实节奏密度，必须用 onset 差。
    """
    if not _HAS_MIDO:
        return [], [], [], 0.0

    mid = mido.MidiFile(midi_path, clip=True)

    # 提取 tempo（默认 500000us = 120 BPM）
    tempo_us = 500000
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                tempo_us = msg.tempo
                break

    ticks_per_beat = mid.ticks_per_beat
    sec_per_tick = tempo_us / (ticks_per_beat * 1_000_000.0)

    tokens: List[int] = []
    durations: List[float] = []
    starts: List[float] = []
    # (channel, pitch) → FIFO 队列 [start_sec,...]。
    # 用队列而非单值：生成 midi 若含重叠同音（legato），同音多个 note_on
    # 会互相覆盖 start，导致配对错乱、音符丢失（曾 1025→792）。
    note_ons: Dict[Tuple[int, int], List[float]] = {}

    abs_ticks: int = 0
    for track in mid.tracks:
        for msg in track:
            abs_ticks += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                note_ons.setdefault((msg.channel, msg.note), []).append(
                    abs_ticks * sec_per_tick
                )
            elif msg.type in ("note_off",) or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                key = (msg.channel, msg.note)
                if key in note_ons and note_ons[key]:
                    start = note_ons[key].pop(0)  # FIFO 先起先收
                    end = abs_ticks * sec_per_tick
                    tokens.append(int(msg.note))
                    durations.append(float(max(0.01, end - start)))
                    starts.append(float(start))

    # 处理剩余未配对音符
    final_sec = abs_ticks * sec_per_tick
    for (_, pitch), queued_starts in note_ons.items():
        for start in queued_starts:
            tokens.append(int(pitch))
            durations.append(float(max(0.01, final_sec - start)))
            starts.append(float(start))

    total_sec = float(final_sec + 0.01)
    return tokens, durations, starts, total_sec


def load_ground_truth(song_id: str) -> Optional[Dict[str, Any]]:
    """
    从 BasicPitch AMT 转录（output/BasicPitch/{song_id}.mid）加载单首曲目 GT。

    数据源（2026-08-31 数据源规范）：
      评估的唯一 GT 参照 = BasicPitch 对 dapu 大谱音频的自动转录 midi
      （output/BasicPitch/*.mid）。structured_data/*_midi.json 为后处理产物，
      判定不准确，整体弃用，不再读取。速度/调式由 MIDI 音符客观推导（与
      dataset_nanyin.py 训练目标一致），MFCC 从 GT 主干渲染音频
      （results/gt_audio/）提取。

    参数:
        song_id: 曲目 ID（如 dapu14kouhuangtian）
    返回:
        包含以下键的字典，加载失败返回 None：
        - tokens:      GT token 序列 (MIDI pitch)
        - durations:   GT 时值序列 (秒)
        - pitches:     GT 音高值
        - tempo_bpm:   GT 速度 (BPM，由 onset 间隔客观估计)
        - mode_key:    GT 南音调式键名 (gong/shang/jue/zhi)
        - mode_name:   GT 调式中文名
        - mfcc_mean:   GT MFCC 均值向量 (13维，从 GT 渲染音频提取)
        - ornament_flags: GT 装饰音标记 (0/1 序列，客观规则推导)
        - audio_path:  GT 参考音频绝对路径 (results/gt_audio/{song}/{song}_gt.wav)
    """
    mid_path = os.path.join(_DATASET_BASE, f"{song_id}.mid")
    if not os.path.exists(mid_path):
        print(f"  [GT] BasicPitch 转录缺失: {mid_path}")
        return None

    # 1. 解析 BasicPitch midi → tokens/durations/pitches/starts
    gt_tokens: List[int] = []
    gt_durations: List[float] = []
    gt_pitches: List[float] = []
    gt_starts: List[float] = []
    try:
        import pretty_midi  # noqa: PLC0415

        pm = pretty_midi.PrettyMIDI(mid_path)
        for inst in pm.instruments:
            for n in inst.notes:
                p = int(n.pitch)
                gt_tokens.append(np.clip(p, 0, 127))
                gt_durations.append(float(max(0.01, n.end - n.start)))
                gt_pitches.append(float(p))
                gt_starts.append(float(n.start))
    except Exception:
        # 回退：mido 解析（parse_midi_to_sequence 返回真实 onset starts）
        tokens, durations, m_starts, _ = parse_midi_to_sequence(mid_path)
        if tokens:
            gt_tokens = list(tokens)
            gt_durations = list(durations)
            gt_pitches = [float(t) for t in tokens]
            if len(m_starts) == len(tokens):
                gt_starts = list(m_starts)
            else:
                gt_starts = [float(i) for i in range(len(tokens))]

    if not gt_tokens:
        print(f"  [GT] BasicPitch 转录无音符: {song_id}")
        return None

    # 2. 速度/调式：由 MIDI 音符客观推导（与 dataset_nanyin.py 训练目标一致）
    tempo_bpm = _estimate_tempo_bpm(gt_starts)
    mode_key, mode_name = _estimate_mode_from_pitch(gt_pitches)

    # 3. 装饰音标记：客观规则推导（短时值 + 邻音跳进，与训练目标一致）
    ornament_flags = _derive_ornament_flags(gt_durations, gt_pitches)

    # 4. 参考音频：GT 主干渲染（results/gt_audio/{song}/{song}_gt.wav）
    audio_path = os.path.join(_RESULTS_DIR, "gt_audio", song_id, f"{song_id}_gt.wav")
    if not os.path.exists(audio_path):
        audio_path = ""  # 不是必需的

    # 5. MFCC：从 GT 渲染音频提取（缺失时全零）
    mfcc_mean = _extract_mfcc_mean(audio_path)

    print(
        f"  [GT] {song_id}: {len(gt_tokens)} 音符, tempo={tempo_bpm:.1f}BPM, "
        f"mode={mode_key}({mode_name}), audio={audio_path or '无'}"
    )
    return {
        "tokens":          gt_tokens,
        "durations":       gt_durations,
        "starts":          list(gt_starts),   # 真实 onset（秒），供 IOI 节奏评估
        "pitches":         gt_pitches,
        "tempo_bpm":       tempo_bpm,
        "mode_key":        mode_key,
        "mode_name":       mode_name,
        "mfcc_mean":       mfcc_mean,
        "ornament_flags":  ornament_flags,
        "audio_path":      audio_path,
    }


def _estimate_tempo_bpm(start_list: Sequence[float]) -> float:
    """从音符起始时间估计速度（BPM）。
    与 dataset_nanyin.estimate_tempo_bpm 完全一致（保证训练/评估速度口径相同）。"""
    if len(start_list) < 2:
        return 120.0
    onsets = np.diff(np.sort(np.array(start_list, dtype=np.float64)))
    onsets = onsets[(onsets > 0.05) & (onsets < 3.0)]
    if len(onsets) == 0:
        return 120.0
    return float(np.clip(60.0 / np.median(onsets), 40.0, 240.0))


# 南音四调对应的宫调音级集合（与 dataset_nanyin._MODE_PENTATONIC 完全一致）
_GT_MODE_PENTATONIC: Dict[str, List[int]] = {
    "五空管":     [0, 2, 4, 7, 9],
    "五空四仪管": [7, 9, 11, 2, 4],
    "倍思管":     [2, 4, 6, 9, 11],
    "四空管":     [5, 7, 9, 0, 2],
}


def _estimate_mode_from_pitch(pitch_list: Sequence[float]) -> Tuple[str, str]:
    """从 MIDI 音高集合估计南音调式（与 dataset_nanyin.estimate_mode_from_pitch 一致）。
    返回 (mode_key, mode_name)；无法可靠估计时回退 (gong, 五空管)。"""
    if not pitch_list:
        return "gong", "五空管"
    pc = np.bincount(np.asarray(pitch_list, dtype=int) % 12, minlength=12).astype(np.float64)
    pc /= (pc.sum() + 1e-8)
    best_name, best_score = "五空管", -1.0
    for name, pcs in _GT_MODE_PENTATONIC.items():
        score = float(pc[pcs].sum())
        if score > best_score:
            best_score, best_name = score, name
    if best_score < 0.4:
        return "gong", "五空管"
    return TRADITIONAL_MODE_MAP.get(best_name, "gong"), best_name


def _derive_ornament_flags(
    durations: Sequence[float], pitches: Sequence[float]
) -> List[float]:
    """装饰音标记：短时值 + 邻音跳进（与 dataset_nanyin.build_target_dict 一致）。"""
    flags = np.zeros(len(durations), dtype=np.float32)
    n = len(durations)
    for i, d in enumerate(durations):
        if d < 0.15:
            prev_gap = abs(pitches[i] - pitches[i - 1]) if i > 0 else 99.0
            next_gap = abs(pitches[i] - pitches[i + 1]) if i + 1 < n else 99.0
            if prev_gap >= 3.0 or next_gap >= 3.0:
                flags[i] = 1.0
    return flags.tolist()


def _extract_mfcc_mean(audio_path: str) -> List[float]:
    """从 GT 渲染音频提取 13 维 MFCC 均值；音频缺失/解析失败返回全零。"""
    if not audio_path or not os.path.exists(audio_path) or not _HAS_LIBROSA:
        return [0.0] * 13
    try:
        import librosa  # noqa: PLC0415

        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        if y.size == 0:
            return [0.0] * 13
        mean = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1).tolist()
        if len(mean) < 13:
            mean += [0.0] * (13 - len(mean))
        return mean[:13]
    except Exception:
        return [0.0] * 13


def _find_json(song_dir: str, suffix: str) -> Optional[str]:
    """在曲目目录下匹配 *_&lt;suffix&gt;.json 文件。"""
    for fname in sorted(os.listdir(song_dir)):
        if fname.endswith(f"_{suffix}.json"):
            return os.path.join(song_dir, fname)
    return None


def resolve_mode_pcs(mode_pcs: Optional[Union[str, Sequence[int]]]) -> List[int]:
    """将调式参数统一解析为 pitch-class 列表。"""
    if mode_pcs is None:
        return NANYIN_MODES["gong"]
    if isinstance(mode_pcs, str):
        if mode_pcs in NANYIN_MODES:
            return NANYIN_MODES[mode_pcs]
        mapped = TRADITIONAL_MODE_MAP.get(mode_pcs)
        if mapped:
            return NANYIN_MODES[mapped]
        return NANYIN_MODES["gong"]
    return list(mode_pcs)


# ==============================================================================
# 三、音高类指标（0–100，越高越好）
# ==============================================================================

def calculate_pitch_accuracy(
    gen_tokens: Sequence[int],
    gt_tokens: Sequence[int],
    skip_first: int = 0,
) -> float:
    """
    pitch_accuracy — 音高准确率 (0–100)。

    定义：生成 token 序列与 ground-truth token 序列的逐位置匹配比例。
    match = sum(gen[i] == gt[i]) / min(N_gen, N_gt)

    文献：
      [Huang+18] Music Transformer (arXiv:1809.04281)
        在旋律生成评估中以 token-level accuracy 作为核心音高指标。
      [MelodyGLM] (arXiv:2309.10738)
        使用 teacher-forcing → argmax 生成，以 MIDI pitch 匹配率评估音高准确性。

    参数:
        gen_tokens: 生成音符 token 序列 (MIDI pitch, 0–127)
        gt_tokens:  ground-truth token 序列
        skip_first: 跳过序列前 N 个 token（排除 prompt 区域，默认 0=不跳过）
    返回:
        0–100 分值
    """
    gen = np.asarray(gen_tokens, dtype=int)
    gt  = np.asarray(gt_tokens, dtype=int)
    # 跳过 prompt 区域
    if skip_first > 0:
        gen = gen[skip_first:]
        gt  = gt[skip_first:]
    n = min(len(gen), len(gt))
    if n == 0:
        return 0.0
    match = int(np.sum(gen[:n] == gt[:n]))
    return 100.0 * float(match) / n


def calculate_pitch_stability(
    gen_tokens: Sequence[int],
) -> float:
    """
    pitch_stability — 腔调平稳度 (0–100)。

    定义：基于相邻音高跳动的归一化抖动。
    jitter = std(Δpitch) / mean(|pitch|)
    stability = 100 × (1 - jitter)

    南音特点在于旋律线条平稳，过大跳进意味着偏离传统腔调。

    文献：
      [Umbert+13] "Objective Evaluation of Singing Voice" (ISMIR 2013)
        该文使用 f0 jitter/shimmer 评估嗓音平稳度，此处适配为符号级音高抖动。
      [Serra+11] "Melodic Similarity: State of the Art" (IEEE TASLP 19(1), 2011)
        旋律轮廓平滑度作为风格特征之一。

    参数:
        gen_tokens: 生成音符 token 序列
    返回:
        0–100 分值
    """
    arr = np.asarray(gen_tokens, dtype=float)
    if len(arr) < 2:
        return 100.0

    diffs = np.abs(np.diff(arr))
    jitter = float(np.std(diffs)) / max(float(np.mean(np.abs(arr))), 1e-6)
    return max(0.0, min(100.0, 100.0 * (1.0 - jitter)))


# 保留旧名称（向后兼容）
calculate_pitch_entropy = calculate_pitch_stability


# ==============================================================================
# 三-B、旋律线质量指标（无参考自评估，0–100，越高越好）
#   t1 新增指标（2026-09-04）：transition_smoothness / melody_coherence
#   两者均只依赖生成 token 自身，不比对 GT，符合"无参考"口径。
# ==============================================================================

def calculate_transition_smoothness(
    gen_tokens: Sequence[int],
) -> float:
    """
    transition_smoothness — 相邻音过渡平滑度 (0–100, 符号级·无参考)。

    定义：对生成 token 序列计算相邻音高的绝对音程 step = |Δpitch|（半音），
    取均值 m = mean(step)，线性映射：
        smooth = 100 × clip(1 − (m − 2) / 6, 0, 1)
    即平均步长 ≤ 2 半音（级进/邻音）得满分；平均步长 ≥ 8 半音（大跳频繁）得 0。

    动机：南音"润腔"旋律以平稳级进与小跳为主，频繁大跳使过渡生硬、
    偏离传统腔韵。与 pitch_stability（关注跳动幅度的*波动一致性*）互补，
    本指标关注相邻过渡步长的*平均水平*。

    参数:
        gen_tokens: 生成音符 token 序列（MIDI pitch）
    返回:
        0–100 分值
    """
    arr = np.asarray(gen_tokens, dtype=float)
    if len(arr) < 3:
        return 100.0
    steps = np.abs(np.diff(arr))
    m = float(np.mean(steps))
    return max(0.0, min(100.0, 100.0 * (1.0 - (m - 2.0) / 6.0)))


def calculate_melody_coherence(
    gen_tokens: Sequence[int],
) -> float:
    """
    melody_coherence — 旋律轮廓连贯度 (0–100, 符号级·无参考)。

    定义：记 d1[n] = Δpitch（一阶差 = 相邻音高变化），d2[n] = Δd1（二阶差，
    即相邻两次音高变化的方向/幅度突变，反映轮廓"转折强度"）。取
    c = mean(|d2|)，映射：
        coherent = 100 × clip(1 − c / 6, 0, 1)
    即旋律轮廓平均每相邻三步的方向改变 ≤ 6 半音时视为连贯。

    动机：连贯的旋律轮廓表现为平滑弧线/阶梯（二阶差接近 0）；音高随机跳变
    或轮廓频繁"折断"会显著拉大二阶差。该指标与 transition_smoothness
    （一阶步长的平均水平）互补：前者看相邻过渡大小，后者看轮廓整体是否
    有机、可预期地连贯。

    参数:
        gen_tokens: 生成音符 token 序列（MIDI pitch）
    返回:
        0–100 分值
    """
    arr = np.asarray(gen_tokens, dtype=float)
    if len(arr) < 4:
        return 100.0
    d1 = np.diff(arr)
    d2 = np.diff(d1)
    c = float(np.mean(np.abs(d2)))
    return max(0.0, min(100.0, 100.0 * (1.0 - c / 6.0)))


# ==============================================================================
# 四、节奏类指标（0–100，越高越好）
# ==============================================================================

def _starts_to_ioi(
    starts: Sequence[float],
    skip_first: int = 0,
) -> np.ndarray:
    """从真实 onset 序列计算 IOI（音符起始间隔，秒），可选跳过最早 N 个音符。

    为什么节奏指标必须用 IOI 而非 duration：
      南音原曲（BasicPitch 转录）的音符之间常含休止/撩拍间隙，
      生成 MIDI（tokens_to_midi 接龙排布）的 onset 间隔 ≈ 自身时值。
      若只比 duration 分布，两者几乎相同 → rhythm/tempo 虚高，
      但人耳听到的"音符密度/速度"由 IOI 决定，故必须以 IOI 为准。

    实现细节：
      1. 必须先 np.sort 再 diff：BasicPitch 转录含多轨/同时发声音符，
         跨轨道拼接的 starts 非单调（曾导致 interval 从 1024 骤减到 737）。
         排序后 diff 得到时间有序的"发声音头事件流"。
      2. skip_first 对排序后的序列生效（跳过时间上最早的 N 个发声音头），
         与其它逐位指标的 skip_prompt 语义对齐。
    """
    arr = np.asarray(starts, dtype=float)
    if len(arr) < 2:
        return np.array([], dtype=float)
    arr = np.sort(arr)
    if skip_first > 0:
        arr = arr[skip_first:]
    if len(arr) < 2:
        return np.array([], dtype=float)
    ioi = np.diff(arr)
    return ioi[ioi > 0]


def _estimate_dominant_beat_unit(
    intervals: Sequence[float],
    lo: float = 0.08,
    hi: float = 4.0,
) -> float:
    """从生成 IOI 序列自动估计主导基本时值 u（秒）。

    无参考版 rhythm_consistency 需要先把生成片段的 onset 事件量化为
    “撩/拍”时值网格：南音节奏时值取自撩拍网格单位时间的整数倍层级
    （如 u、2u、4u、8u）。u 即该网格的基本格长。

    实现：在一维候选网格上扫描，使各 IOI 对 u 的整数倍量化误差
      cost(u) = mean_i |ioi_i − round(ioi_i / u)·u| / u
    最小的 u 即为主导基本时值（等价于以众数/聚类方式选取节拍基底，
    但比众数更稳健：即便时值层级不全落整数倍也能取到全局最优基底）。

    参数:
        intervals: 音符 onset 间隔序列 (秒, IOI)
        lo, hi:    候选基本时值的下限/上限 (秒)
    返回:
        主导基本时值 u（秒）；输入不足时返回中位数/2 的保守估计
    """
    vals = np.asarray(intervals, dtype=float)
    vals = vals[(vals >= lo) & (vals <= hi)]
    if len(vals) < 4:
        med = float(np.median(vals)) if len(vals) else 0.25
        return max(lo, min(hi, med / 2.0))
    med = float(np.median(vals))
    search_lo = max(lo, med / 8.0)
    search_hi = min(hi, med * 2.0)
    if search_hi <= search_lo:
        return float(search_lo)
    grid = np.linspace(search_lo, search_hi, 320)
    best_u, best_c = float(search_lo), float(np.inf)
    for u in grid:
        k = np.clip(np.round(vals / u), 1.0, 16.0)
        err = float(np.mean(np.abs(vals - k * u) / u))
        if err < best_c:
            best_c, best_u = err, u
    return best_u


def calculate_rhythm_consistency(
    gen_intervals: Sequence[float],
    skip_first: int = 0,
) -> float:
    """
    rhythm_consistency — 节奏一致性（无参考·自评估版，0–100）。

    定义（2026-09-04 无参考化）：
      不再与任何 GT/原曲 IOI 分布做 KS 比对，仅评估生成片段自身的
      节奏组织质量：
        consistency = 0.5 × IOI 稳定性 + 0.5 × 撩拍分布规整度

    计算步骤：
      1. 主导撩拍时值 u：对生成 IOI 做一维网格搜索，找出使各 IOI 近似
         为 u 整数倍的整体误差最小的基本时值（见 _estimate_dominant_beat_unit）。
      2. 乐句切分：IOI > 4·u 视为乐句间呼吸/停歇，把片段切成若干乐句
         （不足 2 个间隔的碎片句丢弃；若全部为长音则整段视作一句）。
      3. 逐乐句内部计算两个分量，再按句内间隔数加权汇总为 0–100：
         a. IOI 稳定性：句内相邻 IOI 的比值（大/小）应贴近简单整数比
            {1:1, 2:1, 3:1, 4:1}（南音节奏型由整倍层级构成），比值对
            最近整数比的偏离越大，说明相邻时值关系越“散拍”。
         b. 撩拍分布：句内音头相对句首的累计拍位 pos = cumsum(IOI)/u，
            统计 pos 偏离最近整“撩/拍”格位的平均幅度；该量衡量音头
            是否始终落在撩拍网格上、不随乐句推进累积漂移。

    说明：
      - 纯自评估不奖励“与原曲相似”，也不依赖任何 GT 数据；机械等时
        值序列会得高分，需与 tempo_accuracy / mode / 装饰音等指标联合解读。
      - 参数仍为 IOI（相邻 onset 之差）而非 duration：南音原曲含休止/
        撩拍间隙，仅比对 duration 会系统性高估节奏规整度。
      - GT 行（gt_refer）在该指标下不再恒为 100，而是真实南音自身的
        节奏规整度，作为“可达参照”而非“自比满分”。

    参数:
        gen_intervals: 生成音符 onset 间隔序列 (秒, IOI)
        skip_first:    跳过序列前 N 个元素（排除 prompt 区域，默认 0）
    返回:
        0–100 分值，越高表示生成片段自身的节奏组织越规整稳定
    """
    arr = np.asarray(gen_intervals, dtype=float)
    if skip_first > 0:
        arr = arr[skip_first:]
    arr = arr[arr > 0]
    if len(arr) < 8:
        return 0.0

    u = _estimate_dominant_beat_unit(arr)
    if u <= 0:
        return 0.0

    # 乐句切分：超过 4 个基本时值的间隔视为乐句间呼吸/停歇
    is_boundary = arr > 4.0 * u
    phrases: List[np.ndarray] = []
    cur: List[float] = []
    for b, x in zip(is_boundary, arr):
        if not b:
            cur.append(x)
        elif len(cur) >= 2:
            phrases.append(np.asarray(cur, dtype=float))
            cur = []
    if len(cur) >= 2:
        phrases.append(np.asarray(cur, dtype=float))
    if not phrases:  # 全为长音停歇、无有效句 → 整段视作一句
        phrases = [arr]

    scores: List[float] = []
    weights: List[float] = []
    for ph in phrases:
        if len(ph) < 2:
            continue
        # a) IOI 稳定性：相邻时值比值量化到 {1,2,3,4} 的干净度
        big = np.maximum(ph[:-1], ph[1:])
        small = np.minimum(ph[:-1], ph[1:])
        ratio = np.clip(big / np.maximum(small, 1e-9), 1.0, 4.0)
        q = np.clip(np.round(ratio), 1.0, 4.0)
        stab_err = float(np.mean(np.minimum(np.abs(ratio - q) / 0.5, 1.0)))
        stab = 100.0 * (1.0 - stab_err)

        # b) 撩拍分布：句内音头累计拍位贴合整数撩/拍格位的程度
        pos = np.cumsum(ph) / u
        resid = np.abs(pos - np.round(pos))  # ∈ [0, 0.5]
        grid_err = float(np.mean(resid / 0.5)) if len(pos) else 0.0
        grid = 100.0 * (1.0 - min(1.0, grid_err))

        scores.append(0.5 * stab + 0.5 * grid)
        weights.append(len(ph))

    if not scores:
        return 0.0
    total = float(sum(weights))
    if total <= 0:
        return 0.0
    out = float(np.average(scores, weights=weights))
    return max(0.0, min(100.0, out))


def calculate_tempo_accuracy(
    gen_intervals: Sequence[float],
    gt_intervals: Sequence[float],
    skip_first: int = 0,
) -> float:
    """
    tempo_accuracy — 速度准确率 (0–100)。

    定义：对两个序列分别计算平均 IOI 的匹配度。
    avg_ioi_gen = mean(gen_intervals)
    avg_ioi_gt  = mean(gt_intervals)
    accuracy = 100 × (1 - |ioi_gen - ioi_gt| / ioi_gt)

    注：参数必须为 IOI（相邻 onset 之差，秒）。原实现误用 duration
    均值充当"平均 IOI"，而生成接龙 MIDI 的 duration ≈ onset 间隔、
    GT 的 duration < onset 间隔（含休止），导致 tempo 虚高（96 vs 真实 57）。
    改用真实 onset 间隔后，速度差异（人耳听感"快"）才能被正确反映。

    文献：
      [McKinney+06] "Evaluation of Audio Beat Tracking" (JNMR 35(1), 2006)
        该文以检测 tempo 与 target tempo 的相对误差衡量准确率。

    参数:
        gen_intervals: 生成音符 onset 间隔序列 (秒, IOI)
        gt_intervals:  ground-truth onset 间隔序列 (秒, IOI)
        skip_first:   跳过序列前 N 个元素（排除 prompt 区域，默认 0=不跳过）
    返回:
        0–100 分值
    """
    gen = np.asarray(gen_intervals, dtype=float)
    gt  = np.asarray(gt_intervals, dtype=float)
    if skip_first > 0:
        gen = gen[skip_first:]
        gt  = gt[skip_first:]
    if len(gt) == 0:
        return 0.0

    avg_gen = float(np.mean(gen)) if len(gen) > 0 else 0.0
    avg_gt  = float(np.mean(gt))
    if avg_gt <= 0:
        return 0.0

    rel_err = abs(avg_gen - avg_gt) / avg_gt
    return max(0.0, min(100.0, 100.0 * (1.0 - rel_err)))


# ==============================================================================
# 五、装饰音类指标（0–100，越高越好）
# ==============================================================================

def calculate_ornament_density(
    gen_tokens: Sequence[int],
    threshold_semitones: float = 0.5,
) -> float:
    """
    ornament_density — 装饰音密度 (0–100)。

    定义：相邻音符半音变化的快速波动帧占比。
    density = 100 × count(|Δpitch| > threshold) / (N - 1)

    南音以滑音（yayin）、颤音（chanyin）、花音（huayin）为特征装饰手法，
    快速小幅度音高波动反映了装饰音的存在。

    文献：
      [Chordia+07] "Ornament Detection in Hindustani Music" (ISMIR 2007)
        该文以音高轨迹的微波动幅度检测装饰音事件。
      [Serra+11] "Melodic Similarity" (IEEE TASLP 19(1), 2011)
        旋律装饰密度作为风格区分特征。

    参数:
        gen_tokens:         生成音符 token 序列
        threshold_semitones: 装饰音判定阈值（半音）
    返回:
        0–100 分值
    """
    arr = np.asarray(gen_tokens, dtype=float)
    if len(arr) < 2:
        return 0.0

    diff = np.abs(np.diff(arr))
    rapid = int(np.sum(diff > threshold_semitones))
    return 100.0 * float(rapid) / max(len(diff), 1)


def calculate_ornament_score(
    gen_tokens: Sequence[int],
    gen_intervals: Sequence[float],
    skip_first: int = 0,
) -> float:
    """
    ornament_score (ORS) — 装饰音综合得分 (0–100)。

    ORS = 0.40 × melodic_blend + 0.25 × rhythm_fit + 0.20 × density + 0.15 × evenness

    其中：
      - melodic_blend ≈ pitch_stability（腔调平稳度→装饰音融合自然度）
      - rhythm_fit     ≈ rhythm_consistency（节奏适配，无参考自评估：
                         IOI 稳定性 + 撩拍网格规整度，不再依赖 GT 分布）
      - density        ≈ ornament_density（装饰音密度）
      - evenness       ≈ 1 - std(ornament_changes) / mean(ornament_changes)

    文献：
      [Chordia+07] "Ornament Detection in Hindustani Music" (ISMIR 2007)
        该文提出多维度融合的装饰音质量评估框架。
      [Serra+11] "Melodic Similarity" (IEEE TASLP 19(1))
        装饰融合度是风格一致性评估的重要组成部分。

    参数:
        gen_tokens:     生成 token 序列
        gen_intervals:  生成音符 onset 间隔序列 (秒, IOI)
        skip_first:     跳过序列前 N 个元素（默认 0）
    返回:
        0–100 分值
    """
    melodic_blend = calculate_pitch_stability(gen_tokens)
    rhythm_fit    = calculate_rhythm_consistency(gen_intervals, skip_first=skip_first)
    density       = calculate_ornament_density(gen_tokens)

    # evenness: 装饰音变化幅度的均匀性
    arr = np.asarray(gen_tokens, dtype=float)
    evenness = 100.0
    if len(arr) >= 3:
        diff = np.abs(np.diff(arr))
        rapid = diff[diff > 0.5]
        if len(rapid) >= 2:
            ev = 1.0 - float(np.std(rapid)) / (float(np.mean(rapid)) + 1e-6)
            evenness = 100.0 * max(0.0, min(1.0, ev))

    return (
        ORS_WEIGHTS["melodic_blend"] * melodic_blend
        + ORS_WEIGHTS["rhythm_fit"]    * rhythm_fit
        + ORS_WEIGHTS["density"]       * density
        + ORS_WEIGHTS["evenness"]      * evenness
    )


# ==============================================================================
# 六、调式类指标（0–100，越高越好）
# ==============================================================================

def calculate_mode_matching_score(
    gen_tokens: Sequence[int],
    mode_pcs: Optional[Union[str, Sequence[int]]] = None,
) -> float:
    """
    mode_matching_score — 调式匹配度 (0–100)。

    定义：生成 token 序列中落在指定南音调式 pitch-class 集合内的音符占比。
    score = 100 × count(note in mode_pcs) / total_notes

    南音四种调式各对应 5 个音级（五声音阶），生成旋律应严格遵守该约束。

    文献：
      [Wiggins+93] "Tonal Center Detection" (Computing in Musicology, 1993)
        该文以 pitch-class 集合覆盖度衡量调式一致性。
      南音管门体系 (Nanyin Guanmen System)
        五空管/四空管/倍思管/尺调 各有一套固定的五声音阶，偏离视为失范。

    参数:
        gen_tokens: 生成 token 序列 (MIDI pitch)
        mode_pcs:   目标调式（键名/中文名/pitch-class列表），默认 gong
    返回:
        0–100 分值
    """
    pcs = resolve_mode_pcs(mode_pcs)
    arr = np.asarray(gen_tokens, dtype=int)
    if len(arr) == 0:
        return 0.0

    pitch_classes = arr % 12
    in_mode = int(np.sum(np.isin(pitch_classes, pcs)))
    return 100.0 * float(in_mode) / len(arr)


# 向后兼容别名
calculate_mode_consistency = calculate_mode_matching_score

def calculate_ornament_evenness(
    gen_tokens: Sequence[int],
    threshold_semitones: float = 0.5,
) -> float:
    """
    [向后兼容] 装饰音均匀度 (0–100)。

    基于快速音高变化幅度的分布稳定性。
    """
    arr = np.asarray(gen_tokens, dtype=float)
    if len(arr) < 3:
        return 100.0
    diff = np.abs(np.diff(arr))
    rapid = diff[diff > threshold_semitones]
    if len(rapid) < 2:
        return 100.0
    ev = 1.0 - float(np.std(rapid)) / (float(np.mean(rapid)) + 1e-6)
    return 100.0 * max(0.0, min(1.0, ev))


# 向后兼容：旧 audio 加载函数
def load_audio(path: str, sr: int = 22050, duration: Optional[float] = None
               ) -> Tuple[np.ndarray, int]:
    """[向后兼容] 加载音频，与旧 evaluation.py 接口一致。"""
    return _load_audio(path, sr=sr, duration=duration)


def load_reference_profile(audio_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    [向后兼容] 从 WAV 同目录加载 features.json 参考统计特征。

    新代码请使用 load_ground_truth(song_id)。
    """
    if not audio_path or not os.path.exists(audio_path):
        return None
    song_dir = os.path.dirname(audio_path)
    feat_path = _find_json(song_dir, "features")
    if feat_path is None:
        return None
    try:
        with open(feat_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        af = data.get("audio_features", {})
        f0s = af.get("f0_statistics", {})
        td  = af.get("tempo_detection", {})
        md  = af.get("mode_detection", {})
        bg  = data.get("background_info", {})
        mf  = af.get("mfcc_features", {})
        return {
            "f0_mean_hz": f0s.get("f0_mean_hz"),
            "f0_std_hz": f0s.get("f0_std_hz"),
            "f0_min_hz": f0s.get("f0_min_hz"),
            "f0_max_hz": f0s.get("f0_max_hz"),
            "tempo_bpm": td.get("tempo_bpm"),
            "traditional_mode": md.get("traditional_mode") or bg.get("traditional_mode"),
            "estimated_key": md.get("estimated_key"),
            "mfcc_mean": mf.get("mfcc_mean"),
        }
    except Exception:
        return None


def calculate_mfcc_stats(y: np.ndarray, sr: int, n_mfcc: int = 13
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """[向后兼容] 提取 MFCC 均值与协方差。"""
    return _extract_mfcc_stats(y, sr, n_mfcc)


# ==============================================================================
# 七、整体风格类指标
# ==============================================================================

# --- 7.1 FAD 距离（音频级，需要 librosa）--- #

def _load_audio(path: str, sr: int = 22050, duration: Optional[float] = None
                ) -> Tuple[np.ndarray, int]:
    """加载音频文件并重采样。"""
    if not _HAS_LIBROSA:
        # 纯 numpy 降级：读取原始 PCM（仅支持 WAV 16-bit mono）
        try:
            import wave
            with wave.open(path, "rb") as wf:
                nch = wf.getnchannels()
                sw  = wf.getsampwidth()
                fs  = wf.getframerate()
                nf  = wf.getnframes()
                raw = wf.readframes(nf)
            dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
            data = np.frombuffer(raw, dtype=dtype_map.get(sw, np.int16)).astype(np.float32)
            if nch > 1:
                data = data.reshape(-1, nch).mean(axis=1)  # mono
            data /= (2 ** (8 * sw - 1))
            if fs != sr:
                from scipy.signal import resample
                n_out = int(len(data) * sr / fs)
                resampled = resample(data, n_out)
                data = np.asarray(resampled, dtype=np.float32)
            if duration:
                n = int(duration * sr)
                data = data[:n]
            return data, sr
        except Exception:
            dummy = np.zeros(sr, dtype=np.float32)
            return dummy, sr

    y, _ = librosa.load(path, sr=sr, duration=duration)
    return y, sr


def _extract_mfcc_stats(y: np.ndarray, sr: int, n_mfcc: int = 13
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """提取 MFCC 均值向量与协方差矩阵。"""
    if not _HAS_LIBROSA:
        # 降级：返回零向量（避免崩溃）
        return np.zeros(n_mfcc, dtype=np.float64), np.eye(n_mfcc, dtype=np.float64)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mean = np.mean(mfcc, axis=1)
    cov  = np.cov(mfcc)
    return mean, cov


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Fréchet 距离核心公式。

    d² = ||μ1 - μ2||² + Tr(Σ1 + Σ2 - 2·sqrt(Σ1·Σ2))

    文献：
      [Kilgour+19] "Fréchet Audio Distance" (arXiv:1812.08466)
        该文将图像生成中的 FID 指标迁移至音频领域，使用 MFCC 均值/协方差
        计算两个音频集合的 Fréchet 距离以量化生成质量。
    """
    diff = mu1.astype(np.float64) - mu2.astype(np.float64)
    covmean_raw = cast(np.ndarray, sqrtm(sigma1.dot(sigma2) + eps * np.eye(sigma1.shape[0])))
    if np.iscomplexobj(covmean_raw):
        covmean = np.real(covmean_raw).astype(np.float64)
    else:
        covmean = covmean_raw.astype(np.float64)
    trace_val = float(np.trace(sigma1 + sigma2 - 2.0 * covmean))
    return float(np.dot(diff, diff) + trace_val)


def calculate_fad_score(
    ref_audio_path: str,
    gen_audio_path: str,
    sr: int = 22050,
    duration: Optional[float] = None,
) -> float:
    """
    fad_distance — Fréchet Audio Distance 映射得分 (0–100)。

    计算参考 WAV 与生成 WAV 的 MFCC 分布 Fréchet 距离，再通过
    指数映射转换为 0–100 评分：
        fad_score = 100 × exp(-FAD / scale)

    数值越高表示生成音频与参考风格越接近。

    文献：
      [Kilgour+19] Fréchet Audio Distance (arXiv:1812.08466)

    参数:
        ref_audio_path:  参考音频 (GT WAV) 路径
        gen_audio_path:  生成音频 (合成 WAV) 路径
        sr:              采样率
        duration:        分析时长 (秒)，None=全曲
    返回:
        0–100 分值
    """
    if not os.path.exists(ref_audio_path) or not os.path.exists(gen_audio_path):
        return 0.0

    y_ref, _ = _load_audio(ref_audio_path, sr=sr, duration=duration)
    y_gen, _ = _load_audio(gen_audio_path, sr=sr, duration=duration)

    mean_ref, cov_ref = _extract_mfcc_stats(y_ref, sr)
    mean_gen, cov_gen = _extract_mfcc_stats(y_gen, sr)

    fad = calculate_frechet_distance(mean_ref, cov_ref, mean_gen, cov_gen)
    # FAD → 0–100: 使用指数映射，scale=20 使典型 FAD(~10) 映射到 ~60 分
    fad_score = 100.0 * math.exp(-fad / 20.0)
    return max(0.0, min(100.0, fad_score))


# --- 7.2 旋律余弦相似度（符号级）--- #

def calculate_melody_cosine_similarity(
    gen_tokens: Sequence[int],
    gt_tokens: Sequence[int],
    skip_first: int = 0,
) -> float:
    """
    melody_cosine_similarity — 旋律余弦相似度 (0–100)。

    定义：基于 pitch-class 分布直方图（12 维 bag-of-notes）计算余弦相似度，
    然后缩放至 0–100。

    文献：
      [Yang+12] "Music Similarity and Retrieval" (ACM Computing Surveys 44(2), 2012)
        该文综述了基于 bag-of-features 和余弦距离的音乐相似度方法。
      [Serra+11] "Melodic Similarity" (IEEE TASLP 19(1))
        旋律轮廓分布作为整体风格相似度的关键对比维度。

    参数:
        gen_tokens: 生成 token 序列
        gt_tokens:  ground-truth token 序列
        skip_first: 跳过序列前 N 个 token（排除 prompt 区域，默认 0=不跳过）
    返回:
        0–100 分值
    """
    gen = np.asarray(gen_tokens, dtype=int)
    gt  = np.asarray(gt_tokens, dtype=int)
    if skip_first > 0:
        gen = gen[skip_first:]
        gt  = gt[skip_first:]

    # 构建 12 维 pitch-class 直方图
    hist_gen = np.bincount(gen % 12, minlength=12).astype(np.float64)
    hist_gt  = np.bincount(gt  % 12, minlength=12).astype(np.float64)

    # L2 归一化
    norm_gen = np.linalg.norm(hist_gen) + 1e-8
    norm_gt  = np.linalg.norm(hist_gt)  + 1e-8
    hist_gen_n = hist_gen / norm_gen
    hist_gt_n  = hist_gt  / norm_gt

    # 余弦相似度 [-1, 1] → [0, 100]
    cos_sim = float(np.dot(hist_gen_n, hist_gt_n))
    return max(0.0, min(100.0, 100.0 * (cos_sim + 1.0) / 2.0))


# 向后兼容（音频级风格余弦相似度）
def calculate_style_consistency(
    gen_audio_path: str,
    ref_audio_path: str,
    sr: int = 22050,
    duration: Optional[float] = None,
) -> float:
    """
    style_consistency — 音频级风格余弦相似度 (0–100, 向后兼容)。

    使用生成 WAV 与参考 WAV 的 MFCC 均值向量计算余弦相似度。
    """
    if not os.path.exists(ref_audio_path) or not os.path.exists(gen_audio_path):
        return 0.0

    y_ref, _ = _load_audio(ref_audio_path, sr=sr, duration=duration)
    y_gen, _ = _load_audio(gen_audio_path, sr=sr, duration=duration)
    mean_ref, _ = _extract_mfcc_stats(y_ref, sr)
    mean_gen, _ = _extract_mfcc_stats(y_gen, sr)

    if _HAS_SKLEARN:
        sim = _cosine_similarity(mean_ref.reshape(1, -1), mean_gen.reshape(1, -1))
        cos = float(sim[0, 0])
    else:
        nr = np.linalg.norm(mean_ref) + 1e-8
        ng = np.linalg.norm(mean_gen) + 1e-8
        cos = float(np.dot(mean_ref, mean_gen) / (nr * ng))
    return max(0.0, min(100.0, 100.0 * (cos + 1.0) / 2.0))


# --- 7.3 南音 f0 风格画像 + 音高轮廓相似度（t2 新增指标，音频级）--- #

# 11 首南音原始音频目录（需求：全部原始 dapu 无人声音频放在 test_audio/）
_NANYIN_AUDIO_DIR: str = os.path.join(_SCRIPT_DIR, "test_audio")
# f0 画像缓存（避免每次评估重算 11 首长音频，音频未变则长期复用）
_F0_PROFILE_NPZ: str = os.path.join(_SCRIPT_DIR, "output", "nanyin_f0_profile.npz")
# 每首南音音频取前 N 秒参与画像统计（秒）
_F0_PROFILE_SECONDS: float = 45.0
# f0 提取参数（与 evaluate_noef 一致）
_F0_PROFILE_FMIN: float = 82.407  # C2
_F0_PROFILE_FMAX: float = 1046.50  # C6


def _nanyin_audio_paths() -> List[str]:
    """返回 test_audio/ 下全部南音 dapu*.wav 路径（应恰为 11 首）。"""
    if not os.path.isdir(_NANYIN_AUDIO_DIR):
        return []
    pats = sorted(glob.glob(os.path.join(_NANYIN_AUDIO_DIR, "dapu*.wav")))
    pats += sorted(glob.glob(os.path.join(_NANYIN_AUDIO_DIR, "dapu*.mp3")))
    return pats


def _build_nanyin_f0_profile(force: bool = False) -> Dict[str, Any]:
    """
    从 11 首南音音频提取 f0 轮廓，统计均值/方差 → "南音 f0 风格画像"。

    流程（2026-09-04 用户定义口径）：
      1. 遍历 test_audio/ 下全部 dapu*.wav（11 首：9 训练 + 2 验证）；
      2. 每首截取前 _F0_PROFILE_SECONDS 秒，用 librosa.pyin (C2–C6)
         提取 voiced 帧的 f0（Hz），转 MIDI 音高；
      3. 合并 11 首全部 voiced 帧 → μ_p（均值）、σ_p（标准差）。

    画像会缓存为 npz，音频未变时后续评估直接读取，不重复计算。
    """
    if not _HAS_LIBROSA:
        return {"mu_midi": 0.0, "sigma_midi": 0.0, "n_frames": 0, "src": "no_librosa"}

    if not force and os.path.exists(_F0_PROFILE_NPZ):
        try:
            data = np.load(_F0_PROFILE_NPZ)
            return {
                "mu_midi": float(data["mu_midi"]),
                "sigma_midi": float(data["sigma_midi"]),
                "n_frames": int(data["n_frames"]),
                "src": "cache",
            }
        except Exception:
            pass

    paths = _nanyin_audio_paths()
    if not paths:
        return {"mu_midi": 0.0, "sigma_midi": 0.0, "n_frames": 0, "src": "empty"}

    all_midi: List[float] = []
    for p in paths:
        try:
            y, sr = _load_audio(p, sr=22050, duration=_F0_PROFILE_SECONDS)
            f0, voiced_flags, _ = librosa.pyin(
                y,
                fmin=_F0_PROFILE_FMIN,
                fmax=_F0_PROFILE_FMAX,
                sr=sr,
            )
            voiced = f0[voiced_flags]
            all_midi.extend(float(v) for v in librosa.hz_to_midi(voiced))
        except Exception:
            continue
    if len(all_midi) < 50:
        return {"mu_midi": 0.0, "sigma_midi": 0.0, "n_frames": len(all_midi), "src": "empty"}

    arr = np.asarray(all_midi, dtype=float)
    profile = {
        "mu_midi": float(np.mean(arr)),
        "sigma_midi": float(np.std(arr)),
        "n_frames": int(len(arr)),
        "src": f"built_from_{len(paths)}_songs",
    }
    try:
        os.makedirs(os.path.dirname(_F0_PROFILE_NPZ), exist_ok=True)
        np.savez(
            _F0_PROFILE_NPZ,
            mu_midi=profile["mu_midi"],
            sigma_midi=profile["sigma_midi"],
            n_frames=profile["n_frames"],
        )
    except Exception:
        pass
    return profile


def _get_nanyin_f0_profile() -> Dict[str, Any]:
    """模块级惰性缓存版 _build_nanyin_f0_profile。"""
    global _F0_PROFILE
    if _F0_PROFILE is None:
        _F0_PROFILE = _build_nanyin_f0_profile(force=False)
    return _F0_PROFILE


def calculate_pitch_contour_correlation(
    gen_midi: Sequence[float],
    profile: Optional[Dict[str, Any]] = None,
) -> float:
    """
    pitch_contour_correlation — 生成音频 f0 轮廓与南音 f0 风格画像的相似度
    (0–1, t2 音频级·无逐点比对)。

    口径（2026-09-04 用户定义）：
      1. 从 11 首南音音频（test_audio/dapu*.wav）提取 f0 轮廓，
         统计均值 μ_p 与标准差 σ_p → "南音 f0 风格画像"；
      2. 提取生成音频的 f0 轮廓，得 μ_g / σ_g；
      3. 相似度 = exp(−d)，其中
            d = |μ_g − μ_p| / σ_p  +  |log(σ_g / σ_p)|
         均值偏离按南音自身 f0 波动 σ_p 归一化；方差差异取对数比。
      数值越接近 1 = 生成音高轮廓的音域与波动越贴近真实南音。

    与 Tier1 同名/相邻指标的分工（文档口径）：
      - Tier1 符号级 melody_coherence / transition_smoothness：在 token 上算；
      - Tier2 音频级 pitch_contour_correlation：在渲染 WAV 的 f0 上算，
        且参照对象是"11 首南音的整体 f0 画像"而非逐曲 GT 比对。

    参数:
        gen_midi: 生成音频 voiced 帧的 MIDI 音高序列（Hz 转 MIDI）
        profile:  南音 f0 风格画像字典；None 时自动加载（惰性构建缓存）
    返回:
        0–1 分值
    """
    arr = np.asarray(gen_midi, dtype=float)
    if arr.size < 20:
        return 0.0
    if profile is None:
        profile = _get_nanyin_f0_profile()
    mu_p = float(profile.get("mu_midi", 0.0))
    sigma_p = float(profile.get("sigma_midi", 0.0))
    if sigma_p <= 1e-3:
        return 0.0

    mu_g = float(np.mean(arr))
    sigma_g = float(np.std(arr))
    d_mu = abs(mu_g - mu_p) / sigma_p
    d_sigma = abs(math.log(max(sigma_g, 1e-6) / sigma_p))
    score = math.exp(-(d_mu + d_sigma))
    return max(0.0, min(1.0, score))


_F0_PROFILE: Optional[Dict[str, Any]] = None


# ==============================================================================
# 八、单首曲目完整评估
# ==============================================================================

def evaluate_single_song(
    gen_tokens: Sequence[int],
    gen_durations: Sequence[float],
    gt: Dict[str, Any],
    gen_audio_path: str = "",
    skip_prompt: int = PROMPT_LEN,
    gen_starts: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """
    对单首曲目的生成结果计算全部 9 项 NOEF 指标。

    参数:
        gen_tokens:     生成 token 序列
        gen_durations:  生成时值序列 (秒)
        gen_starts:     生成音符真实 onset (秒)；缺省时退化为按 durations
                        累积的接龙 onset（仅兼容旧调用，无法还原休止）
        gt:             load_ground_truth() 返回的真值字典（须含 "starts"）
        gen_audio_path: 生成 WAV 路径（用于 FAD），空字符串则跳过
        skip_prompt:    跳过生成/参照序列开头的 prompt 区域音符数
                        （默认 PROMPT_LEN=16；设为 0 则不跳过，
                        用于 GT-vs-GT 天花板评估）
    返回:
        {metric_name: score}  全部 0–100 分值
    """
    results: Dict[str, float] = {}

    # ---- 节奏（IOI = 相邻 onset 之差，包含音符间休止/撩拍间隙）----
    # 生成 MIDI 为接龙排布时 onset≈时值（无 gap），GT 的 onset 含真实 gap；
    # 仅凭 duration 序列无法区分二者 → 必须用真实 onset（starts）。
    if gen_starts is not None and len(gen_starts) == len(gen_tokens):
        gen_onsets = np.asarray(gen_starts, dtype=float)
    else:  # 兼容旧调用：durations 接龙近似 onset
        gen_onsets = np.cumsum(np.maximum(np.asarray(gen_durations, dtype=float), 0.0))
    if gt.get("starts") is not None and len(gt["starts"]) == len(gt["tokens"]):
        gt_onsets = np.asarray(gt["starts"], dtype=float)
    else:  # 兼容旧 GT dict：durations 接龙近似 onset
        gt_onsets = np.cumsum(np.maximum(np.asarray(gt["durations"], dtype=float), 0.0))

    gen_iois = _starts_to_ioi(gen_onsets, skip_prompt)
    gt_iois  = _starts_to_ioi(gt_onsets, skip_prompt)

    # 音高类
    # [遗留] pitch_accuracy 已于 2026-09-04 从 t1 指标体系删除（点对点指标），
    # 仅保留计算以兼容旧接口（evaluation.py / batch_evaluate_generated）。
    results["pitch_accuracy"] = calculate_pitch_accuracy(gen_tokens, gt["tokens"], skip_first=skip_prompt)
    results["pitch_stability"] = calculate_pitch_stability(gen_tokens)
    # 旋律线质量类（无参考，2026-09-04 t1 新增；与 pitch_stability 同口径：
    # 作用于整条 token 序列，prompt 区保留以同时考察"续写衔接"的过渡质量）
    results["transition_smoothness"] = calculate_transition_smoothness(gen_tokens)
    results["melody_coherence"] = calculate_melody_coherence(gen_tokens)

    # 节奏类（IOI 已在前面按 skip_prompt 裁好，此处不再重复裁）
    # rhythm_consistency 为无参考自评估（2026-09-04）：不再比对 GT IOI 分布，
    # 只评估生成片段自身节奏组织；tempo_accuracy 仍以 GT 平均速度为南音速度参照。
    results["rhythm_consistency"] = calculate_rhythm_consistency(gen_iois)
    results["tempo_accuracy"] = calculate_tempo_accuracy(gen_iois, gt_iois)

    # 装饰音类（rhythm_fit 分量同样基于生成 IOI 的无参考自评估）
    results["ornament_score"] = calculate_ornament_score(gen_tokens, gen_iois)
    results["ornament_density"] = calculate_ornament_density(gen_tokens)

    # 调式类
    results["mode_matching_score"] = calculate_mode_matching_score(
        gen_tokens, gt["mode_key"]
    )

    # 整体风格类
    if gen_audio_path and os.path.exists(gen_audio_path) and gt.get("audio_path"):
        results["fad_distance"] = calculate_fad_score(
            gt["audio_path"], gen_audio_path
        )
    else:
        results["fad_distance"] = 0.0

    # [遗留] melody_cosine_similarity 已于 2026-09-04 从 t1 指标体系删除（点对点指标），
    # 仅保留计算以兼容旧接口（evaluation.py / batch_evaluate_generated）。
    results["melody_cosine_similarity"] = calculate_melody_cosine_similarity(
        gen_tokens, gt["tokens"], skip_first=skip_prompt
    )

    return results


def compute_composite_score(metrics: Dict[str, float]) -> float:
    """
    根据 NOEF 五大类权重计算加权综合得分 (0–100)。

    各类内部取已有指标均值。
    """
    pitch_vals = [metrics.get("pitch_accuracy", 0), metrics.get("pitch_stability", 0)]
    pitch = float(np.mean(pitch_vals))

    rhythm_vals = [metrics.get("rhythm_consistency", 0), metrics.get("tempo_accuracy", 0)]
    rhythm = float(np.mean(rhythm_vals))

    orn_vals = [metrics.get("ornament_score", 0), metrics.get("ornament_density", 0)]
    ornament = float(np.mean(orn_vals))

    mode = metrics.get("mode_matching_score", 0)

    style_vals = []
    if "melody_cosine_similarity" in metrics:
        style_vals.append(metrics["melody_cosine_similarity"])
    if "fad_distance" in metrics:
        style_vals.append(metrics["fad_distance"])
    style = float(np.mean(style_vals)) if style_vals else 0.0

    return (
        NOEF_CATEGORY_WEIGHTS["pitch"]    * pitch
        + NOEF_CATEGORY_WEIGHTS["rhythm"]   * rhythm
        + NOEF_CATEGORY_WEIGHTS["ornament"] * ornament
        + NOEF_CATEGORY_WEIGHTS["mode"]     * mode
        + NOEF_CATEGORY_WEIGHTS["style"]    * style
    )


# ==============================================================================
# 九、批量评估 + 输出 CSV
# ==============================================================================

def batch_evaluate_generated(
    midi_base: str = _MIDI_BASE,
    audio_base: str = _AUDIO_BASE,
    output_csv: str = _SCORES_CSV,
    model_subdirs: Optional[Dict[str, str]] = None,
    song_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    批量评估全部生成 MIDI 文件，自动对比 GT 标注，输出 CSV。

    流程：
      1. 遍历 song_ids 中每首曲目，从数据集加载 ground-truth
      2. 遍历 model_subdirs 中各模型子目录，读入 {song_id}_{model}.mid
      3. 计算全部 9 项 NOEF 指标 + composite_score
      4. 按行写入 output_csv（含中英文表头）

    参数:
        midi_base:     生成 MIDI 根目录（默认: results/generated_midi/）
        audio_base:    生成 WAV 根目录（默认: results/generated_audio/）
        output_csv:    输出 CSV 路径
        model_subdirs: 模型名 → 子目录名 映射
        song_ids:      待评估曲目 ID 列表
    返回:
        records:       每行评估记录的字典列表
    """
    if model_subdirs is None:
        model_subdirs = _MODEL_SUBDIRS
    if song_ids is None:
        song_ids = _VAL_SONG_IDS

    # 确保输出目录
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # 写入 CSV 表头
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(NOEF_CSV_HEADER)

    records: List[Dict[str, Any]] = []
    total = len(song_ids) * len(model_subdirs)

    print("=" * 70)
    print("NOEF 南音客观评估 — 批量评估")
    print("=" * 70)
    print(f"  曲目数: {len(song_ids)}")
    print(f"  模型数: {len(model_subdirs)}")
    print(f"  总计评估: {total} 次")
    print(f"  输出 CSV: {output_csv}")
    print()

    idx = 0
    for model_name, subdir in model_subdirs.items():
        midi_dir  = os.path.join(midi_base,  subdir)
        audio_dir = os.path.join(audio_base, subdir)

        for song_id in song_ids:
            idx += 1
            song_name = _SONG_CHINESE_NAMES.get(song_id, song_id)
            print(f"[{idx}/{total}] 评估: {song_name} ({song_id}) / {model_name}")

            # ---- 加载 GT ----
            gt = load_ground_truth(song_id)
            if gt is None:
                print(f"  ⚠ 跳过：GT 加载失败")
                continue

            # ---- 解析生成 MIDI ----
            midi_filename = f"{song_id}_{model_name.lower()}.mid"
            midi_path = os.path.join(midi_dir, midi_filename)
            if not os.path.exists(midi_path):
                print(f"  ⚠ 跳过：生成 MIDI 不存在 ({midi_path})")
                # 仍写一行空记录（方便检查遗漏）
                _write_empty_row(output_csv, song_id, song_name, model_name)
                continue

            gen_tokens, gen_durations, gen_starts, _ = parse_midi_to_sequence(midi_path)
            if not gen_tokens:
                print(f"  ⚠ 跳过：MIDI 解析为空")
                _write_empty_row(output_csv, song_id, song_name, model_name)
                continue

            # ---- 生成 WAV 路径（用于 FAD）----
            wav_filename = f"{song_id}_{model_name.lower()}.wav"
            gen_audio_path = os.path.join(audio_dir, wav_filename)

            # ---- 计算全部指标 ----
            metrics = evaluate_single_song(
                gen_tokens=gen_tokens,
                gen_durations=gen_durations,
                gen_starts=gen_starts,
                gt=gt,
                gen_audio_path=gen_audio_path,
            )
            composite = compute_composite_score(metrics)

            # ---- 打印各指标 ----
            _print_song_metrics(song_name, model_name, gen_tokens, metrics, composite, gt)

            # ---- 写入 CSV ----
            row = {
                "song_id":        song_id,
                "song_name":      song_name,
                "model_name":     model_name,
                **metrics,
                "composite_score": round(composite, 2),
            }
            _append_csv_row(output_csv, row)
            records.append(row)

    # 汇总
    print(f"\n{'=' * 70}")
    print(f"批量评估完成: {len(records)}/{total} 首")
    _print_model_summary(records, model_subdirs)
    print(f"\n结果已写入: {output_csv}")
    print(f"{'=' * 70}")

    return records


def _print_song_metrics(
    song_name: str,
    model_name: str,
    gen_tokens: List,
    metrics: Dict[str, float],
    composite: float,
    gt: Dict,
):
    """打印单首曲目各指标结果。"""
    sep = "-" * 56
    print(f"  {sep}")
    gt_mode = gt.get("mode_name", "?")
    gt_n = len(gt.get("tokens", []))
    gen_n = len(gen_tokens)
    print(f"  {song_name} | {model_name} | GT seq={gt_n}, Gen seq={gen_n}, GT 调式={gt_mode}")
    print(f"  {'指标':<28s} {'分值':>8s}   ('音高类' 下同)")
    print(f"  {sep}")
    print(f"  {'1. pitch_accuracy':.<40s} {metrics['pitch_accuracy']:>6.2f}")
    print(f"  {'2. pitch_stability':.<40s} {metrics['pitch_stability']:>6.2f}")
    print(f"  {'3. rhythm_consistency':.<40s} {metrics['rhythm_consistency']:>6.2f}")
    print(f"  {'4. tempo_accuracy':.<40s} {metrics['tempo_accuracy']:>6.2f}")
    print(f"  {'5. ornament_score':.<40s} {metrics['ornament_score']:>6.2f}")
    print(f"  {'6. ornament_density':.<40s} {metrics['ornament_density']:>6.2f}")
    print(f"  {'7. mode_matching_score':.<40s} {metrics['mode_matching_score']:>6.2f}")
    print(f"  {'8. fad_distance':.<40s} {metrics['fad_distance']:>6.2f}")
    print(f"  {'9. melody_cosine_similarity':.<40s} {metrics['melody_cosine_similarity']:>6.2f}")
    print(f"  {sep}")
    print(f"  {'★ composite_score':.<40s} {composite:>6.2f}")


def _print_model_summary(records: List[Dict], model_subdirs: Dict[str, str]):
    """打印两套模型的对比汇总。"""
    print()
    for model_name in model_subdirs:
        model_records = [r for r in records if r.get("model_name") == model_name]
        if not model_records:
            continue
        scores = {
            k: float(np.mean([r[k] for r in model_records]))
            for k in NOEF_METRIC_NAMES
        }
        comp = float(np.mean([r["composite_score"] for r in model_records]))
        print(f"  [{model_name}] 平均 {len(model_records)} 首:")
        print(f"    pitch_accuracy={scores['pitch_accuracy']:.2f}  "
              f"pitch_stability={scores['pitch_stability']:.2f}")
        print(f"    rhythm_consistency={scores['rhythm_consistency']:.2f}  "
              f"tempo_accuracy={scores['tempo_accuracy']:.2f}")
        print(f"    ornament_score={scores['ornament_score']:.2f}  "
              f"ornament_density={scores['ornament_density']:.2f}")
        print(f"    mode_matching={scores['mode_matching_score']:.2f}  "
              f"fad={scores['fad_distance']:.2f}  "
              f"melody_cos={scores['melody_cosine_similarity']:.2f}")
        print(f"    ★ composite={comp:.2f}")


def _append_csv_row(csv_path: str, row: Dict[str, Any]):
    """追加一行评估记录到 CSV。"""
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow([row.get(k, "") for k in NOEF_CSV_HEADER])


def _write_empty_row(csv_path: str, song_id: str, song_name: str, model_name: str):
    """写入空行（MIDI 缺失时占位）。"""
    row = {k: "" for k in NOEF_CSV_HEADER}
    row["song_id"] = song_id
    row["song_name"] = song_name
    row["model_name"] = model_name
    _append_csv_row(csv_path, row)


# ==============================================================================
# 十、向后兼容层（evaluation.py 使用）
# ==============================================================================

# 原始 audio 级别 evaluate_noef（保持原有签名，供旧调用方使用）
def evaluate_noef(
    audio_path: str,
    target_bpm: Optional[float] = None,
    mode_pcs: Optional[Union[str, Sequence[int]]] = "gong",
    reference_pitches: Optional[Sequence[float]] = None,
    ref_audio_path: Optional[str] = None,
    duration: Optional[float] = 15.0,
    sr: int = 22050,
) -> Dict[str, Optional[float]]:
    """
    [向后兼容] 音频级 NOEF 评估（原始接口，返回值仍为 0–1 范围）。

    新代码请使用符号级 batch_evaluate_generated() 获得 0–100 评分。
    """
    if not _HAS_LIBROSA:
        print("[nanyin_metrics] librosa 未安装，evaluate_noef 不可用")
        # 补齐音频级全部键（含只在音频级才有的 pitch_contour_correlation）
        base: Dict[str, Optional[float]] = {k: None for k in NOEF_METRIC_NAMES}
        base["pitch_contour_correlation"] = None
        return base

    y, sr_out = _load_audio(audio_path, sr=sr, duration=duration)

    # F0 提取
    fmin = librosa.note_to_hz("C2")
    fmax = librosa.note_to_hz("C6")
    f0, vf, _ = librosa.pyin(y, fmin=fmin, fmax=fmax)
    if isinstance(vf, np.ndarray):
        vf = vf.astype(bool)
    vr = float(np.sum(vf)) / max(len(vf), 1)
    if vr < 0.05:
        f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr_out, frame_length=2048)
        vf = (~np.isnan(f0)) & (f0 > fmin)
    voiced = f0[bool_arr(vf)]
    has_voiced = len(voiced) > 0

    # tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr_out)
    tempo_val = _scalar(tempo)

    results: Dict[str, Optional[float]] = {}
    gen_midi: np.ndarray = np.array([], dtype=float)

    if has_voiced:
        gen_midi = librosa.hz_to_midi(voiced)
        # pitch_accuracy 简化为自比对
        results["pitch_accuracy"] = 0.5  # 无参考时给中性分

        jitter = float(np.std(np.diff(gen_midi))) / max(float(np.mean(np.abs(gen_midi))), 1e-6)
        results["pitch_stability"] = max(0.0, min(1.0, 1.0 - jitter))
    else:
        results["pitch_accuracy"] = 0.0
        results["pitch_stability"] = 0.0

    # rhythm
    _, beats = librosa.beat.beat_track(y=y, sr=sr_out)
    beats = np.asarray(beats).reshape(-1)
    if len(beats) >= 2:
        intervals = np.diff(beats) / sr_out
        cv = float(np.std(intervals)) / max(float(np.mean(intervals)), 1e-6)
        results["rhythm_consistency"] = max(0.0, min(1.0, 1.0 - cv))
    else:
        results["rhythm_consistency"] = 0.0

    if target_bpm and tempo_val:
        results["tempo_accuracy"] = max(0.0, min(1.0, 1.0 - abs(tempo_val - target_bpm) / target_bpm))
    else:
        results["tempo_accuracy"] = 0.0

    # ornament
    if has_voiced and len(gen_midi) >= 2:
        diff = np.abs(np.diff(gen_midi))
        results["ornament_density"] = float(np.sum(diff > 0.5)) / max(len(diff), 1)
        rapid = diff[diff > 0.5]
        ev = 1.0 if len(rapid) < 2 else 1.0 - float(np.std(rapid)) / (float(np.mean(rapid)) + 1e-6)
        ps = results.get("pitch_stability") or 0.0
        rc = results.get("rhythm_consistency") or 0.0
        od = results.get("ornament_density") or 0.0
        results["ornament_score"] = (
            0.40 * ps
            + 0.25 * rc
            + 0.20 * od
            + 0.15 * max(0.0, min(1.0, ev))
        )
    else:
        results["ornament_density"] = 0.0
        results["ornament_score"] = 0.0

    # mode
    if has_voiced:
        pcs = resolve_mode_pcs(mode_pcs)
        pc_arr = np.round(gen_midi).astype(int) % 12
        results["mode_matching_score"] = float(np.sum(np.isin(pc_arr, pcs))) / len(pc_arr)
    else:
        results["mode_matching_score"] = 0.0

    # 向后兼容：mode_consistency 键名
    results["mode_consistency"] = results["mode_matching_score"]
    results["pitch_entropy"] = results.get("pitch_stability", 0.0)

    # pitch_contour_correlation（t2 新增，音频级，0–1）：
    # 生成 f0 轮廓 vs 11 首南音音频 f0 风格画像的相似度（无逐点比对）。
    # 画像首次调用时自动构建并缓存到 output/nanyin_f0_profile.npz。
    results["pitch_contour_correlation"] = 0.0
    if has_voiced and len(gen_midi) >= 20:
        try:
            results["pitch_contour_correlation"] = calculate_pitch_contour_correlation(gen_midi)
        except Exception as _e:  # noqa: BLE001
            results["pitch_contour_correlation"] = 0.0

    # FAD / style
    results["fad_distance"] = None
    results["melody_cosine_similarity"] = None
    results["fad"] = None
    results["style_consistency"] = None

    if ref_audio_path:
        y_ref, _ = _load_audio(ref_audio_path, sr=sr, duration=duration)
        m_r, c_r = _extract_mfcc_stats(y_ref, sr)
        m_g, c_g = _extract_mfcc_stats(y, sr)
        fad_val = calculate_frechet_distance(m_r, c_r, m_g, c_g)
        results["fad"] = fad_val
        results["fad_distance"] = 100.0 * math.exp(-fad_val / 20.0)

        if _HAS_SKLEARN:
            sim = _cosine_similarity(m_g.reshape(1, -1), m_r.reshape(1, -1))
            cos = float(sim[0, 0])
        else:
            nr = np.linalg.norm(m_r) + 1e-8
            ng = np.linalg.norm(m_g) + 1e-8
            cos = float(np.dot(m_g, m_r) / (nr * ng))
        results["style_consistency"] = max(0.0, min(1.0, (cos + 1) / 2))

    return results


def compute_noef_composite_score(metrics: Dict[str, Optional[float]]) -> float:
    """
    [向后兼容] 原接口的复合得分 (0–1 范围)。

    新代码请使用 compute_composite_score() 获得 0–100 评分。
    """
    def _avg(keys):
        vals = [metrics[k] for k in keys if metrics.get(k) is not None]
        return float(np.mean(vals)) if vals else 0.0

    pitch = _avg(["pitch_accuracy", "pitch_stability", "pitch_entropy"])
    rhythm = _avg(["rhythm_consistency", "tempo_accuracy"])
    orn = _avg(["ornament_score", "ornament_density"])
    mode = metrics.get("mode_matching_score") or metrics.get("mode_consistency") or 0.0
    style_vals: List[float] = []
    _sc = metrics.get("style_consistency")
    if _sc is not None:
        style_vals.append(float(_sc))
    _fad = metrics.get("fad")
    if _fad is not None:
        style_vals.append(1.0 / (1.0 + float(_fad)))
    else:
        _fad_dist = metrics.get("fad_distance")
        if _fad_dist is not None:
            style_vals.append(float(_fad_dist) / 100.0)
    style = float(np.mean(style_vals)) if style_vals else 0.0

    return (
        NOEF_CATEGORY_WEIGHTS["pitch"] * pitch
        + NOEF_CATEGORY_WEIGHTS["rhythm"] * rhythm
        + NOEF_CATEGORY_WEIGHTS["ornament"] * orn
        + NOEF_CATEGORY_WEIGHTS["mode"] * mode
        + NOEF_CATEGORY_WEIGHTS["style"] * style
    )


def format_noef_results(metrics: Dict[str, Optional[float]]) -> str:
    """格式化 NOEF 结果字符串。"""
    lines = []
    for k in NOEF_METRIC_NAMES:
        v = metrics.get(k)
        if v is None:
            lines.append(f"  {k}: N/A")
        else:
            lines.append(f"  {k}: {v:.4f}")
    c = compute_noef_composite_score(metrics)
    lines.append(f"  noef_composite: {c:.4f}")
    return "\n".join(lines)


# evaluation.py 别名
evaluate_audio = evaluate_noef
format_results = format_noef_results


# ==============================================================================
# 十一、辅助工具
# ==============================================================================

def _scalar(value: Any) -> Optional[float]:
    """将 ndarray/list 转为 Python 标量。"""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return float(value.flat[0])
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        return float(value[0])
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bool_arr(arr) -> np.ndarray:
    """安全转为 bool ndarray。"""
    return np.asarray(arr, dtype=bool)


# ==============================================================================
# 十二、命令行入口 — 默认批量评估
# ==============================================================================

def main():
    """
    NOEF 批量评估主入口。

    无参数运行：自动评估 results/generated_midi/ 下所有 MIDI，
    输出 generate_scores.csv。
    """
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # pyright: ignore[reportAttributeAccessIssue]

    if not _HAS_MIDO:
        print("[错误] mido 库未安装，无法解析 MIDI。请执行: pip install mido")
        sys.exit(1)

    # 检查 MIDI 目录
    if not os.path.isdir(_MIDI_BASE):
        print(f"[错误] 生成 MIDI 目录不存在: {_MIDI_BASE}")
        print("[提示] 请先运行 generate_music.py 生成 .mid 文件")
        sys.exit(1)

    # 批量评估
    batch_evaluate_generated()


if __name__ == "__main__":
    """
    独立运行说明:
      - 依赖: numpy, scipy, mido; 可选: librosa, sklearn
      - 需先运行 generate_music.py 生成 .mid 文件
      - 若同时运行 midi_to_audio.py，可获得完整 FAD 评分

    完整运行命令 (Windows):
      venv/Scripts/python.exe nanyin_metrics.py
    """
    main()
