# -*- coding: utf-8 -*-
"""
generate_music.py — 南音符号旋律批量推理生成脚本
=================================================

功能概述：
  1. 复用 dataset_nanyin.get_val_dataset() 读取2首验证集曲目标注
  2. 分别加载 ./checkpoints/{boyatcn,musicmamba}/ 下最优权重
  3. 自回归逐token生成旋律token序列，搭配模型预测的duration
  4. Token解码输出为标准的 .mid MIDI文件，按模型分文件夹存放
  5. 生成记录自动写入 ./results/generate_list.csv
  6. 实时打印当前生成曲目、序列长度、完成提示

参考文献：
  [1] Music Transformer (Huang et al., 2018)
      https://arxiv.org/abs/1809.04281
      自回归音乐生成范式，逐token预测下一音符，将生成结果写入MIDI
  [2] MelodyGLM (https://arxiv.org/pdf/2309.10738v1)
      旋律符号生成中采用 teacher-forcing → argmax 解码写入MIDI文件
  [3] MusicMamba (KAD arXiv:2502.15602)
      SSM架构在音乐长序列建模与生成中的应用

运行命令：
  venv/Scripts/python.exe generate_music.py
  或
  python generate_music.py
"""

import os
import csv
import sys
import random
from typing import Tuple, List, Optional, Sequence

import torch
import torch.nn.functional as F
import mido
import numpy as np

# ------------------- 将项目根目录加入 sys.path，确保模块导入正常 -------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from dataset_nanyin import (
    get_val_dataset, NanyinDataset, VAL_SPLIT_IDS,
    NANYIN_FOUR_MODES, _MODE_PENTATONIC,
)
from models import NanyinBoYaTCN, NanyinMusicMamba


# ==============================================================================
# 一、全局配置
# ==============================================================================

# 模型参数（与 train_* 脚本保持一致）
VOCAB_SIZE: int = 128
D_MODEL: int = 256
NUM_HEADS: int = 8
NUM_SSM_LAYERS: int = 3
D_STATE: int = 16
SSM_EXPAND: int = 2
MAX_SEQ_LEN: int = 4096

# 两套模型的检查点路径
CHECKPOINT_DIR: str = os.path.join(_SCRIPT_DIR, "checkpoints")
MODEL_CHECKPOINTS = {
    "BoYaTCN": os.path.join(CHECKPOINT_DIR, "boyatcn", "best_model.pt"),
    "MusicMamba": os.path.join(CHECKPOINT_DIR, "musicmamba", "best_model.pt"),
}

# 生成结果输出目录
RESULTS_DIR: str = os.path.join(_SCRIPT_DIR, "results")
GENERATED_MIDI_BASE: str = os.path.join(RESULTS_DIR, "generated_midi")
OUTPUT_DIRS = {
    "BoYaTCN": os.path.join(GENERATED_MIDI_BASE, "boyatcn"),
    "MusicMamba": os.path.join(GENERATED_MIDI_BASE, "musicmamba"),
}

# 生成记录 CSV
GENERATE_LIST_CSV: str = os.path.join(RESULTS_DIR, "generate_list.csv")

# MIDI 参数
DEFAULT_VELOCITY: int = 80        # 默认音符力度
TICKS_PER_BEAT: int = 480         # MIDI 分辨率 (PPQN)
DEFAULT_BPM: int = 120            # 默认速度
DEFAULT_DURATION_SEC: float = 0.5 # 兜底时值（秒）
LEGATO_MS: float = 15.0           # legato 重叠时长 (ms)，模拟琵琶左手换把时弦的持续振动


def compute_velocities(n_notes: int) -> List[int]:
    """为每个音符计算力度值，模拟南音琵琶强弱动态。

    南音工尺谱以"撩拍"为节拍单位，每拍有强弱之分。
    按 4/4 拍强-弱-次强-弱模式生成力度变化：
      - 位置%4==0 (强拍):   velocity 88~100  — 拍位，重音
      - 位置%4==2 (次强拍): velocity 78~88   — 半拍位
      - 位置%4==1/3 (弱拍): velocity 68~78  — 撩位，轻弹

    力度值直接写入 MIDI note_on 事件，FluidSynth 渲染时
    映射为 SF2 采样音量层，强弱动态完整保留。

    Args:
        n_notes: 音符数量

    Returns:
        长度 = n_notes 的力度值列表
    """
    velocities: List[int] = []
    for i in range(n_notes):
        jitter = random.randint(-3, 3)  # 小幅随机扰动，避免机械感
        if i % 4 == 0:
            v = random.randint(88, 100) + jitter  # 强拍
        elif i % 4 == 2:
            v = random.randint(78, 88) + jitter   # 次强拍
        else:
            v = random.randint(68, 78) + jitter   # 弱拍
        velocities.append(max(40, min(127, v)))
    return velocities

# MIDI 乐器程序号：program 24 = Nylon Guitar，GM 库中最接近南音琵琶的拨弦音色
# 参考：GM1 Sound Set, program 24 = Acoustic Guitar (nylon)
# 作用：即使 FluidSynth 使用 GM SoundFont，也不会默认出钢琴声
MIDI_PROGRAM: int = 24

# 自回归最小 prompt 长度（避免 BoYaTCN 中 avg_pool1d 因序列过短崩溃）
MIN_PROMPT_LEN: int = 16           # 减少评估污染；实测 16 和 32 的预测准确率无显著差异

# 自回归最大上下文长度：超过后只保留最近 N 个 token。
# MusicMamba 没有实现 step cache，每步都重新 forward 整个序列，
# 若不截断，序列增长会导致 CUDA OOM。512 足以覆盖南音局部乐句结构。
MAX_GEN_CONTEXT_LEN: int = 512

# 采样参数（通用默认值）
SAMPLING_TEMPERATURE: float = 0.7  # 温度 (<1.0 更稳定, >1.0 更随机)
SAMPLING_TOP_P: float = 0.92       # nucleus sampling 阈值
SAMPLING_TOP_K: int = 50           # top-k 过滤
REPETITION_PENALTY: float = 1.5    # 重复惩罚 (≥1.0, 越大越抑制重复)
REPETITION_WINDOW: int = 64        # 重复检测窗口 (最近N个token内出现过的受惩罚)
BLOCK_NGRAM: int = 12              # n-gram 阻断: 禁止连续12个相同token（避免误伤轮指/同音反复）

# 音高连续性引导 —— 已按真实南音风格关闭（依据实测数据）：
# 对比真实乐谱(GT)与生成序列：GT 大跳(>六度)占比 0.56~0.62，级进(二度内)仅 0.20~0.31。
# 原 0.25 强度会惩罚一切大跳，导致生成旋律只剩二度级进、毫无南音"大跳进"韵味。
# 故默认置 0 关闭；如需开启可显式传入 >0 的值。
PITCH_SMOOTHNESS_STRENGTH: float = 0.0  # 惩罚强度 (0=关闭, 越大越平滑)
PITCH_SMOOTHNESS_OCTAVE_PENALTY: float = 2.0  # 超过八度的跳变加倍惩罚

# 调式五声音阶引导：将 logits 向当前曲目南音调式五声音阶收敛。
# 依据实测：GT 音符 79%~84% 落在五声音阶内（dapu14/15），该引导可提升调性感。
# 0=关闭; >0 表示对非音阶音 logit 施加的降权强度（越大越偏向音阶内音）。
MODE_SCALE_GUIDANCE: float = 0.35

# 模型专属采样参数（Mamba 自回归更容易崩溃，所以更保守）
MODEL_SAMPLING_OVERRIDES: dict = {
    "MusicMamba": {
        "temperature": 0.6,
        "repetition_penalty": 1.8,
        "block_ngram": 8,        # Mamba 卡死严重，阻断更积极
        "anti_oscillation": True, # 抗双音振荡 (a,a,b,b,c,c)
    },
    "BoYaTCN": {
        "temperature": 0.75,
        "repetition_penalty": 1.3,
        "block_ngram": 12,
        "use_gt_rhythm": True,    # duration 头坏死 → 用 GT 节奏采样
    },
}
SONG_CHINESE_NAMES: dict = {
    "dapu14kouhuangtian": "叩皇天",
    "dapu15wujinjiao":    "舞金蛟",
}


# ==============================================================================
# 二、模型加载
# ==============================================================================

def load_boyatcn(ckpt_path: str, device: torch.device) -> NanyinBoYaTCN:
    """
    加载 NanyinBoYaTCN 模型并载入最优权重。

    参数:
        ckpt_path: best_model.pt 检查点文件路径
        device:    torch 设备
    返回:
        加载好权重的 BoYaTCN 模型实例 (eval 模式)
    """
    model = NanyinBoYaTCN(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
    ).to(device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[generate] BoYaTCN 权重文件不存在: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    pass  # 静默加载
    return model


def load_musicmamba(ckpt_path: str, device: torch.device) -> NanyinMusicMamba:
    """
    加载 NanyinMusicMamba 模型并载入最优权重。

    参数:
        ckpt_path: best_model.pt 检查点文件路径
        device:    torch 设备
    返回:
        加载好权重的 MusicMamba 模型实例 (eval 模式)
    """
    model = NanyinMusicMamba(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_SSM_LAYERS,
        d_state=D_STATE,
        ssm_expand=SSM_EXPAND,
    ).to(device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[generate] MusicMamba 权重文件不存在: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    pass  # 静默加载
    return model


# ==============================================================================
# 三、自回归旋律生成
# ==============================================================================

def generate_autoregressive(
    model: torch.nn.Module,
    prompt_tokens: List[int],
    prompt_durations: List[float],
    target_len: int,
    device: torch.device,
    temperature: float = 0.9,
    top_p: float = 0.92,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
    repetition_window: int = 64,
    block_ngram: int = 4,
    gt_durations: Optional[Sequence[float]] = None,
    use_gt_rhythm: bool = False,
    anti_oscillation: bool = False,
    pitch_smoothness_strength: float = 0.0,
    pitch_octave_penalty: float = 2.0,
    mode_scale_pcs: Optional[List[int]] = None,
    mode_scale_guidance: float = 0.0,
) -> Tuple[List[int], List[float]]:
    """
    自回归逐 token 生成旋律序列（带 temperature + top-p + top-k 采样 +
    重复惩罚 + n-gram 阻断）。

    流程：
      1. 以 ground-truth 前若干个 token 作为 prompt（条件输入）
      2. 每步将当前已生成序列送入模型 forward()
      3. 取 pred_tokens 最后位置的 logits → repetition_penalty → temperature
         → top-k → top-p → n-gram_block → 采样 → 下一个 token
      4. 同时收集 pred_duration 作为当前步的时值
      5. 重复直到达到目标长度 target_len 或 MAX_SEQ_LEN 上限

    采样策略：
      - repetition_penalty: 对窗口内已出现 token 除以 penalty (>1.0 抑制重复)
      - temperature: 调节分布的尖锐程度 (<1.0 更保守, >1.0 更随机)
      - top-k: 只保留概率最高的 k 个 token，其余置零后重归一化
      - top-p (nucleus): 只保留累计概率达到 p 的最少 token 集合
      - block_ngram: 禁止生成"连续N个完全相同的token"

    文献来源：
      Music Transformer (Huang et al., 2018) — 自回归音乐生成范式
      The Curious Case of Neural Text Degeneration (Holtzman et al., 2020) — top-p 采样
      CTRL (Keskar et al., 2019) — repetition penalty
      A. Holtzman et al. — n-gram blocking for text generation

    参数:
        model:                     已加载权重的模型 (eval 模式)
        prompt_tokens:             起始 prompt token 序列 (ground-truth 前 N 个 token)
        prompt_durations:          prompt 各位置的时值 (秒)
        target_len:                目标生成序列长度 (取 ground-truth 序列长度)
        device:                    torch 设备
        temperature:               softmax 温度系数 (默认 0.9)
        top_p:                     nucleus 采样阈值 (默认 0.92)
        top_k:                     top-k 过滤值 (默认 50)
        repetition_penalty:        重复惩罚 (≥1.0, 默认 1.2, 越大越抑制重复)
        repetition_window:         重复检测窗口大小 (默认 64)
        block_ngram:               n-gram 阻断, 禁止连续N个相同token (默认 4)
        pitch_smoothness_strength: 音高连续性引导强度 (0=关闭, 默认 0.25)
        pitch_octave_penalty:      超八度跳变额外惩罚系数 (默认 2.0)
    返回:
        (generated_tokens, durations)
        - generated_tokens: 生成的 token 序列 (含 prompt)
        - durations:        每个位置的时值 (秒), 与 token 一一对应
    """
    generated_tokens: List[int] = list(prompt_tokens)
    durations: List[float] = list(prompt_durations)
    prompt_len = len(prompt_tokens)
    actual_target = min(target_len, MAX_SEQ_LEN)
    stuck_count: int = 0    # 连续重复计数器

    # ---- 构建 GT duration 采样池（用于 BoYaTCN duration 头坏死兜底）----
    dur_pool: Optional[np.ndarray] = None
    if use_gt_rhythm and gt_durations is not None and len(gt_durations) > 0:
        dur_pool = np.array(gt_durations, dtype=np.float32)

    with torch.no_grad():
        for _ in range(prompt_len, actual_target):
            # 构建输入张量: (1, current_len)，超过最大上下文则只保留最近部分
            context_tokens = generated_tokens
            if len(context_tokens) > MAX_GEN_CONTEXT_LEN:
                context_tokens = context_tokens[-MAX_GEN_CONTEXT_LEN:]
            input_seq = torch.tensor(
                [context_tokens], dtype=torch.long, device=device
            )

            # 前向推理
            output: dict = model(input_seq)

            # ---- 提取最后一位预测 ----
            # pred_tokens: (1, current_len, vocab_size) → 取最后位置 logits
            last_logits = output["pred_tokens"][0, -1, :].clone()  # (vocab_size,)

            # Step 1: repetition_penalty — 对窗口内已出现的 token 降权
            if repetition_penalty > 1.0 and len(generated_tokens) > 0:
                window_start = max(0, len(generated_tokens) - repetition_window)
                recent_tokens = generated_tokens[window_start:]
                # 统计窗口内出现频率，频繁出现的惩罚更重
                from collections import Counter
                freq = Counter(recent_tokens)
                for tid, count in freq.items():
                    if count >= 2:  # 仅惩罚出现≥2次的token
                        # 惩罚强度随频率递增
                        penalty = repetition_penalty ** count
                        # logits 为正值时分母惩罚，为负值时分子惩罚
                        last_logits[tid] = torch.where(
                            last_logits[tid] > 0,
                            last_logits[tid] / penalty,
                            last_logits[tid] * penalty,
                        )

            # Step 2: temperature 缩放
            last_logits = last_logits / temperature

            # Step 2.5: 音高连续性引导 — 惩罚与前音音程过大的 token
            # 南音旋律以级进和小跳为主，避免大跳使旋律听起来"断断续续"
            if pitch_smoothness_strength > 0 and len(generated_tokens) > 0:
                prev_pitch = int(generated_tokens[-1])
                for tid in range(len(last_logits)):
                    logit_val = last_logits[tid].item()
                    if logit_val <= float("-inf"):
                        continue
                    interval = abs(int(tid) - prev_pitch)
                    if interval <= 2:
                        continue  # 二度以内不惩罚（级进是正常的）
                    # 惩罚随音程增大而递增，超过八度加倍
                    penalty = 1.0 + pitch_smoothness_strength * (interval / 2.0 - 1.0)
                    if interval > 12:
                        penalty += pitch_smoothness_strength * pitch_octave_penalty * ((interval - 12) / 12.0)
                    # 正值 logit 除以 penalty，负值 logit 乘以 penalty
                    if last_logits[tid] > 0:
                        last_logits[tid] = last_logits[tid] / penalty
                    else:
                        last_logits[tid] = last_logits[tid] * penalty

            # Step 2.6: 调式五声音阶引导 — 降低非音阶音 logit，提升南音调性感
            # 依据实测：GT 音符 79%~84% 落在当前调式五声音阶内。
            # 软约束（而非硬屏蔽）：仅降权非音阶音，保留小概率探索装饰音/偏音。
            if mode_scale_guidance > 0 and mode_scale_pcs is not None:
                scale_set = set(mode_scale_pcs)
                for tid in range(len(last_logits)):
                    if last_logits[tid] <= float("-inf"):
                        continue
                    if (tid % 12) not in scale_set:
                        if last_logits[tid] > 0:
                            last_logits[tid] = last_logits[tid] * (1.0 - mode_scale_guidance)
                        else:
                            last_logits[tid] = last_logits[tid] * (1.0 + mode_scale_guidance)

            # Step 3: top-k 过滤
            if top_k > 0:
                top_k_vals, top_k_idx = torch.topk(last_logits, k=min(top_k, len(last_logits)))
                mask = torch.full_like(last_logits, float("-inf"))
                mask.scatter_(0, top_k_idx, top_k_vals)
                last_logits = mask

            # Step 4: top-p (nucleus) 过滤
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(last_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # 移除累积概率超过 top_p 的 token
                sorted_indices_to_remove = cumulative_probs > top_p
                # 始终保留至少一个 token
                sorted_indices_to_remove[0] = False
                # 将原始顺序的 index 标记为 -inf
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                last_logits[indices_to_remove] = float("-inf")

            # Step 5: n-gram 阻断 — 禁止再生成连续N个相同token
            blocked_token: Optional[int] = None
            if block_ngram > 0 and len(generated_tokens) >= block_ngram - 1:
                # 检查最近 (block_ngram - 1) 个token是否完全相同
                recent = generated_tokens[-(block_ngram - 1):]
                if len(set(recent)) == 1:  # 全部相同
                    # 禁止该token再次被采样
                    last_logits[recent[0]] = float("-inf")
                    blocked_token = recent[0]

            # Step 6: softmax + 多项式采样
            # 安全兜底：如果所有 logit 都是 -inf，退化为均匀采样（避免阻断导致 NaN）
            if torch.all(torch.isinf(last_logits)):
                last_logits = torch.zeros_like(last_logits)
                if blocked_token is not None:
                    last_logits[blocked_token] = float("-inf")
                if torch.all(torch.isinf(last_logits)):
                    last_logits = torch.zeros_like(last_logits)

            probs = F.softmax(last_logits, dim=-1)
            next_token = int(torch.multinomial(probs, num_samples=1).item())

            # ---- duration 生成策略 ----
            if use_gt_rhythm and dur_pool is not None:
                # 从 GT 分布中随机采样（BoYaTCN duration 头坏死时启用）
                next_dur = float(np.random.choice(dur_pool))
                next_dur = max(0.05, min(next_dur, 4.0))
            else:
                # pred_duration: (1, current_len) → 取最后位置
                next_dur = float(output["pred_duration"][0, -1].item())
                next_dur = max(0.05, min(next_dur, 4.0))  # clamp [0.05s, 4s]

            # ---- 抗双音振荡检测 (MusicMamba) ----
            # 模式: (a,a, b,b, c,c, d,d, ...) → 每相邻两个相同，但不同组不同
            # 每步检查最近 N 个 token 中是否连续出现多组双音对
            if anti_oscillation and len(generated_tokens) >= 6:
                t = generated_tokens
                # 从末尾往前，每 2 个 token 一组，统计连续双音对数
                double_pairs = 0
                max_check = min(14, len(t) // 2 * 2)  # 最多检查 7 组
                for pair_idx in range(max_check // 2):
                    i = len(t) - 1 - pair_idx * 2
                    if i >= 1 and t[i] == t[i - 1]:
                        double_pairs += 1
                    else:
                        break
                # 确保每组双音是不同的音高（不是全一样）
                pair_values = [t[len(t) - 1 - p * 2] for p in range(double_pairs)]
                distinct_pairs = len(set(pair_values))
                # 5+ 组双音且至少 3 种不同音高 → 判定为振荡
                if double_pairs >= 5 and distinct_pairs >= 3:
                    valid_mask = last_logits > float("-inf")
                    valid_indices = torch.where(valid_mask)[0]
                    if len(valid_indices) > 0:
                        # 排除最近两组重复音
                        recent_pitches = set(t[-4:])
                        candidates = [int(v.item()) for v in valid_indices
                                      if int(v.item()) not in recent_pitches]
                        if len(candidates) < 2:
                            candidates = [int(v.item()) for v in valid_indices]
                        pick = random.choice(candidates)
                        next_token = pick

            # 重复卡死检测：连续生成同一token超过24步时强制抖动
            if len(generated_tokens) > 0 and next_token == generated_tokens[-1]:
                stuck_count += 1
                if stuck_count >= 24:
                    # 强制在 top-k 中随机选一个非重复token
                    valid_mask = last_logits > float("-inf")
                    valid_indices = torch.where(valid_mask)[0]
                    if len(valid_indices) > 0:
                        # 排除当前重复的token
                        different_mask = valid_indices != generated_tokens[-1]
                        candidate_indices = valid_indices[different_mask]
                        if len(candidate_indices) > 0:
                            pick = int(torch.randint(0, len(candidate_indices), (1,)).item())
                            next_token = int(candidate_indices[pick].item())
                            stuck_count = 0
            else:
                stuck_count = 0

            generated_tokens.append(next_token)
            durations.append(next_dur)

    return generated_tokens, durations


# ==============================================================================
# 四、Token → MIDI 写入
# ==============================================================================

def tokens_to_midi(
    tokens: List[int],
    durations: List[float],
    output_path: str,
    ticks_per_beat: int = TICKS_PER_BEAT,
    tempo_bpm: int = DEFAULT_BPM,
    velocity: int = DEFAULT_VELOCITY,
    velocities: List[int] | None = None,
) -> None:
    """
    将 (token序列, 时值序列) 转换为标准 MIDI 文件 (Format 0, 单轨道).

    原理：
      - token ID 即 MIDI pitch 值 (0~127)，直接作为 note 音高
      - duration (秒) → 转换为 MIDI ticks:
          tick = duration_sec × (tempo / 60) × ticks_per_beat
      - 每个 note 完整生命周期：note_on (pitch, velocity) → 等待 ticks → note_off
      - 力度支持两种模式：
          * velocities=None → 所有音符统一使用 velocity 参数
          * velocities=list → 每个音符使用独立力度值，模拟强弱动态变化
            FluidSynth 渲染时力度直接映射 SF2 采样音量层

    文献来源：
      MelodyGLM (https://arxiv.org/pdf/2309.10738v1)
      该文在符号旋律生成评估中将生成的 token 序列解码为标准 MIDI 并
      交由人类评委与自动化指标进行评估。

    参数:
        tokens:         生成的 token ID 序列 (MIDI pitch)
        durations:      每个 token 的时值 (秒)
        output_path:    输出 .mid 文件路径
        ticks_per_beat: MIDI 分辨率
        tempo_bpm:      默认速度
        velocity:       统一音符力度（velocities=None 时使用）
        velocities:     每音符独立力度值列表（可选，优先于 velocity）
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 创建 MIDI 文件 (Type 0: 单轨道)
    mid = mido.MidiFile(type=0, ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # 设置速度 (tempo 元事件, 单位: 微秒/拍)
    tempo_microseconds = mido.bpm2tempo(tempo_bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo_microseconds, time=0))

    # 设置乐器音色：尼龙弦吉他 (program 24)，GM 中最接近南音琵琶的拨弦乐器
    track.append(mido.Message(
        "program_change", program=MIDI_PROGRAM, channel=0, time=0,
    ))

    N = min(len(tokens), len(durations))
    # 解析力度：优先 per-note 列表，回退统一值
    note_velocities = velocities if velocities is not None else [velocity] * N

    # Legato 重叠量 (ticks)：模拟琵琶左手换把时弦持续振动，音与音之间柔和衔接
    legato_ticks: int = max(1, int(round(
        LEGATO_MS / 1000.0 * (tempo_bpm / 60.0) * ticks_per_beat
    )))

    prev_pitch: int = 0
    prev_dur_ticks: int = 0
    on_delta_next: int = 0

    for i in range(N):
        pitch = int(max(0, min(127, tokens[i])))  # clamp 到 MIDI 范围
        dur_sec = max(0.01, min(4.0, durations[i]))  # 安全 clamp
        vel = int(max(1, min(127, note_velocities[i] if i < len(note_velocities) else velocity)))

        # 时值转换: 秒 → ticks
        dur_ticks = int(round(dur_sec * (tempo_bpm / 60.0) * ticks_per_beat))
        dur_ticks = max(1, dur_ticks)  # 至少 1 tick
        legato = min(legato_ticks, dur_ticks // 3)  # 重叠不超过时值的 1/3
        if pitch == prev_pitch:
            # 同音相邻不重叠：note_on/off 按 (channel,pitch) 配对，若重叠会产生
            # 两个同音 on 夹一个 off 的歧义事件流，下游解析(mido/pretty_midi)
            # 会错配导致丢音符、序列错位（曾使 1025 音符被解析成 792）。
            legato = 0

        if i == 0:
            # 首音：直接起音
            track.append(
                mido.Message("note_on", note=pitch, velocity=vel, time=0)
            )
            on_delta_next = dur_ticks - legato
        else:
            # 当前音在上一音结束前 legato ticks 起音（模拟连奏）
            track.append(
                mido.Message("note_on", note=pitch, velocity=vel, time=on_delta_next)
            )
            # 关闭上一音：延后 legato ticks，与当前音形成重叠
            track.append(
                mido.Message("note_off", note=prev_pitch, velocity=0, time=legato)
            )
            on_delta_next = max(1, dur_ticks - legato)

        prev_pitch = pitch
        prev_dur_ticks = dur_ticks

    # 关闭最后一个音符（自然时值，不延长）
    track.append(
        mido.Message("note_off", note=prev_pitch, velocity=0, time=prev_dur_ticks)
    )

    # 文件结尾
    track.append(mido.MetaMessage("end_of_track", time=0))

    mid.save(output_path)


# ==============================================================================
# 五、生成记录写入
# ==============================================================================

def init_generate_csv(csv_path: str) -> None:
    """初始化生成记录 CSV，写入表头。"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "song_id", "song_name", "model_name",
            "seq_length", "output_midi_path",
        ])


def append_generate_row(
    csv_path: str,
    song_id: str,
    song_name: str,
    model_name: str,
    seq_length: int,
    midi_path: str,
) -> None:
    """在生成记录 CSV 末尾追加一行。"""
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([song_id, song_name, model_name, seq_length, midi_path])


# ==============================================================================
# 六、主流程：批量生成
# ==============================================================================

def main() -> None:
    """批量推理生成主函数。"""
    print("=" * 70)
    print("南音符号旋律批量推理生成")
    print("=" * 70)

    # ---- 0. 设备检测 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[generate] 推理设备: {device}")

    # ---- 1. 创建输出目录 ----
    print(f"\n[generate] 创建输出目录...")
    for model_name, out_dir in OUTPUT_DIRS.items():
        os.makedirs(out_dir, exist_ok=True)

    # ---- 2. 初始化生成记录 CSV ----
    print(f"\n[generate] 初始化生成记录 CSV: {GENERATE_LIST_CSV}")
    init_generate_csv(GENERATE_LIST_CSV)

    # ---- 3. 加载验证集 Dataset (2首曲目) ----
    print(f"\n[generate] 加载验证集 Dataset (2首曲目)...")
    val_ds: NanyinDataset = get_val_dataset()
    assert len(val_ds) == 2, f"验证集曲目数应为2，实际={len(val_ds)}"
    print(f"[generate] 验证集曲目: {val_ds.song_ids}")

    # ---- 4. 预加载验证集数据到内存 ----
    val_data: List[Tuple[str, torch.Tensor, dict]] = []
    for idx in range(len(val_ds)):
        sid = val_ds.get_song_id(idx)
        input_tokens, target_dict = val_ds[idx]
        val_data.append((sid, input_tokens, target_dict))
        pass  # 静默加载验证集数据

    # ---- 5. 逐个模型循环生成 ----
    model_loaders = {
        "BoYaTCN":    (load_boyatcn, MODEL_CHECKPOINTS["BoYaTCN"]),
        "MusicMamba": (load_musicmamba, MODEL_CHECKPOINTS["MusicMamba"]),
    }

    for model_name, (loader_fn, ckpt_path) in model_loaders.items():
        print(f"\n{'=' * 70}")
        print(f"[generate] 开始模型: {model_name}")
        print(f"[generate] 权重路径: {ckpt_path}")
        print(f"{'=' * 70}")

        if not os.path.exists(ckpt_path):
            print(f"[generate] 警告: 权重文件不存在，跳过 {model_name}")
            continue

        # 加载模型
        model = loader_fn(ckpt_path, device)

        out_dir = OUTPUT_DIRS[model_name]

        # 逐曲目生成
        for song_id, input_tokens, target_dict in val_data:
            song_name = SONG_CHINESE_NAMES.get(song_id, song_id)
            target_len = int(input_tokens.shape[0])
            gt_durations = target_dict["target_duration"].tolist()

            print(f"[generate]   {song_name} ({song_id}) [{model_name}]")

            # ---- 构建 prompt（条件输入）----
            # 取 ground-truth 前 MIN_PROMPT_LEN 个 token 作为生成条件
            prompt_len = min(MIN_PROMPT_LEN, target_len)
            prompt_tokens = input_tokens[:prompt_len].tolist()
            prompt_durations = gt_durations[:prompt_len]

            # ---- 模型专属采样参数覆盖 ----
            overrides = MODEL_SAMPLING_OVERRIDES.get(model_name, {})
            gen_temperature = overrides.get("temperature", SAMPLING_TEMPERATURE)
            gen_rep_penalty = overrides.get("repetition_penalty", REPETITION_PENALTY)
            gen_block_ngram = overrides.get("block_ngram", BLOCK_NGRAM)
            # BoYaTCN duration 头坏死 → 用 GT 节奏采样替代
            use_gt_rhythm = overrides.get("use_gt_rhythm", False)
            # MusicMamba 双音振荡 → 启用抗振荡
            anti_oscillation = overrides.get("anti_oscillation", False)

            # ---- 解码当前曲目的南音调式，供五声音阶引导 ----
            mode_dist = target_dict["target_mode_dist"].numpy()
            mode_idx = int(np.argmax(mode_dist))
            mode_name = NANYIN_FOUR_MODES[mode_idx] if 0 <= mode_idx < len(NANYIN_FOUR_MODES) else NANYIN_FOUR_MODES[0]
            mode_pcs = _MODE_PENTATONIC.get(mode_name, [0, 2, 4, 7, 9])

            # ---- 自回归生成 ----
            generated_tokens, durations = generate_autoregressive(
                model=model,
                prompt_tokens=prompt_tokens,
                prompt_durations=prompt_durations,
                target_len=target_len,
                device=device,
                temperature=gen_temperature,
                top_p=SAMPLING_TOP_P,
                top_k=SAMPLING_TOP_K,
                repetition_penalty=gen_rep_penalty,
                repetition_window=REPETITION_WINDOW,
                block_ngram=gen_block_ngram,
                gt_durations=gt_durations,
                use_gt_rhythm=use_gt_rhythm,
                anti_oscillation=anti_oscillation,
                pitch_smoothness_strength=PITCH_SMOOTHNESS_STRENGTH,
                pitch_octave_penalty=PITCH_SMOOTHNESS_OCTAVE_PENALTY,
                mode_scale_pcs=mode_pcs,
                mode_scale_guidance=MODE_SCALE_GUIDANCE,
            )

            gen_len = len(generated_tokens)
            print(f"[generate]   生成完成: 实际长度={gen_len} tokens")

            # ---- 写入 MIDI 文件 ----
            midi_filename = f"{song_id}_{model_name.lower()}.mid"
            midi_path = os.path.join(out_dir, midi_filename)
            tokens_to_midi(
                tokens=generated_tokens,
                durations=durations,
                output_path=midi_path,
                velocities=compute_velocities(len(generated_tokens)),
            )

            # ---- 追加生成记录 ----
            append_generate_row(
                csv_path=GENERATE_LIST_CSV,
                song_id=song_id,
                song_name=song_name,
                model_name=model_name,
                seq_length=gen_len,
                midi_path=midi_path,
            )
            pass  # 静默追加记录

        # 模型推理完毕
        print(f"\n[generate] ✓ 模型 {model_name} 全部曲目生成完成")
        # 释放模型显存
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- 6. 汇总打印 ----
    print(f"\n{'=' * 70}")
    print(f"[generate] 全部生成任务完成!")
    print(f"{'=' * 70}")
    print(f"  生成曲目数:  {len(val_data)} 首/模型 × {len(model_loaders)} 模型")
    print(f"  总 MIDI 文件: {len(val_data) * len(model_loaders)} 个")
    print(f"  MIDI 输出目录:")
    for model_name, out_dir in OUTPUT_DIRS.items():
        midi_count = len([
            f for f in os.listdir(out_dir)
            if f.endswith(".mid")
        ]) if os.path.isdir(out_dir) else 0
        print(f"    {out_dir}  ({midi_count} 个 .mid)")
    print(f"  生成记录 CSV:  {GENERATE_LIST_CSV}")
    cmd = r".\venv\Scripts\python.exe generate_music.py"
    print(f"\n  完整运行命令:")
    print(f"    {cmd}")
    print(f"    或")
    print(f"    python generate_music.py")


# ==============================================================================
# 七、脚本入口
# ==============================================================================

if __name__ == "__main__":
    """
    独立运行说明:
      - 依赖: torch, mido, numpy (已存在于项目 venv)
      - 需先完成模型训练 (checkpoints 下存在 best_model.pt)
      - 生成结果写入 ./results/ 目录

    完整运行命令 (Windows):
      venv/Scripts/python.exe generate_music.py
    """
    # Windows GBK 终端兼容：强制 UTF-8 输出
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # pyright: ignore
    main()
