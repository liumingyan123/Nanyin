# -*- coding: utf-8 -*-
"""
generate_music.py — 南音符号旋律批量推理生成脚本（E1：复合事件口径）
====================================================================

E1 改动要点（复合事件 = pitch + duration + time-shift）：
  1. 生成阶段对同一时间步**同时采样** pitch / duration / time-shift 三路分类分布
  2. 音符 onset 由 **time-shift 累加** 得到（而非旧版的"上一个音多长就往后排多久"），
     因此可以真实表达休止与停顿（撩拍气口）
  3. MIDI 输出逻辑重写：**绝对时间事件流**（所有 note_on/note_off 按 tick 排序后
     按 delta 写入），取代旧版的"逐个 note_on → note_off 接龙"写法
  4. 所有采样 / 量化 / 词表参数统一来自 configs/e1_compound_events.yaml

⚠ 当前状态：E1 阶段只完成复合事件与 MIDI 往返编解码改造，**尚未训练**复合事件模型，
   因此 checkpoints 下不存在对应权重；直接运行本脚本会明确提示并安全退出，
   不会输出任何 MIDI、不会伪造生成结果。

参考文献：
  [1] Music Transformer (Huang et al., 2018)
      https://arxiv.org/abs/1809.04281
      自回归音乐生成范式，逐 token 预测下一音符，将生成结果写入 MIDI
  [2] Compound Word / REMI 风格事件表示（CP Word: Hsiao et al., 2021, arXiv:2102.02412）
      音符级复合事件 (pitch, duration, time-shift) 同时预测
  [3] MusicMamba (KAD arXiv:2502.15602)
      SSM 架构在音乐长序列建模与生成中的应用

运行命令：
  venv/Scripts/python.exe generate_music.py
  或
  python generate_music.py
"""

import os
import csv
import sys
import time
import random
import argparse
from typing import Tuple, List, Optional, Sequence, Dict, Any

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
    CompoundEventCodec, get_compound_codec,
    CompoundEventWindowDataset,
)
from models import NanyinBoYaTCN, NanyinMusicMamba
from nanyin_config import load_config, get_section


# ==============================================================================
# 一、全局配置（全部来自 configs/e1_compound_events.yaml，禁止硬编码）
# ==============================================================================

CFG: dict = load_config()
CODEC_CFG: dict = CFG["codec"]
MODEL_CFG: dict = CFG["model"]
GEN_CFG: dict = CFG["generation"]

# 复合事件编解码器（pitch + duration + time-shift）
CODEC: CompoundEventCodec = get_compound_codec()

# 模型结构参数
VOCAB_SIZE: int = int(CODEC_CFG["pitch_vocab_size"])
D_MODEL: int = int(MODEL_CFG["d_model"])
NUM_HEADS: int = int(MODEL_CFG["num_heads"])
NUM_SSM_LAYERS: int = int(MODEL_CFG["num_ssm_layers"])
D_STATE: int = int(MODEL_CFG["d_state"])
SSM_EXPAND: int = int(MODEL_CFG["ssm_expand"])
MAX_SEQ_LEN: int = int(MODEL_CFG["max_seq_len"])

# 三路属性的采样温度 / top-p / top-k
PITCH_TEMPERATURE: float = float(GEN_CFG["pitch_temperature"])
PITCH_TOP_P: float = float(GEN_CFG["pitch_top_p"])
PITCH_TOP_K: int = int(GEN_CFG["pitch_top_k"])
DURATION_TEMPERATURE: float = float(GEN_CFG["duration_temperature"])
DURATION_TOP_P: float = float(GEN_CFG["duration_top_p"])
DURATION_TOP_K: int = int(GEN_CFG["duration_top_k"])
SHIFT_TEMPERATURE: float = float(GEN_CFG["shift_temperature"])
SHIFT_TOP_P: float = float(GEN_CFG["shift_top_p"])
SHIFT_TOP_K: int = int(GEN_CFG["shift_top_k"])

# 两套模型的检查点路径（复合事件口径，独立子目录，避免与旧 [-only pitch] 权重混淆）
CHECKPOINT_DIR: str = os.path.join(_SCRIPT_DIR, GEN_CFG["checkpoint_dir"])
MODEL_CHECKPOINTS = {
    "BoYaTCN": os.path.join(CHECKPOINT_DIR, "compound_boyatcn", "best_model.pt"),
    "MusicMamba": os.path.join(CHECKPOINT_DIR, "compound_musicmamba", "best_model.pt"),
}

# 生成结果输出目录
RESULTS_DIR: str = os.path.join(_SCRIPT_DIR, "results")
GENERATED_MIDI_BASE: str = os.path.join(_SCRIPT_DIR, GEN_CFG["output_dir"])
OUTPUT_DIRS = {
    "BoYaTCN": os.path.join(GENERATED_MIDI_BASE, "boyatcn"),
    "MusicMamba": os.path.join(GENERATED_MIDI_BASE, "musicmamba"),
}

# 生成记录 CSV
GENERATE_LIST_CSV: str = os.path.join(RESULTS_DIR, "generate_list_compound.csv")

# MIDI / 生成参数（来自 yaml）
DEFAULT_VELOCITY: int = int(GEN_CFG["velocity"])
TICKS_PER_BEAT: int = int(GEN_CFG["ticks_per_beat"])
DEFAULT_BPM: int = int(GEN_CFG["default_bpm"])
MAX_GEN_CONTEXT_LEN: int = int(GEN_CFG["max_gen_context_len"])


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

# MIDI 乐器程序号（来自 yaml；24 = Nylon Guitar，GM 中最接近南音琵琶的拨弦音色）
MIDI_PROGRAM: int = int(GEN_CFG["midi_program"])

# 自回归最小 prompt 长度（取 GT 前 N 个复合事件作为条件）
# E1 阶段不做生成实验，该值仅作为默认参数；如需变更请写入 yaml。
MIN_PROMPT_LEN: int = 16

# 调式五声音阶引导 keep-in-scale：0=关闭。
# E1 阶段统一由模型自身学习，默认关闭（如需调参请写入 yaml，勿在代码中硬编码）。
MODE_SCALE_GUIDANCE: float = 0.0
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

def _filter_and_sample(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
    generator: Optional[torch.Generator] = None,
) -> int:
    """对单路属性 logits 做 temperature → top-k → top-p 过滤后多项式采样。

    安全兜底：若过滤后所有 logit 均为 -inf（极端情况），退化为均匀分布采样，
    避免 softmax 产生 NaN。

    参数:
        generator: E4 新增。传入独立的 torch.Generator 时，采样随机流由该发生器
                   决定，与全局 RNG 解耦 → 同一 (权重, 采样参数, 随机种子) 组合可复现。
    """
    logits = logits / max(temperature, 1e-6)

    if top_k > 0 and top_k < logits.numel():
        top_k_vals, top_k_idx = torch.topk(logits, k=min(top_k, logits.numel()))
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(0, top_k_idx, top_k_vals)
        logits = mask

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative_probs > top_p
        remove[0] = False                      # 始终保留至少一个候选
        logits[sorted_indices[remove]] = float("-inf")

    if torch.all(torch.isinf(logits)):
        logits = torch.zeros_like(logits)

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def generate_compound_autoregressive(
    model: torch.nn.Module,
    prompt_pitch_ids: List[int],
    prompt_duration_ids: List[int],
    prompt_shift_ids: List[int],
    target_len: int,
    device: torch.device,
    codec: CompoundEventCodec = CODEC,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[Dict[str, int]] = None,
    generator: Optional[torch.Generator] = None,
    prepend_bos: bool = True,
) -> Dict[str, List[int]]:
    """
    复合事件口径的自回归生成：每一步**同时采样** pitch / duration / time-shift。

    与旧版（仅 pitch 单 token + 单独回归 duration）的本质区别：
      - 旧版：每个 token 的 onset = 前一个音的时值接龙排布，无法表达休止/气口
      - 新版：onset 由采样到的 time-shift 累加得到 → 可真实表达撩拍停顿

    流程：
      1. 以 GT 前 N 个复合事件三元组作为 prompt
      2. 每步把已生成的三路 ID 序列送入 forward(pitch_ids, duration_ids, shift_ids)
      3. 取最后位置的三路 logits（pitch / duration / shift），各自过滤后采样
      4. 追加到序列，重复至目标长度

    注意：
      - shift 采样结果决定下一个音符相对当前的 onset 偏移；
      - 序列末尾一步的 shift 在训练侧是伪标签，生成时该值不会被使用（最后一个音符之后
        没有下一个音符），仅保留在序列中以 succeeded 长度一致。

    E4 新增参数（只影响采样行为，不改变序列长度/prompt 规则）：
        temperature: 三路共用的采样温度；None 时沿用 e1 yaml 的分路默认值
        top_p:       三路共用的 nucleus 阈值；None 时沿用 e1 yaml 默认值
        top_k:       {"pitch": int, "duration": int, "shift": int}；None 时沿用默认值
        generator:   独立的 torch.Generator —— 用于固定随机种子下的结果复现
        prepend_bos: 输入序列是否以各路 BOS 开头（**True 时与训练样本输入格式一致**）

    返回:
        {"pitch_ids": [...], "duration_ids": [...], "shift_ids": [...]}（长度 = target_len）
    """
    p_ids: List[int] = list(prompt_pitch_ids)
    d_ids: List[int] = list(prompt_duration_ids)
    s_ids: List[int] = list(prompt_shift_ids)
    prompt_len = len(p_ids)
    actual_target = min(target_len, MAX_SEQ_LEN)

    # E4：三路采样超参（未显式传入时沿用 e1 yaml 的分路默认值 → 旧行为逐位不变）
    t_pitch = float(PITCH_TEMPERATURE) if temperature is None else float(temperature)
    t_duration = float(DURATION_TEMPERATURE) if temperature is None else float(temperature)
    t_shift = float(SHIFT_TEMPERATURE) if temperature is None else float(temperature)
    p_top_p = float(PITCH_TOP_P) if top_p is None else float(top_p)
    p_duration_p = float(DURATION_TOP_P) if top_p is None else float(top_p)
    p_shift_p = float(SHIFT_TOP_P) if top_p is None else float(top_p)
    if top_k is None:
        k_pitch, k_duration, k_shift = PITCH_TOP_K, DURATION_TOP_K, SHIFT_TOP_K
    else:
        k_pitch = int(top_k.get("pitch", PITCH_TOP_K))
        k_duration = int(top_k.get("duration", DURATION_TOP_K))
        k_shift = int(top_k.get("shift", SHIFT_TOP_K))

    with torch.no_grad():
        for _ in range(prompt_len, actual_target):
            if len(p_ids) > MAX_GEN_CONTEXT_LEN:
                ctx_p = p_ids[-MAX_GEN_CONTEXT_LEN:]
                ctx_d = d_ids[-MAX_GEN_CONTEXT_LEN:]
                ctx_s = s_ids[-MAX_GEN_CONTEXT_LEN:]
            else:
                ctx_p, ctx_d, ctx_s = p_ids, d_ids, s_ids

            # E4：与训练样本输入格式对齐 → 输入 = [BOS] + 事件序列
            if prepend_bos:
                ctx_p = [codec.pitch_bos_id] + list(ctx_p)
                ctx_d = [codec.duration_bos_id] + list(ctx_d)
                ctx_s = [codec.shift_bos_id] + list(ctx_s)

            input_p = torch.tensor([ctx_p], dtype=torch.long, device=device)
            input_d = torch.tensor([ctx_d], dtype=torch.long, device=device)
            input_s = torch.tensor([ctx_s], dtype=torch.long, device=device)

            output: dict = model(input_p, input_d, input_s)

            last_pitch_logits = output["pred_pitch_logits"][0, -1, :]
            last_duration_logits = output["pred_duration_logits"][0, -1, :]
            last_shift_logits = output["pred_shift_logits"][0, -1, :]

            next_pitch = _filter_and_sample(
                last_pitch_logits, t_pitch, p_top_p, k_pitch, generator
            )
            next_duration = _filter_and_sample(
                last_duration_logits, t_duration, p_duration_p, k_duration, generator
            )
            next_shift = _filter_and_sample(
                last_shift_logits, t_shift, p_shift_p, k_shift, generator
            )

            p_ids.append(next_pitch)
            d_ids.append(next_duration)
            s_ids.append(next_shift)

    return {"pitch_ids": p_ids, "duration_ids": d_ids, "shift_ids": s_ids}


# ==============================================================================
# 四、复合事件 → 音符序列 → MIDI 写入（E1 重写：绝对时间事件流）
# ==============================================================================

def events_to_notes(
    pitch_ids: List[int],
    duration_ids: List[int],
    shift_ids: List[int],
    codec: CompoundEventCodec = CODEC,
) -> List[dict]:
    """复合事件三元组 → 音符列表；**onset 由 time-shift 累加**得到。

    参数:
        pitch_ids / duration_ids / shift_ids: 三路等长 ID 序列
        codec: 复合事件编解码器
    返回:
        [{"note": int, "duration": float, "start_time": float}, ...]
    """
    return codec.decode_events(pitch_ids, duration_ids, shift_ids, start_offset_sec=0.0)


def notes_to_midi(
    notes: List[dict],
    output_path: str,
    ticks_per_beat: int = TICKS_PER_BEAT,
    tempo_bpm: int = DEFAULT_BPM,
    velocity: int = DEFAULT_VELOCITY,
    velocities: Optional[List[int]] = None,
    program: int = MIDI_PROGRAM,
) -> Dict[str, int]:
    """
    音符列表（含绝对 onset，秒）→ 标准 MIDI 文件 (Format 0, 单轨道)。

    与旧版接龙写法的区别：
      旧版 token_to_midi 按"上一音时值"依次排布，隐含 IOI == duration，无法表达休止；
      本函数按 **绝对时间** 构造事件流：先把所有 note_on / note_off 连同其绝对 tick
      时间放进列表，排序后按相邻 delta 写入，休止（time-shift 产生的空隙）自然保留。

    同音重叠防护：
      MIDI 的 note_on/note_off 按 (channel, pitch) 配对。若同一音高出现两个重叠区间，
      下游 pretty_midi / mido 解析会错配丢音符（历史上曾把 1025 个音符解析成 792 个）。
      因此这里对"同音高相邻且重叠"的情况，把前一个音截断到新音的起点（至少保留 1 tick），
      并在返回值中统计被截断的数量，供自测报告记录。

    参数:
        notes:           [{"note": int, "duration": float, "start_time": float}, ...]
        output_path:     输出 .mid 路径
        ticks_per_beat:  MIDI 分辨率 (PPQN)
        tempo_bpm:       速度（秒 → tick 换算用）
        velocity:        统一力度
        velocities:      每音符独立力度列表（可选，优先于 velocity）
        program:         GM 音色号
    返回:
        统计信息 {"num_notes": N, "truncated_overlap": M, "num_events": K}
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    sec_per_tick = 60.0 / float(tempo_bpm) / float(ticks_per_beat)
    note_velocities = velocities if velocities is not None else [velocity] * len(notes)

    # 事件元组：(note_off 绝对 tick, note_on 绝对 tick, 事件类型, pitch, velocity)
    events_list: List[Tuple[int, int, str, int, int]] = []

    last_off_tick_by_pitch: Dict[int, int] = {}
    truncated_overlap = 0

    for i, n in enumerate(notes):
        pitch = int(max(0, min(127, int(n["note"]))))
        vel = int(max(1, min(127, note_velocities[i] if i < len(note_velocities) else velocity)))
        start_tick = int(round(float(n["start_time"]) / sec_per_tick))
        end_tick = int(round((float(n["start_time"]) + max(float(n["duration"]), 0.0)) / sec_per_tick))
        if end_tick <= start_tick:
            end_tick = start_tick + 1                     # 至少 1 tick

        prev_off = last_off_tick_by_pitch.get(pitch)
        if prev_off is not None and start_tick < prev_off:
            # 同音重叠：把前一个音（已在 events 中）的 note_off 提前到当前音符起点，
            # 保证 note_off 至少比它自己的 note_on 晚 1 tick，消除配对歧义。
            truncated_overlap += 1
            for idx in range(len(events_list) - 1, -1, -1):
                ev = events_list[idx]
                if ev[2] == "off" and ev[3] == pitch and ev[0] == prev_off:
                    events_list[idx] = (max(ev[1] + 1, start_tick), ev[1], "off", pitch, 0)
                    break

        events_list.append((start_tick, start_tick, "on", pitch, vel))
        events_list.append((end_tick, start_tick, "off", pitch, 0))
        last_off_tick_by_pitch[pitch] = end_tick

    # 排序：同一 tick 先 note_off 后 note_on（避免零长度/同音歧义）
    events_list.sort(key=lambda e: (e[0], 0 if e[2] == "off" else 1, e[3]))

    mid = mido.MidiFile(type=0, ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    track.append(mido.Message("program_change", program=program, channel=0, time=0))

    last_tick = 0
    for abs_tick, _, ev_type, pitch, vel in events_list:
        delta = max(0, abs_tick - last_tick)
        if ev_type == "on":
            track.append(mido.Message("note_on", note=pitch, velocity=vel, time=delta))
        else:
            track.append(mido.Message("note_off", note=pitch, velocity=0, time=delta))
        last_tick = abs_tick

    track.append(mido.MetaMessage("end_of_track", time=0))
    mid.save(output_path)

    return {
        "num_notes": len(notes),
        "truncated_overlap": truncated_overlap,
        "num_events": len(track),
    }


def events_to_midi(
    pitch_ids: List[int],
    duration_ids: List[int],
    shift_ids: List[int],
    output_path: str,
    codec: CompoundEventCodec = CODEC,
    ticks_per_beat: int = TICKS_PER_BEAT,
    tempo_bpm: int = DEFAULT_BPM,
    velocity: int = DEFAULT_VELOCITY,
    velocities: Optional[List[int]] = None,
) -> Dict[str, int]:
    """复合事件三元组 → MIDI 文件（events_to_notes + notes_to_midi 的组合入口）。"""
    notes = events_to_notes(pitch_ids, duration_ids, shift_ids, codec=codec)
    return notes_to_midi(
        notes,
        output_path=output_path,
        ticks_per_beat=ticks_per_beat,
        tempo_bpm=tempo_bpm,
        velocity=velocity,
        velocities=velocities,
    )


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
            "num_notes", "total_sec", "output_midi_path",
        ])


def append_generate_row(
    csv_path: str,
    song_id: str,
    song_name: str,
    model_name: str,
    num_notes: int,
    total_sec: float,
    midi_path: str,
) -> None:
    """在生成记录 CSV 末尾追加一行。"""
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([song_id, song_name, model_name, num_notes, f"{total_sec:.3f}", midi_path])


# ==============================================================================
# E4：固定权重 + temperature / top-p 网格搜索
# ==============================================================================
#
# ⚠ E4 硬性约束（写在此处以免后续误改）：
#   1. 不训练、不微调、不改权重 —— 只加载 E3-L3 三组种子的 best-val 早停 checkpoint；
#   2. prompt 规则固定：每个验证窗口取前 16 个 GT 事件，自回归续写到总长度 256 事件；
#   3. 禁止任何 MIDI 后处理 / 时值钳位修正（配置 postprocess.* 全 false），评估用原生输出；
#   4. 每条序列的随机流由独立的 torch.Generator 决定，种子由 _rng_seed() 显式计算 → 可复现。
# ==============================================================================

E4_DEFAULT_CONFIG: str = os.path.join(_SCRIPT_DIR, "configs", "e4_sampling_search.yaml")


def build_midi_name(song_id: str, window_index: int, model_seed: int,
                    temperature: float, top_p: float) -> str:
    """E4 生成产物命名：{song_id}_w{窗口号}_seed{模型种子}_T{temp}_p{topp}.mid"""
    t_tag = f"{float(temperature):g}".replace(".", "")
    p_tag = f"{float(top_p):g}".replace(".", "")
    return f"{song_id}_w{window_index}_seed{model_seed}_T{t_tag}_p{p_tag}.mid"


def _rng_seed(base: int, model_seed: int, grid_index: int, window_index: int,
              seed_slot: int = 97) -> int:
    """E4 每条序列独立的随机种子（确定性公式，保证可复现）。

        rng_seed = base + model_seed * seed_slot ** 3 + grid_index * seed_slot ** 2 + window_index

    说明：模型种子之间的间隔足够大，避免不同 (模型种子, 网格点, 窗口) 组合碰撞。
    """
    return int(base) + int(model_seed) * (seed_slot ** 3) + int(grid_index) * (seed_slot ** 2) + int(window_index)


def load_val_windows(cfg: dict, verbose: bool = True) -> List[Dict[str, Any]]:
    """构建 E2/E3 口径的验证窗口（非重叠，10 个）并返回 GT 事件列表。

    每个元素:
        {"song_id", "window_index", "pitch_ids", "duration_ids", "shift_ids"}
        其中 *_ids 为**训练同源顺序**（onset 升序）的事件序列，长度 = window_size。
    """
    wcfg = cfg["window"]
    window_size = int(wcfg["window_size_events"])
    val_ds = CompoundEventWindowDataset(
        song_ids=VAL_SPLIT_IDS,
        split=wcfg.get("split", "val"),
        cfg=cfg,
    )
    _ = verbose  # 保留参数位（窗口统计由 dataset 内部 verbose=True 打印）
    windows: List[Dict[str, Any]] = []
    for i in range(len(val_ds)):
        sample = val_ds[i]
        # 目标序列 = target_*_ids[0:window_size]（末尾一个元素是 EOS）
        tgt_p = sample["target_pitch_ids"][:window_size].tolist()
        tgt_d = sample["target_duration_ids"][:window_size].tolist()
        tgt_s = sample["target_shift_ids"][:window_size].tolist()
        # 窗口末尾 shift 在训练侧是伪标签（-100），GT 侧统一还原为 PAD(0)
        tgt_s[-1] = 0
        windows.append({
            "song_id": sample["song_id"],
            "window_index": int(sample["window_index"]),
            "start": int(sample["start"]),
            "end": int(sample["end"]),
            "pitch_ids": tgt_p,
            "duration_ids": tgt_d,
            "shift_ids": tgt_s,
        })
    if verbose:
        print(f"[E4] 验证窗口: {len(windows)} 个 "
              f"({', '.join(sorted({w['song_id'] for w in windows}))}), "
              f"窗口长度={window_size} 事件")
    return windows


def load_e4_model(ckpt_path: str, device: torch.device, strict: bool = True):
    """加载 E3-L3 的 NanyinBoYaTCN best-val 权重。"""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[E4] 权重文件不存在: {ckpt_path}")
    model = NanyinBoYaTCN().to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    model.eval()
    print(f"[E4] 已加载权重: {ckpt_path} (epoch={checkpoint.get('epoch', '?')}, "
          f"val_loss={checkpoint.get('val_loss', float('nan')):.4f})")
    return model


def run_sampling_search(
    cfg: Optional[dict] = None,
    model_seeds: Optional[Sequence[int]] = None,
    temperature_grid: Optional[Sequence[float]] = None,
    top_p_grid: Optional[Sequence[float]] = None,
    device: Optional[torch.device] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """E4 主流程：固定权重 × temperature/top-p 网格 × 10 个验证窗口 → MIDI + 清单 CSV。

    参数:
        cfg:              E4 配置字典（None 时读 configs/e4_sampling_search.yaml）
        model_seeds:      覆盖配置里的模型训练种子列表
        temperature_grid: 覆盖温度网格
        top_p_grid:       覆盖 top-p 网格
        device:           推理设备
        overwrite:        已存在的 MIDI 是否重新生成
    返回:
        {"num_generated": int, "csv": str, "midi_dir": str}
    """
    cfg = cfg if cfg is not None else load_config(E4_DEFAULT_CONFIG)
    wcfg = cfg["window"]
    scfg = cfg["sampling"]
    acfg = cfg["weights"]
    outcfg = cfg["outputs"]

    model_seeds = list(model_seeds) if model_seeds is not None else list(acfg["model_seeds"])
    temps = list(temperature_grid) if temperature_grid is not None else list(scfg["temperature_grid"])
    topps = list(top_p_grid) if top_p_grid is not None else list(scfg["top_p_grid"])
    top_k_map = dict(scfg["top_k"])
    prompt_len = int(wcfg["prompt_len_events"])
    target_len = int(wcfg["target_len_events"])
    prepend_bos = bool(wcfg.get("prepend_bos", True))
    rng_base = int(scfg["rng_base_seed"])
    ckpt_tpl = acfg["checkpoint_template"]
    strict = bool(acfg.get("strict_load", True))

    midi_dir = os.path.join(_SCRIPT_DIR, str(outcfg["midi_dir"]))
    csv_path = os.path.join(_SCRIPT_DIR, str(outcfg["generate_list_csv"]))
    os.makedirs(midi_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("E4: 固定权重 + temperature / top-p 采样超参网格搜索")
    print("=" * 70)
    print(f"  推理设备      : {device}")
    print(f"  模型权重      : {acfg['source_stage']} best-val × 种子 {model_seeds}")
    print(f"  网格          : temperature={temps} × top_p={topps} → {len(temps) * len(topps)} 组")
    print(f"  prompt 规则   : 每窗口前 {prompt_len} 个 GT 事件 → 续写到总长 {target_len} 事件")
    print(f"  后处理        : 全部关闭（原生输出）")
    print(f"  MIDI 输出目录 : {midi_dir}")

    windows = load_val_windows(cfg, verbose=True)

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "song_id", "window_index", "window_start_event", "window_end_event",
            "model_seed", "checkpoint", "temperature", "top_p",
            "top_k_pitch", "top_k_duration", "top_k_shift",
            "rng_seed", "prompt_len", "target_len", "num_events",
            "output_midi_path",
        ])

    generated = 0
    t_start = time.time()

    for model_seed in model_seeds:
        ckpt_path = ckpt_tpl.format(seed=model_seed)
        model = load_e4_model(ckpt_path, device, strict=strict)

        grid_pairs = [(t, p) for t in temps for p in topps]
        for g_i, (temp, tpp) in enumerate(grid_pairs):
            print(f"[E4] seed={model_seed} T={temp:g} p={tpp:g} "
                  f"({g_i + 1}/{len(grid_pairs)})")
            rows: List[List[Any]] = []
            for win in windows:
                fname = build_midi_name(
                    win["song_id"], win["window_index"], model_seed, temp, tpp
                )
                midi_path = os.path.join(midi_dir, fname)
                rng = _rng_seed(rng_base, model_seed, g_i, win["window_index"])

                if overwrite or not os.path.exists(midi_path):
                    gen = torch.Generator(device=device.type)
                    gen.manual_seed(rng)
                    events = generate_compound_autoregressive(
                        model=model,
                        prompt_pitch_ids=win["pitch_ids"][:prompt_len],
                        prompt_duration_ids=win["duration_ids"][:prompt_len],
                        prompt_shift_ids=win["shift_ids"][:prompt_len],
                        target_len=target_len,
                        device=device,
                        codec=CODEC,
                        temperature=temp,
                        top_p=tpp,
                        top_k=top_k_map,
                        generator=gen,
                        prepend_bos=prepend_bos,
                    )
                    # ---- 原生输出直写 MIDI：**无任何后处理 / 钳位** ----
                    notes = events_to_notes(
                        events["pitch_ids"], events["duration_ids"],
                        events["shift_ids"], codec=CODEC,
                    )
                    notes_to_midi(
                        notes, output_path=midi_path,
                        velocities=compute_velocities(len(notes)),
                    )

                rows.append([
                    win["song_id"], win["window_index"], win["start"], win["end"],
                    model_seed, ckpt_path, temp, tpp,
                    top_k_map.get("pitch", 0), top_k_map.get("duration", 0), top_k_map.get("shift", 0),
                    rng, prompt_len, target_len, target_len, midi_path,
                ])

            with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerows(rows)
            generated += len(rows)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"\n[E4] 网格生成完成: {generated} 条 MIDI, 耗时 {elapsed / 60:.1f} 分钟")
    print(f"  清单 CSV: {csv_path}")
    return {"num_generated": generated, "csv": csv_path, "midi_dir": midi_dir}


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

    missing_ckpts = [p for p in MODEL_CHECKPOINTS.values() if not os.path.exists(p)]
    if missing_ckpts:
        # E1 阶段只完成复合事件编解码改造，尚未训练复合事件模型，
        # 因此不存在可用权重。此处明确提示并安全退出，**不生成、不伪造任何 MIDI**。
        print("\n" + "!" * 70)
        print("[generate] 未找到复合事件模型权重，跳过生成（E1 不产出复合事件 MIDI）")
        for p in missing_ckpts:
            print(f"    缺失: {p}")
        print("[generate] 说明：E1 阶段仅完成 MIDI → 复合事件 → MIDI 往返编解码改造与自测；")
        print("           复合事件模型的训练在后续阶段（E2+）进行，届时再执行本脚本生成。")
        print("!" * 70)
        return

    for model_name, (loader_fn, ckpt_path) in model_loaders.items():
        print(f"\n{'=' * 70}")
        print(f"[generate] 开始模型: {model_name}")
        print(f"[generate] 权重路径: {ckpt_path}")
        print(f"{'=' * 70}")

        # 加载模型
        model = loader_fn(ckpt_path, device)

        out_dir = OUTPUT_DIRS[model_name]

        # 逐曲目生成
        for song_id, input_tokens, target_dict in val_data:
            song_name = SONG_CHINESE_NAMES.get(song_id, song_id)
            # 目标事件数 = GT 音符数（复合事件口径：一步一个三元组）
            target_len = int(target_dict["target_pitch_ids"].shape[0])

            print(f"[generate]   {song_name} ({song_id}) [{model_name}]")

            # ---- 构建 prompt：GT 前 MIN_PROMPT_LEN 个复合事件三元组 ----
            prompt_len = min(MIN_PROMPT_LEN, target_len)
            prompt_pitch_ids = target_dict["target_pitch_ids"][:prompt_len].tolist()
            prompt_duration_ids = target_dict["target_duration_ids"][:prompt_len].tolist()
            prompt_shift_ids = target_dict["target_shift_ids"][:prompt_len].tolist()

            # ---- 复合事件自回归生成（同时采样 pitch / duration / time-shift）----
            events = generate_compound_autoregressive(
                model=model,
                prompt_pitch_ids=prompt_pitch_ids,
                prompt_duration_ids=prompt_duration_ids,
                prompt_shift_ids=prompt_shift_ids,
                target_len=target_len,
                device=device,
                codec=CODEC,
            )

            # ---- onset 由 time-shift 累加得到 ----
            decoded_notes = events_to_notes(
                events["pitch_ids"], events["duration_ids"], events["shift_ids"], codec=CODEC
            )
            total_sec = float(decoded_notes[-1]["start_time"] + decoded_notes[-1]["duration"]) if decoded_notes else 0.0
            print(f"[generate]   生成完成: {len(decoded_notes)} 音符, 总时长 {total_sec:.2f}s")

            # ---- 写入 MIDI（绝对时间事件流）----
            midi_filename = f"{song_id}_{model_name.lower()}.mid"
            midi_path = os.path.join(out_dir, midi_filename)
            stats = notes_to_midi(
                decoded_notes,
                output_path=midi_path,
                velocities=compute_velocities(len(decoded_notes)),
            )
            if stats["truncated_overlap"] > 0:
                print(f"[generate]   同音重叠截断: {stats['truncated_overlap']} 处")

            # ---- 追加生成记录 ----
            append_generate_row(
                csv_path=GENERATE_LIST_CSV,
                song_id=song_id,
                song_name=song_name,
                model_name=model_name,
                num_notes=len(decoded_notes),
                total_sec=total_sec,
                midi_path=midi_path,
            )

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
    print(f"  复合事件量化单位: {CODEC.time_unit_sec * 1000:.0f} ms/步，"
          f"duration {CODEC.duration_num_classes} 类 / shift {CODEC.shift_num_classes} 类")
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

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """命令行入口参数。"""
    parser = argparse.ArgumentParser(description="南音符号旋律推理生成")
    parser.add_argument(
        "--mode", default="default", choices=["default", "sampling_search"],
        help="default = E1 批量生成；sampling_search = E4 temperature/top-p 网格搜索",
    )
    parser.add_argument(
        "--config", default=None,
        help="覆盖层 YAML 路径（默认 default 模式不读；sampling_search 默认 configs/e4_sampling_search.yaml）",
    )
    parser.add_argument("--seeds", default=None, help="模型训练种子列表，逗号分隔（覆盖配置）")
    parser.add_argument("--temperatures", default=None, help="temperature 网格，逗号分隔（覆盖配置）")
    parser.add_argument("--topps", default=None, help="top-p 网格，逗号分隔（覆盖配置）")
    parser.add_argument("--overwrite", action="store_true", help="已存在的 MIDI 也重新生成")
    return parser.parse_args(argv)


if __name__ == "__main__":
    """
    独立运行说明:
      - 依赖: torch, mido, numpy (已存在于项目 venv)
      - 需先完成模型训练 (checkpoints 下存在 best_model.pt)
      - 生成结果写入 ./results/ 目录

    完整运行命令 (Windows):
      venv/Scripts/python.exe generate_music.py
      venv/Scripts/python.exe generate_music.py --mode sampling_search \
          --config configs/e4_sampling_search.yaml
    """
    # Windows GBK 终端兼容：强制 UTF-8 输出
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # pyright: ignore

    args = _parse_args()
    if args.mode == "sampling_search":
        cfg = load_config(args.config) if args.config else load_config(E4_DEFAULT_CONFIG)
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
        temps = [float(s) for s in args.temperatures.split(",")] if args.temperatures else None
        topps = [float(s) for s in args.topps.split(",")] if args.topps else None
        run_sampling_search(
            cfg=cfg, model_seeds=seeds, temperature_grid=temps,
            top_p_grid=topps, overwrite=args.overwrite,
        )
    else:
        main()
