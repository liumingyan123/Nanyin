"""
WAV → midi.json 转录脚本

从音频中自动提取音符序列，替换 complete_annotations.py 中的占位数据。

方法:
    1. librosa.pyin 连续基频(F0)追踪
    2. F0 → MIDI 音高(半音量化)
    3. 检测音高变化点作为音符边界
    4. 合并极短音符、平滑抖动
    5. 写入现有 midi.json 的 track[0].notes

用法:
    python wav_to_midi_json.py              # 处理所有曲目
    python wav_to_midi_json.py --song yizhixiangsi   # 只处理一首
    python wav_to_midi_json.py --dry-run    # 预览不写入
"""

import os
import json
import argparse
import numpy as np
import librosa
import time
import sys
from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


# ===========================================================================
# 配置
# ===========================================================================

BASE_DIR = r"d:\科研\NanyinPingGu\南音数据集\nanyin_final\structured_data"

# pyin 参数
FMIN = librosa.note_to_hz("C2")   # ~65 Hz — 南音琵琶最低音附近
FMAX = librosa.note_to_hz("C7")   # ~2093 Hz
SR_PROCESS = 11025                # 降采样以加速处理
HOP_LENGTH = 512                  # pyin hop, 512→每帧~46ms (时间精度足够, 大幅加速)

# 音符分割参数
MIN_NOTE_DURATION = 0.05          # 秒, 低于此值的音符合并到相邻
SMOOTH_WINDOW = 5                 # 中值滤波窗口(帧), 消除F0抖动
GAP_TOLERANCE = 0.05              # 秒, 两个音符间短间隙的容忍度(自动填补)

# voicing 参数
VOICED_THRESHOLD = 0.3            # pyin voiced_prob 低于此值视为静音/无音高

# 分块处理参数 (长音频分段处理)
CHUNK_SECONDS = 300               # 每块 5 分钟
CHUNK_OVERLAP = 2.0               # 块间重叠 2 秒(避免边界切断音符)


@dataclass
class Note:
    """单个音符"""
    pitch: int        # MIDI 音高 0-127
    start_time: float # 秒
    duration: float   # 秒
    velocity: int     # 力度 0-127


# ===========================================================================
# 核心: 音频 → 音符列表
# ===========================================================================

def hz_to_midi(freq: float) -> int:
    """频率转 MIDI 音高(四舍五入到最近半音)"""
    if freq <= 0 or np.isnan(freq):
        return -1
    midi = 12.0 * np.log2(freq / 440.0) + 69.0
    return int(round(midi))


def _f0_to_notes(
    f0: np.ndarray,
    voiced_flag: np.ndarray,
    voiced_prob: np.ndarray,
    frame_time: float,
    hop_length: int,
    sr: int,
    voiced_thresh: float,
    smooth_window: int,
    min_note_dur: float,
    gap_tol: float,
    time_offset: float = 0.0,
) -> List[Tuple[float, float, int]]:
    """
    将 pyin 输出的 F0 数组转换为 (start_s, end_s, pitch) 音符段列表。

    参数:
        time_offset: 该段在原始音频中的起始偏移(秒), 用于分块处理
    返回:
        [(start_s, end_s, midi_pitch), ...]
    """
    n_frames = len(f0)

    # F0 → MIDI 音高序列
    midi_seq = np.full(n_frames, -1, dtype=int)
    for i in range(n_frames):
        if voiced_flag[i] and voiced_prob[i] >= voiced_thresh:
            midi_seq[i] = hz_to_midi(f0[i])

    n_voiced = np.sum(midi_seq >= 0)
    if n_voiced < 3:
        return []

    # 中值滤波平滑
    if smooth_window > 1:
        orig = midi_seq.copy()
        half_w = smooth_window // 2
        for i in range(n_frames):
            lo = max(0, i - half_w)
            hi = min(n_frames, i + half_w + 1)
            valid = orig[lo:hi]
            valid = valid[valid >= 0]
            midi_seq[i] = int(np.median(valid)) if len(valid) > 0 else -1

    # 检测音高变化 → 音符边界
    raw = []  # (start_frame, end_frame, pitch)
    in_note = False
    note_start = 0
    note_pitch = -1

    for i in range(n_frames):
        if midi_seq[i] >= 0:
            if not in_note:
                note_start = i
                note_pitch = midi_seq[i]
                in_note = True
            elif midi_seq[i] != note_pitch:
                raw.append((note_start, i, note_pitch))
                note_start = i
                note_pitch = midi_seq[i]
        else:
            if in_note:
                raw.append((note_start, i, note_pitch))
                in_note = False
    if in_note:
        raw.append((note_start, n_frames, note_pitch))

    # 转为秒 + 后处理
    segments: List[Tuple[float, float, int]] = []
    for sf, ef, pitch in raw:
        ss = time_offset + sf * frame_time
        es = time_offset + ef * frame_time
        if es - ss >= min_note_dur:
            segments.append((ss, es, pitch))

    # 合并相邻同音高+短间隙
    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        if seg[2] == prev[2] and (seg[0] - prev[1]) <= gap_tol:
            merged[-1] = (prev[0], seg[1], prev[2])
        else:
            merged.append(seg)

    return [s for s in merged if (s[1] - s[0]) >= min_note_dur]


def extract_notes_from_audio(
    audio_path: str,
    fmin: float = FMIN,
    fmax: float = FMAX,
    sr_process: int = SR_PROCESS,
    hop_length: int = HOP_LENGTH,
    min_note_dur: float = MIN_NOTE_DURATION,
    smooth_window: int = SMOOTH_WINDOW,
    gap_tol: float = GAP_TOLERANCE,
    voiced_thresh: float = VOICED_THRESHOLD,
    chunk_seconds: float = CHUNK_SECONDS,
    chunk_overlap: float = CHUNK_OVERLAP,
    max_duration: float = None,
) -> List[Note]:
    """
    从音频文件中提取音符序列 (长音频自动分块处理)。

    参数:
        audio_path:   音频文件路径
        max_duration: 最大处理时长(秒), None=全部, 用于快速测试
    """
    t0 = time.time()

    # ---- 1. 加载并降采样 ----
    print(f"    加载音频: {os.path.basename(audio_path)}")
    y, sr_orig = librosa.load(audio_path, sr=None, mono=True)
    total_dur = len(y) / sr_orig
    print(f"    原始: {total_dur:.1f}s @ {sr_orig}Hz")

    if max_duration is not None and max_duration < total_dur:
        n_keep = int(max_duration * sr_orig)
        y = y[:n_keep]
        total_dur = max_duration
        print(f"    截取前 {max_duration:.0f}s 用于测试")

    y = librosa.resample(y, orig_sr=sr_orig, target_sr=sr_process)
    sr = sr_process
    frame_time = hop_length / sr
    print(f"    降采样: {sr}Hz, hop={hop_length} → 每帧 {frame_time*1000:.1f}ms")

    # ---- 2. 分块 pyin 处理 ----
    chunk_samples = int(chunk_seconds * sr)
    overlap_samples = int(chunk_overlap * sr)

    all_segments: List[Tuple[float, float, int]] = []
    n_samples = len(y)

    if n_samples <= chunk_samples:
        # 短音频: 一次性处理
        n_frames = 1 + (n_samples - hop_length) // hop_length
        print(f"    pyin F0追踪: ~{n_frames} frames (一次性)...")
        f0, vf, vp = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
        if f0 is not None and len(f0) > 0:
            segs = _f0_to_notes(f0, vf, vp, frame_time, hop_length, sr,
                                voiced_thresh, smooth_window, min_note_dur, gap_tol, 0.0)
            all_segments.extend(segs)
            print(f"    pyin 完成: {len(f0)} frames, {len(segs)} 个音符段")
    else:
        # 长音频: 分块处理
        n_chunks = int(np.ceil(n_samples / (chunk_samples - overlap_samples)))
        print(f"    分块 pyin F0追踪: {n_chunks} 个块...")

        for ci in range(n_chunks):
            chunk_start = max(0, ci * (chunk_samples - overlap_samples))
            chunk_end = min(n_samples, chunk_start + chunk_samples)
            chunk = y[chunk_start:chunk_end]
            chunk_offset = chunk_start / sr  # 该块在原音频中的起始时间

            n_chunk_frames = 1 + (len(chunk) - hop_length) // hop_length
            if n_chunk_frames < 10:
                continue

            print(f"      块 {ci+1}/{n_chunks}: {chunk_offset:.0f}s - {chunk_end/sr:.0f}s "
                  f"({len(chunk)/sr:.1f}s, ~{n_chunk_frames} frames)...", end=" ", flush=True)
            t_chunk = time.time()

            f0, vf, vp = librosa.pyin(chunk, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
            if f0 is not None and len(f0) > 0:
                segs = _f0_to_notes(f0, vf, vp, frame_time, hop_length, sr,
                                    voiced_thresh, smooth_window, min_note_dur, gap_tol,
                                    time_offset=chunk_offset)
                all_segments.extend(segs)
                print(f"{len(f0)} frames, {len(segs)} 段, {time.time()-t_chunk:.1f}s")
            else:
                print("空")

    if not all_segments:
        print("    ⚠ 未提取到任何音符")
        return []

    # ---- 3. 跨块去重(相邻块重叠区可能有重复音符) ----
    all_segments.sort(key=lambda s: s[0])
    deduped: List[Tuple[float, float, int]] = []
    for seg in all_segments:
        if not deduped:
            deduped.append(seg)
            continue
        prev = deduped[-1]
        # 时间重叠且同音高 → 跳过
        if seg[0] < prev[1] and seg[2] == prev[2]:
            deduped[-1] = (prev[0], max(prev[1], seg[1]), prev[2])
        else:
            deduped.append(seg)

    all_segments = deduped

    # ---- 4. 计算力度(基于RMS能量) ----
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_max = np.max(rms) if np.max(rms) > 0 else 1.0

    notes = []
    for start_s, end_s, pitch in all_segments:
        sf = int(start_s / frame_time)
        ef = min(int(end_s / frame_time), len(rms) - 1)
        if ef > sf:
            seg_rms = float(np.mean(rms[sf:ef]))
        else:
            seg_rms = float(rms[min(sf, len(rms) - 1)])
        velocity = int(np.clip(seg_rms / rms_max * 100 + 30, 30, 127))

        notes.append(Note(
            pitch=int(pitch),
            start_time=round(start_s, 4),
            duration=round(end_s - start_s, 4),
            velocity=velocity,
        ))

    elapsed = time.time() - t0
    print(f"    转录完成: {len(notes)} 个音符, 总耗时 {elapsed:.1f}s")
    return notes


# ===========================================================================
# 写入 midi.json
# ===========================================================================

def update_midi_json(song_dir: str, notes: List[Note]) -> bool:
    """
    将转录出的音符写入该曲目的 midi.json, 只更新 track[0].notes 和 note_count。

    参数:
        song_dir: 曲目目录路径
        notes:    提取出的音符列表
    返回:
        是否成功
    """
    # 找到 midi.json 文件
    midi_files = [f for f in os.listdir(song_dir) if f.endswith("_midi.json")]
    if not midi_files:
        print(f"    ⚠ 找不到 _midi.json 文件")
        return False

    midi_path = os.path.join(song_dir, midi_files[0])

    # 读取现有文件
    with open(midi_path, "r", encoding="utf-8") as f:
        midi_data = json.load(f)

    # 转换 Note 对象为字典
    notes_dict = [
        {
            "note": n.pitch,
            "start_time": n.start_time,
            "duration": n.duration,
            "velocity": n.velocity,
        }
        for n in notes
    ]

    # 更新 track[0]
    tracks = midi_data.get("tracks", [])
    if len(tracks) > 0:
        tracks[0]["notes"] = notes_dict
        tracks[0]["note_count"] = len(notes_dict)
        if notes_dict:
            all_pitches = [n["note"] for n in notes_dict]
            tracks[0]["note_range"] = [min(all_pitches), max(all_pitches)]

    # 更新总时长
    if notes_dict:
        midi_data["total_duration_sec"] = round(
            notes_dict[-1]["start_time"] + notes_dict[-1]["duration"], 2
        )

    # 写回文件
    with open(midi_path, "w", encoding="utf-8") as f:
        json.dump(midi_data, f, ensure_ascii=False, indent=4)

    # 验证
    with open(midi_path, "r", encoding="utf-8") as f:
        verify = json.load(f)
    track0 = verify.get("tracks", [{}])[0]
    print(f"    已写入: {len(track0.get('notes', []))} 个音符 → {midi_files[0]}")

    return True


# ===========================================================================
# 批量处理
# ===========================================================================

def process_single(song_id: str, dry_run: bool = False, max_duration: float = None) -> bool:
    """处理单首曲目"""
    song_dir = os.path.join(BASE_DIR, song_id)
    if not os.path.isdir(song_dir):
        print(f"⚠ 目录不存在: {song_dir}")
        return False

    # 找音频文件
    audio_path = None
    for name in ["audio.wav", "audio.mp3"]:
        p = os.path.join(song_dir, name)
        if os.path.exists(p):
            audio_path = p
            break

    if audio_path is None:
        print(f"⚠ [{song_id}] 未找到音频文件")
        return False

    print(f"\n{'='*60}")
    print(f"转录: {song_id}")
    print(f"{'='*60}")

    try:
        notes = extract_notes_from_audio(audio_path, max_duration=max_duration)
    except Exception as e:
        print(f"    ✗ 转录失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    if len(notes) == 0:
        print(f"    ✗ 未提取到任何音符")
        return False

    # 打印统计
    pitches = [n.pitch for n in notes]
    durations = [n.duration for n in notes]
    total_dur = sum(durations)
    print(f"    音符统计: {len(notes)} 个, 音域 [{min(pitches)}, {max(pitches)}], "
          f"平均时长 {np.mean(durations):.3f}s, 总长 {total_dur:.1f}s")

    # 打印前10个音符预览
    print(f"    前10个音符: {[(n.pitch, f'{n.duration:.2f}s') for n in notes[:10]]}")

    if dry_run:
        print(f"    [DRY-RUN] 未写入文件")
        return True

    return update_midi_json(song_dir, notes)


def process_all(dry_run: bool = False, max_duration: float = None):
    """处理所有曲目"""
    song_ids = sorted([
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
    ])

    print(f"找到 {len(song_ids)} 首曲目:")
    for s in song_ids:
        print(f"  - {s}")
    print()

    success = 0
    failed = []
    for i, song_id in enumerate(song_ids, 1):
        print(f"\n[{i}/{len(song_ids)}]", end=" ")
        try:
            if process_single(song_id, dry_run=dry_run, max_duration=max_duration):
                success += 1
            else:
                failed.append(song_id)  
        except KeyboardInterrupt:
            print("\n用户中断")
            break
        except Exception as e:
            print(f"    ✗ {type(e).__name__}: {e}")
            failed.append(song_id)

    print(f"\n{'='*60}")
    print(f"完成: {success}/{len(song_ids)} 成功")
    if failed:
        print(f"失败: {failed}")
    print(f"{'='*60}")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="WAV → midi.json 转录: 从音频自动提取音符序列"
    )
    parser.add_argument(
        "--song", type=str, default=None,
        help="只处理指定曲目ID (如 yizhixiangsi)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="预览模式, 不写入文件"
    )
    parser.add_argument(
        "--max-duration", type=float, default=None,
        help="最大处理时长(秒), 用于快速测试 (如 --max-duration 60)"
    )
    args = parser.parse_args()

    if args.song:
        process_single(args.song, dry_run=args.dry_run, max_duration=args.max_duration)
    else:
        process_all(dry_run=args.dry_run, max_duration=args.max_duration)


if __name__ == "__main__":
    main()
