#!/usr/bin/env python3
"""
Multi-Model MIDI Transcription Pipeline for Nanyin Music.
双模型转录管道: Basic Pitch | YourMT3+风格(NMF)
"""

import argparse
import io
import json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import librosa
import soundfile as sf
import pretty_midi

warnings.filterwarnings("ignore")

def _utf8_stream(stream):
    """将流安全切换为 UTF-8；已是 TextIOWrapper 时直接 reconfigure（幂等），
    避免反复 io.TextIOWrapper(...) 双重包装导致流悬空（lost sys.stderr）。"""
    if stream is None:
        return stream
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
        return stream
    except Exception:
        try:
            return io.TextIOWrapper(stream.buffer, encoding="utf-8", errors="replace")
        except Exception:
            return stream

# Fix GBK encoding on Windows
if sys.platform == "win32":
    sys.stdout = _utf8_stream(sys.stdout)
    sys.stderr = _utf8_stream(sys.stderr)

# ─── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output")
VENV38_PYTHON = os.path.join(PROJECT_ROOT, "venv38", "Scripts", "python.exe")
BP_RUNNER = os.path.join(PROJECT_ROOT, "basic_pitch_runner.py")

# Audio params
SR = 22050
HOP_LENGTH = 512

# ─── Helpers ─────────────────────────────────────────────────────────────

def load_audio(path):
    """Load and normalize audio to 22050 Hz mono."""
    y, sr = librosa.load(path, sr=SR, mono=True)
    y = y / (np.max(np.abs(y)) + 1e-8)
    return y

def midi_to_wav(midi_path, wav_path, sr=SR):
    """Synthesize WAV from MIDI using pretty_midi with sine waves."""
    midi = pretty_midi.PrettyMIDI(midi_path)
    audio = np.zeros(int(1 * sr))
    for inst in midi.instruments:
        for note in inst.notes:
            start = int(note.start * sr)
            end = int(note.end * sr)
            if end <= start:
                continue
            if end > len(audio):
                audio = np.pad(audio, (0, end - len(audio)))
            dur = end - start
            dur_sec = dur / sr
            freq = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
            t = np.arange(dur) / sr
            env = np.minimum(1.0, 3.0 * t / dur_sec) * np.exp(-3.0 * t / dur_sec)
            sine = env * np.sin(2 * np.pi * freq * t)
            audio[start:end] += sine * (note.velocity / 127.0)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(wav_path, audio, sr)

def notes_to_midi(notes, midi_path, tempo=120.0):
    """Convert note list [(onset, pitch, duration, velocity), ...] to MIDI."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=0)
    for onset, pitch, dur, vel in notes:
        if dur <= 0:
            dur = 0.01
        inst.notes.append(pretty_midi.Note(
            velocity=min(127, int(vel)),
            pitch=int(pitch),
            start=float(onset),
            end=float(onset + dur)
        ))
    midi.instruments.append(inst)
    midi.write(midi_path)

# 模型文件夹名映射（每个模型产物在其自己的文件夹内，再分 midi/json/wav 子文件夹）
MODEL_FOLDER_MAP = {
    "basicpitch": "Basic Pitch",
    "yourmt3": "yourmt3",
}

def _song_id(audio_path):
    """从 structured_data/{song}/audio.wav 提取歌曲 ID，否则用文件名 stem。"""
    d = os.path.basename(os.path.dirname(audio_path))
    if d and d != "structured_data":
        return d
    return os.path.splitext(os.path.basename(audio_path))[0]

def make_output_dirs(model_name):
    """Create output directories for a model (per-model folder)."""
    dirs = {}
    folder = MODEL_FOLDER_MAP.get(model_name, model_name)
    for k in ["midi", "json", "wav"]:
        p = os.path.join(OUTPUT_ROOT, folder, f"{model_name}_{k}")
        os.makedirs(p, exist_ok=True)
        dirs[k] = p
    return dirs


# ═══════════════════════════════════════════════════════════════════════════
# MODEL 1: Basic Pitch (via venv38 subprocess)
# ═══════════════════════════════════════════════════════════════════════════

def run_basicpitch(audio_path):
    """Run Basic Pitch using venv38 subprocess."""
    dirs = make_output_dirs("basicpitch")
    model_name = "BasicPitch"
    song = _song_id(audio_path)
    t0 = time.time()

    try:
        sys.stdout.flush(); sys.stderr.flush()
        print(f"  [{model_name}] Starting subprocess...", flush=True)
        bp_root = os.path.join(OUTPUT_ROOT, "Basic Pitch")  # 模型文件夹
        result = subprocess.run(
            [VENV38_PYTHON, BP_RUNNER, audio_path, bp_root, song],
            stdout=sys.stdout, stderr=sys.stderr,
            timeout=600
        )
        elapsed = time.time() - t0

        midi_file = os.path.join(dirs["midi"], f"{song}_basicpitch.mid")
        json_file = os.path.join(dirs["json"], f"{song}_basicpitch.json")
        wav_file = os.path.join(dirs["wav"], f"{song}_basicpitch.wav")

        # Count notes from JSON if exists
        note_count = 0
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                j = json.load(f)
                note_count = j.get("note_count", len(j.get("notes", [])))

        info = {
            "model": model_name, "status": "ok", "elapsed_sec": round(elapsed, 1),
            "note_count": note_count,
            "midi": midi_file if os.path.exists(midi_file) else None,
            "json": json_file if os.path.exists(json_file) else None,
            "wav": wav_file if os.path.exists(wav_file) else None,
            "dirs": dirs,
        }
    except Exception as e:
        info = {"model": model_name, "status": "error", "error": str(e), "elapsed_sec": round(time.time() - t0, 1), "dirs": dirs}

    return info


# ═══════════════════════════════════════════════════════════════════════════
# MODEL 2: YourMT3+ 风格 — NMF 多乐器转录
#   用 NMF 分解频谱 → 聚类到乐器组 → 每乐器转录
# ═══════════════════════════════════════════════════════════════════════════

def run_yourmt3_style(audio_path):
    """
    YourMT3+-style multi-instrument transcription using NMF + librosa.
    Instruments: bass, mid, treble (NMF components → instrument groups)
    """
    dirs = make_output_dirs("yourmt3")
    model_name = "YourMT3-BandSplit"
    song = _song_id(audio_path)
    t0 = time.time()

    try:
        y = load_audio(audio_path)
        duration = len(y) / SR
        print(f"  [{model_name}] Audio: {duration:.1f}s, SR={SR}", flush=True)

        all_notes = []

        # ── Frequency-band splitting (O(n), much faster than NMF) ───
        # Bass: < 300 Hz → YIN melody
        # Mid: 300-2000 Hz → YIN + CQT peaks
        # Treble: > 2000 Hz → CQT peaks + onset

        # Reusable STFT for all bands
        print(f"  [{model_name}] Computing STFT ...", flush=True)
        t_stft = time.time()
        S_full = librosa.stft(y, n_fft=2048, hop_length=HOP_LENGTH)
        D = np.abs(S_full)
        F = librosa.fft_frequencies(sr=SR, n_fft=2048)
        print(f"  [{model_name}] STFT done ({time.time()-t_stft:.1f}s)", flush=True)

        # ── Band 1: Bass (< 300 Hz) ──────────────────────────────────
        print(f"  [{model_name}] [1/3] Bass band (< 300 Hz) ...", flush=True)
        t1 = time.time()
        b_bass = F < 300
        D_bass = D[b_bass, :]
        # istft bass-only signal
        S_bass = np.zeros_like(S_full, dtype=complex)
        S_bass[b_bass, :] = S_full[b_bass, :]
        y_bass = librosa.istft(S_bass, hop_length=HOP_LENGTH, length=len(y))
        # HPS-based bass extraction
        f0_bass = librosa.yin(y_bass, fmin=40, fmax=300, sr=SR,
                               hop_length=HOP_LENGTH, frame_length=4096)
        bass_notes = _f0_to_notes(f0_bass, f0_bass > 20, SR, HOP_LENGTH)
        bass_notes = [(o, p, d, 80) for o, p, d, v in bass_notes]
        all_notes.extend(bass_notes)
        print(f"  [{model_name}] Bass done: {len(bass_notes)} notes ({time.time()-t1:.1f}s)", flush=True)

        # ── Band 2: Mid (300-2000 Hz) ────────────────────────────────
        print(f"  [{model_name}] [2/3] Mid band (300-2000 Hz) ...", flush=True)
        t2 = time.time()
        b_mid = (F >= 300) & (F < 2000)
        S_mid = np.zeros_like(S_full, dtype=complex)
        S_mid[b_mid, :] = S_full[b_mid, :]
        y_mid = librosa.istft(S_mid, hop_length=HOP_LENGTH, length=len(y))
        mid_notes = _transcribe_instrument_group_fast(
            y_mid, SR, HOP_LENGTH, fmin=300, fmax=2000, program=1, group_label="Mid"
        )
        all_notes.extend(mid_notes)
        print(f"  [{model_name}] Mid done: {len(mid_notes)} notes ({time.time()-t2:.1f}s)", flush=True)

        # ── Band 3: Treble (> 2000 Hz) ───────────────────────────────
        print(f"  [{model_name}] [3/3] Treble band (> 2000 Hz) ...", flush=True)
        t3 = time.time()
        b_treble = F >= 2000
        S_treble = np.zeros_like(S_full, dtype=complex)
        S_treble[b_treble, :] = S_full[b_treble, :]
        y_treble = librosa.istft(S_treble, hop_length=HOP_LENGTH, length=len(y))
        treble_notes = _transcribe_instrument_group_fast(
            y_treble, SR, HOP_LENGTH, fmin=2000, fmax=8000, program=80, group_label="Treble"
        )
        all_notes.extend(treble_notes)
        print(f"  [{model_name}] Treble done: {len(treble_notes)} notes ({time.time()-t3:.1f}s)", flush=True)

        # Sort by onset
        print(f"  [{model_name}] Sorting notes ...", flush=True)
        all_notes = sorted(set(all_notes), key=lambda x: (x[0], x[1]))
        note_count = len(all_notes)
        print(f"  [{model_name}] Total notes: {note_count}", flush=True)

        # ── Estimate tempo ───────────────────────────────────────────
        print(f"  [{model_name}] Beat tracking ...", flush=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP_LENGTH)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0])

        # ── Save MIDI ────────────────────────────────────────────────
        print(f"  [{model_name}] Saving MIDI ...", flush=True)
        midi_path = os.path.join(dirs["midi"], f"{song}_yourmt3.mid")
        _save_multitrack_midi(all_notes, midi_path, tempo)

        # ── Save JSON ────────────────────────────────────────────────
        print(f"  [{model_name}] Saving JSON ...", flush=True)
        json_path = os.path.join(dirs["json"], f"{song}_yourmt3.json")
        json_data = {
            "model": model_name,
            "method": "frequency-band-splitting",
            "tempo": float(tempo),
            "duration": float(duration),
            "note_count": note_count,
            "bands": {
                "bass": len(bass_notes),
                "mid": len(mid_notes),
                "treble": len(treble_notes)
            },
            "notes": [{"onset": round(n[0], 4), "pitch": int(n[1]),
                        "duration": round(n[2], 4), "velocity": int(n[3])}
                      for n in all_notes[:5000]],
        }
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        # ── Synthesize WAV ───────────────────────────────────────────
        print(f"  [{model_name}] Synthesizing WAV ...", flush=True)
        wav_path = os.path.join(dirs["wav"], f"{song}_yourmt3.wav")
        midi_to_wav(midi_path, wav_path)

        elapsed = time.time() - t0
        info = {
            "model": model_name, "status": "ok", "elapsed_sec": round(elapsed, 1),
            "note_count": note_count,
            "midi": midi_path, "json": json_path, "wav": wav_path, "dirs": dirs,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        info = {"model": model_name, "status": "error", "error": str(e),
                "elapsed_sec": round(time.time() - t0, 1), "dirs": dirs}

    return info


# ─── Internal Helpers ──────────────────────────────────────────────────────

def _f0_to_notes(f0, voiced_flag, sr, hop, min_dur=0.05, program=0, pitch_offset=0):
    """Convert f0 contour to note list."""
    if f0 is None:
        return []
    f0 = np.nan_to_num(f0, nan=0.0)
    voiced_flag = np.array(voiced_flag, dtype=bool)
    notes = []
    in_note = False
    start_frame = 0
    last_pitch = 0
    n_frames = len(f0)

    for i in range(n_frames):
        is_voiced = voiced_flag[i] and f0[i] > 20
        if is_voiced:
            pitch = int(round(69 + 12 * np.log2(f0[i] / 440.0))) + pitch_offset
            pitch = max(21, min(108, pitch))
            if not in_note:
                in_note = True
                start_frame = i
                last_pitch = pitch
            elif abs(pitch - last_pitch) > 2:  # significant pitch change
                dur = (i - start_frame) * hop / sr
                if dur >= min_dur:
                    onset = start_frame * hop / sr
                    notes.append((onset, last_pitch, dur, 100))
                start_frame = i
                last_pitch = pitch
            else:
                last_pitch = int(round(0.7 * last_pitch + 0.3 * pitch))  # smooth
        elif in_note:
            dur = (i - start_frame) * hop / sr
            if dur >= min_dur:
                onset = start_frame * hop / sr
                notes.append((onset, last_pitch, dur, 100))
            in_note = False

    if in_note:
        dur = (n_frames - start_frame) * hop / sr
        if dur >= min_dur:
            onset = start_frame * hop / sr
            notes.append((onset, last_pitch, dur, 100))

    return notes


def _chroma_to_harmony_notes(y, sr, hop, n_notes_per_frame=3):
    """Extract harmony notes from chromagram using energy peaks."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop,
                                          bins_per_octave=24)
    chroma = chroma / (np.max(chroma) + 1e-8)
    notes = []
    n_frames = chroma.shape[1]

    for i in range(n_frames):
        frame = chroma[:, i]
        indices = np.argsort(frame)[-n_notes_per_frame:]  # top N chroma bins
        for idx in indices:
            if frame[idx] > 0.1:
                # Each chroma bin can map to multiple octaves
                root_pitch = 48 + idx % 12  # start from C3
                for octave in [0, 1]:  # two octaves
                    pitch = root_pitch + octave * 12
                    if 36 <= pitch <= 84:
                        onset = i * hop / sr
                        dur = hop / sr * 2  # two frames
                        notes.append((onset, pitch, dur, 70))

    return notes


def _percussion_detect(y, sr, hop):
    """Detect percussion hits from onsets and spectral flux."""
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onset_frames = librosa.onset.onset_detect(onset_envelope=oenv,
                                               sr=sr, hop_length=hop,
                                               backtrack=True)
    notes = []
    drum_pitches = [36, 38, 42, 46, 49]  # Kick, Snare, HH-closed, HH-open, Crash
    for i, of in enumerate(onset_frames):
        onset = of * hop / sr
        pitch = drum_pitches[i % len(drum_pitches)]
        dur = 0.1
        vel = min(127, int(60 + 40 * oenv[of] / (np.max(oenv) + 1e-8)))
        notes.append((onset, pitch, dur, vel))
    return notes


def _chunked_f0_extraction(y, sr, hop, chunk_sec):
    """Extract F0 using YIN in overlapping chunks (much faster than pYIN)."""
    chunk_samples = int(chunk_sec * sr)
    overlap_samples = int(1.0 * sr)  # 1s overlap
    total = len(y)
    notes = []
    chunks_total = max(1, int(np.ceil(total / max(1, (chunk_samples - overlap_samples)))))
    chunk_idx = 0

    for start in range(0, total, chunk_samples - overlap_samples):
        chunk_idx += 1
        end = min(start + chunk_samples, total)
        yc = y[start:end]
        time_offset = start / sr
        if len(yc) < sr * 0.5:  # skip very short chunks
            continue
        if chunk_idx % 10 == 0:
            print(f"    chunk {chunk_idx}/{chunks_total} ({time_offset:.0f}s)")

        try:
            f0_c = librosa.yin(yc, fmin=65, fmax=2000, sr=sr,
                               hop_length=hop, frame_length=2048)
        except Exception:
            continue

        chunk_notes = _f0_to_notes(f0_c, f0_c > 20, sr, hop, min_dur=0.08)
        for onset, pitch, dur, vel in chunk_notes:
            notes.append((onset + time_offset, pitch, dur, vel))

    return notes


def _chunked_harmony(y, sr, hop, chunk_sec, n_notes_per_frame=3):
    """Extract harmony from STFT chroma in chunks with progress.
    Downsampled: only process every N-th frame to avoid explosion."""
    chunk_samples = int(chunk_sec * sr)
    overlap_samples = int(2.0 * sr)
    total = len(y)
    notes = []
    step = max(1, chunk_samples - overlap_samples)
    chunks_total = max(1, int(np.ceil(total / step)))
    frame_skip = 4  # only keep every 4th frame

    print(f"    harmony chunks: {chunks_total}, frame_skip={frame_skip}", flush=True)

    for ci in range(chunks_total):
        start = ci * step
        end = min(start + chunk_samples, total)
        yc = y[start:end]
        time_offset = start / sr
        if len(yc) < sr * 2:
            continue
        if ci % 5 == 0 or ci == chunks_total - 1:
            print(f"    harmony chunk {ci+1}/{chunks_total} ({time_offset:.0f}s)", flush=True)

        try:
            chroma = librosa.feature.chroma_stft(y=yc, sr=sr, hop_length=hop, n_chroma=12)
        except Exception:
            continue

        n_frames = chroma.shape[1]
        for i in range(0, n_frames, frame_skip):
            frame = chroma[:, i]
            indices = np.argsort(frame)[-n_notes_per_frame:]
            for idx in indices:
                if frame[idx] > 0.2:
                    root_pitch = 48 + idx
                    pitch = root_pitch + 12  # single octave above
                    if 36 <= pitch <= 84:
                        onset = time_offset + i * hop / sr
                        dur = hop / sr * frame_skip * 2
                        notes.append((onset, pitch, dur, 65))

    return notes


def _assign_nmf_groups(H, F, n_components):
    """Assign NMF components to Bass / Mid / Treble groups based on spectral centroid."""
    groups = {0: [], 1: [], 2: []}
    for c in range(n_components):
        # Spectral centroid of this component
        centroid = np.sum(F * H[c]) / (np.sum(H[c]) + 1e-8)
        if centroid < 250:
            groups[0].append(c)   # Bass
        elif centroid < 2000:
            groups[1].append(c)   # Mid
        else:
            groups[2].append(c)   # Treble

    # Ensure each group has at least 1 component
    all_assigned = groups[0] + groups[1] + groups[2]
    unassigned = [c for c in range(n_components) if c not in all_assigned]
    for c in unassigned:
        groups[1].append(c)

    return groups


def _transcribe_instrument_group_fast(y_group, sr, hop, fmin=40, fmax=4000,
                                       program=0, group_label=""):
    """Transcribe an isolated instrument group using fast YIN (not pYIN)."""
    notes = []
    if np.max(np.abs(y_group)) < 1e-6:
        return notes

    # Use YIN for monophonic pitch (much faster than pYIN)
    f0 = librosa.yin(y_group, fmin=fmin, fmax=fmax, sr=sr,
                     hop_length=hop, frame_length=2048)
    if f0 is not None:
        voiced = f0 > 20
        melody_notes = _f0_to_notes(f0, voiced, sr, hop,
                                     min_dur=0.06, pitch_offset=0)
        notes.extend(melody_notes)

    # Additionally use CQT peaks for polyphonic content
    max_bins = int(12 * np.log2(sr / 2 / max(1, fmin)))
    n_bins = min(84, max_bins)
    if n_bins <= 0:
        return notes
    cqt = np.abs(librosa.cqt(y_group, sr=sr, hop_length=hop,
                              fmin=fmin, n_bins=n_bins, bins_per_octave=12))
    if cqt.shape[1] > 0:
        cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
        n_frames = cqt_db.shape[1]
        for i in range(n_frames):
            frame = cqt_db[:, i]
            peaks = []
            for j in range(1, len(frame) - 1):
                if frame[j] > frame[j-1] and frame[j] > frame[j+1] and frame[j] > -30:
                    peaks.append((j, frame[j]))
            peaks.sort(key=lambda x: -x[1])
            for j, strength in peaks[:3]:
                onset = i * hop / sr
                dur = hop / sr * 2
                pitch = j + int(np.log2(fmin / 440.0) * 12 + 69)
                if 21 <= pitch <= 108:
                    vel = min(127, int(40 + 40 * (strength + 30) / 30))
                    notes.append((onset, pitch, dur, max(30, vel)))

    return notes


def _transcribe_instrument_group(y_group, sr, hop, fmin=40, fmax=4000,
                                  program=0, group_label=""):
    """Transcribe an isolated instrument group."""
    notes = []
    if np.max(np.abs(y_group)) < 1e-6:
        return notes

    # pYIN for monophonic pitch
    f0, voiced_flag, _ = librosa.pyin(
        y_group, fmin=fmin, fmax=fmax,
        sr=sr, hop_length=hop
    )
    if f0 is not None:
        melody_notes = _f0_to_notes(f0, voiced_flag, sr, hop,
                                     min_dur=0.06,
                                     pitch_offset=0)
        notes.extend(melody_notes)

    # Additionally use CQT peaks for polyphonic content
    max_bins = int(12 * np.log2(sr / 2 / max(1, fmin)))
    n_bins = min(84, max_bins)
    if n_bins <= 0:
        return notes
    cqt = np.abs(librosa.cqt(y_group, sr=sr, hop_length=hop,
                              fmin=fmin, n_bins=n_bins, bins_per_octave=12))
    if cqt.shape[1] > 0:
        cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
        n_frames = cqt_db.shape[1]
        for i in range(n_frames):
            frame = cqt_db[:, i]
            peaks = []
            for j in range(1, len(frame) - 1):
                if frame[j] > frame[j-1] and frame[j] > frame[j+1] and frame[j] > -30:
                    peaks.append((j, frame[j]))
            peaks.sort(key=lambda x: -x[1])
            for j, strength in peaks[:3]:
                onset = i * hop / sr
                dur = hop / sr * 2
                pitch = j + int(np.log2(fmin / 440.0) * 12 + 69)
                if 21 <= pitch <= 108:
                    vel = min(127, int(40 + 40 * (strength + 30) / 30))
                    notes.append((onset, pitch, dur, max(30, vel)))

    return notes


def _save_multitrack_midi(all_notes, midi_path, tempo=120.0):
    """Save notes to multi-track MIDI with instrument grouping by program."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    # Group notes by program number
    from collections import defaultdict
    groups = defaultdict(list)
    for onset, pitch, dur, vel in all_notes:
        if dur <= 0:
            dur = 0.01
        groups[0].append((onset, pitch, dur, vel))  # all into one track for now

    for prog, notes_list in groups.items():
        inst = pretty_midi.Instrument(program=prog, is_drum=(prog == 0 and False))
        for onset, pitch, dur, vel in notes_list:
            inst.notes.append(pretty_midi.Note(
                velocity=min(127, int(vel)),
                pitch=int(pitch),
                start=float(onset),
                end=float(onset + dur)
            ))
        midi.instruments.append(inst)
    midi.write(midi_path)


# ═══════════════════════════════════════════════════════════════════════════
# Main Runner
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Model MIDI Transcription for Nanyin Audio")
    parser.add_argument("--audio", type=str,
                        default=os.path.join(
                            PROJECT_ROOT,
                            r"南音数据集\nanyin_final\structured_data"
                            r"\dapu01qishouban\audio.wav"),
                        help="Path to input audio file")
    parser.add_argument("--models", type=str, nargs="+",
                        default=["basicpitch", "yourmt3"],
                        help="Models to run")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models")
    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        print("  basicpitch  - Basic Pitch via venv38 (polyphonic, general)")
        print("  yourmt3     - YourMT3+-style: NMF multi-instrument decomposition")
        return

    audio_path = os.path.abspath(args.audio)
    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)

    print("=" * 70)
    print(f"Multi-Model Transcription Pipeline")
    print(f"Audio: {audio_path}")
    print(f"Models: {args.models}")
    print("=" * 70)

    results = []
    start_all = time.time()

    for model_name in args.models:
        model_name = model_name.lower().strip()
        print(f"\n{'─'*60}")
        print(f"Running: {model_name}")
        print(f"{'─'*60}")

        if model_name == "basicpitch":
            info = run_basicpitch(audio_path)
        elif model_name == "yourmt3":
            info = run_yourmt3_style(audio_path)
        else:
            print(f"  Unknown model: {model_name}, skipping.")
            continue

        results.append(info)
        status = info.get("status", "?")
        if status == "ok":
            print(f"  [{model_name}] ✅ {info.get('note_count', 0)} notes, "
                  f"{info.get('elapsed_sec', 0):.1f}s")
            print(f"  MIDI: {info.get('midi', '-')}")
            print(f"  JSON: {info.get('json', '-')}")
            print(f"  WAV:  {info.get('wav', '-')}")
        else:
            print(f"  [{model_name}] ❌ {info.get('error', 'unknown error')}")

    total_elapsed = time.time() - start_all

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"SUMMARY ({total_elapsed:.1f}s total)")
    print(f"{'='*70}")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "audio": audio_path,
        "total_elapsed_sec": round(total_elapsed, 1),
        "models": [],
    }
    for r in results:
        entry = {
            "model": r["model"],
            "status": r.get("status", "?"),
            "elapsed_sec": r.get("elapsed_sec", 0),
            "note_count": r.get("note_count", 0),
            "midi": r.get("midi"),
            "json": r.get("json"),
            "wav": r.get("wav"),
        }
        summary["models"].append(entry)
        ok = "✅" if entry["status"] == "ok" else "❌"
        print(f"  {ok} {entry['model']:20s} {entry['note_count']:6d} notes  "
              f"{entry['elapsed_sec']:6.1f}s")

    summary_path = os.path.join(OUTPUT_ROOT, "transcription_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
