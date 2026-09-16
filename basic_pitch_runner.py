"""basic_pitch_runner.py — 在 Python 3.8 venv 中运行 Basic Pitch 转录。
由 model_midi.py 通过 subprocess 调用。
用法: venv38/Scripts/python.exe basic_pitch_runner.py <input_wav> <output_root> [name_prefix]
输出: <output_root>/basicpitch_midi|json|wav/<name_prefix>_basicpitch.{mid,json,wav}
"""
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main():
    if len(sys.argv) < 3:
        print("ERROR: usage: python basic_pitch_runner.py <wav_path> <output_root> [name_prefix]", file=sys.stderr)
        sys.exit(1)

    audio_path = Path(sys.argv[1]).resolve()
    output_root = Path(sys.argv[2]).resolve()
    # 歌曲名前缀（可选），默认用输入 wav 的 stem
    name_prefix = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3].strip() else audio_path.stem
    midi_dir = output_root / "basicpitch_midi"
    json_dir = output_root / "basicpitch_json"
    wav_dir = output_root / "basicpitch_wav"

    for d in [midi_dir, json_dir, wav_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[basic_pitch_runner] audio: {audio_path}")
    print(f"[basic_pitch_runner] output: {output_root}")

    try:
        import numpy as np
        import librosa
        import soundfile as sf
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH
    except ImportError as e:
        print(f"ERROR: import failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    t0 = time.time()

    # 加载音频
    print("[basic_pitch_runner] loading audio ...")
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    duration = len(y) / sr
    print(f"[basic_pitch_runner] audio: {duration:.1f}s, sr={sr}")

    if sr != 22050:
        print(f"[basic_pitch_runner] resampling {sr} -> 22050 ...")
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=22050)
        sr = 22050

    # 转录
    print("[basic_pitch_runner] transcribing (polyphonic) ...")
    try:
        model_output, midi_data, note_events = predict(
            str(audio_path),
            model_or_model_path=ICASSP_2022_MODEL_PATH,
            onset_threshold=0.4,
            frame_threshold=0.2,
            minimum_note_length=58,
            minimum_frequency=80,
            maximum_frequency=2000,
            melodia_trick=False,
        )
    except TypeError:
        print("[basic_pitch_runner] fallback to default params ...")
        model_output, midi_data, note_events = predict(str(audio_path))

    elapsed = time.time() - t0
    note_count = len(note_events) if note_events is not None else 0
    print(f"[basic_pitch_runner] detected {note_count} notes ({elapsed:.0f}s)")

    # 保存 MIDI
    midi_path = midi_dir / f"{name_prefix}_basicpitch.mid"
    midi_data.write(str(midi_path))
    print(f"[basic_pitch_runner] saved MIDI: {midi_path}")

    # 保存 JSON
    notes_list = []
    if note_events is not None:
        for n in note_events:
            # 兼容 tuple 和 NamedTuple 两种返回格式
            if isinstance(n, (list, tuple)):
                notes_list.append({
                    "start_time": float(n[0]) if len(n) > 0 else 0,
                    "end_time": float(n[1]) if len(n) > 1 else 0,
                    "pitch": int(n[2]) if len(n) > 2 else 0,
                    "velocity": int(n[3]) if len(n) > 3 else 0,
                    "midi_note": int(n[2]) if len(n) > 2 else 0,
                    "duration_s": round(float(n[1] - n[0])
                                        if len(n) > 1 else 0, 3),
                })
            else:
                notes_list.append({
                    "start_time": float(getattr(n, "start_time", 0)),
                    "end_time": float(getattr(n, "end_time", 0)),
                    "pitch": int(getattr(n, "pitch", 0)),
                    "velocity": int(getattr(n, "velocity", 0)),
                    "midi_note": int(getattr(n, "pitch", 0)),
                    "duration_s": round(float(getattr(n, "end_time", 0)
                                        - getattr(n, "start_time", 0)), 3),
                })

    json_path = json_dir / f"{name_prefix}_basicpitch.json"
    json_out = {
        "audio": str(audio_path),
        "sample_rate": sr,
        "duration_s": round(duration, 2),
        "total_notes": len(notes_list),
        "model": "basic-pitch ICASSP 2022",
        "parameters": {
            "onset_threshold": 0.4,
            "frame_threshold": 0.2,
            "minimum_note_length_ms": 58,
            "frequency_range_hz": [80, 2000],
        },
        "timestamp": datetime.now().isoformat(),
        "notes": notes_list,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"[basic_pitch_runner] saved JSON: {json_path} ({len(notes_list)} notes)")

    # 合成 WAV (正弦波)
    print("[basic_pitch_runner] synthesizing WAV ...")
    wav_out = wav_dir / f"{name_prefix}_basicpitch.wav"
    _midi_to_wav_sine(midi_data, sr=22050, output_path=wav_out)
    wav_len = os.path.getsize(wav_out) / 1024 / 1024
    print(f"[basic_pitch_runner] saved WAV: {wav_out} ({wav_len:.1f} MB)")

    print("[basic_pitch_runner] DONE")
    print(f"MIDI={midi_path}")


def _midi_to_wav_sine(midi_data, sr=22050, output_path=None):
    """MIDI → WAV (正弦波合成，仅供试听旋律)"""
    import numpy as np

    # pretty_midi.PrettyMIDI 直接提供 notes
    notes = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes.append((note.start, note.end, note.pitch, note.velocity))

    if not notes:
        print("[basic_pitch_runner] WARNING: no notes to synthesize")
        return

    total_seconds = midi_data.get_end_time() + 1.0
    n_samples = int(total_seconds * sr)
    audio = np.zeros(n_samples, dtype=np.float32)

    for start_s, end_s, pitch, vel in notes:
        end_s = min(end_s, total_seconds)
        if end_s <= start_s:
            continue
        freq = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        ns = int((end_s - start_s) * sr)
        t = np.arange(ns, dtype=np.float32) / sr

        attack, decay, sustain_level, release = 0.02, 0.05, 0.7, 0.1
        envelope = np.ones(ns, dtype=np.float32)
        a_n, d_n, r_n = int(attack * sr), int(decay * sr), int(release * sr)
        if a_n > 0 and a_n < ns:
            envelope[:a_n] = np.linspace(0, 1, a_n)
        if d_n > 0:
            a_plus_d = min(a_n + d_n, ns)
            envelope[a_n:a_plus_d] = np.linspace(1, sustain_level, a_plus_d - a_n)
        envelope[a_n + d_n:] = sustain_level
        if r_n < ns:
            envelope[ns - r_n:] = np.linspace(sustain_level, 0, min(r_n, ns - (ns - r_n)) if ns > r_n else ns)

        amp = (vel / 127.0) * 0.3
        wave = np.sin(2 * np.pi * freq * t) * envelope * amp
        idx_start = int(start_s * sr)
        idx_end = min(idx_start + ns, n_samples)
        wave = wave[:idx_end - idx_start]
        audio[idx_start:idx_end] += wave

    peak = np.abs(audio).max()
    if peak > 0:
        audio /= peak * 1.1

    import soundfile as sf
    sf.write(str(output_path), audio, sr)
    mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"[basic_pitch_runner] synthesized WAV: {total_seconds:.1f}s, {mb:.1f} MB")


if __name__ == "__main__":
    main()
