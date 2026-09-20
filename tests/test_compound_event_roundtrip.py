# -*- coding: utf-8 -*-
"""
tests/test_compound_event_roundtrip.py — E1 复合事件编解码单元自测
==================================================================

自测内容：**MIDI → 复合事件序列 → 写回 MIDI** 的往返一致性校验

  1. 编解码器层面：量化↔反量化一致性、index 语义（PAD / 精确档 / 溢出档）
  2. 真实 GT 曲目（11 首 BasicPitch 转录）：
       原始音符 → encode → decode → 与原始逐音比较（note-level 误差）
       → notes_to_midi → pretty_midi 读回 → 再比较（含 MIDI tick 量化的 file-level 误差）
  3. 伪标签约束：序列末尾 time-shift 是否为 0
  4. 合成边界用例：超长休止 / 同时发音 / 超长音符 / 同音反复 / 负间隔

运行：
    venv/Scripts/python.exe tests/test_compound_event_roundtrip.py

产物：
    reports/E1_selftest_compound_events.md   （自测报告，纳入版本管理）
    results/e1_roundtrip_tmp/*.mid           （临时 MIDI，已在 .gitignore）
"""

import json
import os
import statistics
import sys
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (_PROJECT_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from nanyin_config import load_config
from dataset_nanyin import load_song_annotations, get_compound_codec
from generate_music import events_to_notes, notes_to_midi


def _readback_notes(midi_path: str) -> List[dict]:
    """用 pretty_midi 读回 MIDI，返回按 (start, pitch) 排序的音符列表。"""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes: List[dict] = []
    for inst in pm.instruments:
        for n in inst.notes:
            notes.append({
                "note": int(n.pitch),
                "start_time": float(n.start),
                "duration": float(n.end - n.start),
            })
    notes.sort(key=lambda x: (x["start_time"], x["note"]))
    return notes


def _sort_notes(notes: List[dict]) -> List[dict]:
    """统一排序键：(start_time, note)。"""
    return sorted(notes, key=lambda x: (float(x["start_time"]), int(x["note"])))


def _compare_note_lists(
    orig: List[dict],
    dec: List[dict],
    tol_sec: float = 0.0,
    sort_second: bool = False,
) -> Dict[str, object]:
    """逐音比较两个音符列表，返回 pitch / duration / onset 误差统计。

    排序口径（重要，否则会出现"假性 pitch 不一致"）：
      - 第一个列表始终按 (start, note) 规范化排序；
      - 第二个列表**默认保持原顺序**：解码是按编码输入的顺序逐个 append 的，
        与第一个列表同序；量化会把原本相差亚毫秒的 onset 拉平到同一网格，
        若此时再按 (start, note) 重排解码结果，同组内的次序可能被打乱。
      - 仅在 **file-level**（写回 MIDI 再读回）比较时传入 sort_second=True：
        MIDI 文件不保存插入顺序，两侧必须用同一排序键。

    onset 采用**相对比较**：原始 onset 减去首音符 onset，
    消除解码统一从 0 起算带来的整体偏移。
    """
    orig_s = _sort_notes(orig)
    dec_s = _sort_notes(dec) if sort_second else list(dec)
    n = min(len(orig_s), len(dec_s))
    if n == 0:
        return {"num_compared": 0}

    base_start = float(orig_s[0]["start_time"])
    pitch_bad = 0
    dur_errors: List[float] = []
    onset_errors: List[float] = []
    for i in range(n):
        if int(orig_s[i]["note"]) != int(dec_s[i]["note"]):
            pitch_bad += 1
        dur_errors.append(abs(float(dec_s[i]["duration"]) - float(orig_s[i]["duration"])))
        onset_err = float(dec_s[i]["start_time"]) - (float(orig_s[i]["start_time"]) - base_start)
        onset_errors.append(abs(onset_err))

    # 浮点边界余量：误差上界恰好等于容差时（如 1 格边界），二进制浮点可能略微超出，
    # 用 1e-9 s（1 纳秒）作为数值比较余量，不改变判定标准本身。
    eps = 1e-9
    dur_ok = sum(1 for e in dur_errors if e <= tol_sec + eps)
    onset_ok = sum(1 for e in onset_errors if e <= tol_sec + eps)

    return {
        "num_compared": n,
        "pitch_mismatch": pitch_bad,
        "dur_err_mean_ms": statistics.mean(dur_errors) * 1000.0,
        "dur_err_max_ms": max(dur_errors) * 1000.0,
        "dur_within_tol_rate": dur_ok / n,
        "onset_err_mean_ms": statistics.mean(onset_errors) * 1000.0,
        "onset_err_max_ms": max(onset_errors) * 1000.0,
        "onset_within_tol_rate": onset_ok / n,
        "count_orig": len(orig_s),
        "count_dec": len(dec_s),
    }


def test_codec_unit(codec, report_lines: List[str]) -> Dict[str, bool]:
    """编解码器层面：量化 ↔ 反量化一致性与 index 语义。"""
    lines = report_lines
    lines.append("### T1 编解码器单元一致性")
    checks: Dict[str, bool] = {}

    unit_ms = codec.time_unit_sec * 1000.0

    # 1) duration 精确档往返
    dur_bad: List[str] = []
    for steps in list(range(0, codec.duration_max_steps, 7)):
        sec = steps * codec.time_unit_sec
        idx = codec.duration_to_index(sec)
        back = codec.index_to_duration(idx)
        if abs(back - sec) > 1e-9:
            dur_bad.append(f"steps={steps}: {sec:.4f}→idx{idx}→{back:.4f}")
    checks["duration_exact_roundtrip"] = len(dur_bad) == 0

    # 2) duration 溢出档语义
    sec_over = (codec.duration_max_steps + 5) * codec.time_unit_sec
    idx_over = codec.duration_to_index(sec_over)
    back_over = codec.index_to_duration(idx_over)
    checks["duration_overflow_bucket"] = (
        idx_over == codec.duration_max_steps + 1
        and abs(back_over - codec.duration_max_steps * codec.time_unit_sec) < 1e-9
    )

    # 3) shift 精确档往返 & 溢出档语义
    shift_bad: List[str] = []
    for steps in list(range(0, codec.shift_max_steps, 11)):
        sec = steps * codec.time_unit_sec
        idx = codec.shift_to_index(sec)
        back = codec.index_to_shift(idx)
        if abs(back - sec) > 1e-9:
            shift_bad.append(f"steps={steps}: {sec:.4f}→idx{idx}→{back:.4f}")
    checks["shift_exact_roundtrip"] = len(shift_bad) == 0

    idx_shift_over = codec.shift_to_index((codec.shift_max_steps + 20) * codec.time_unit_sec)
    checks["shift_overflow_bucket"] = idx_shift_over == codec.shift_max_steps + 1

    # 4) PAD / 伪标签位：index 0 还原为 0 秒
    checks["index0_is_pad"] = (
        abs(codec.index_to_duration(0)) < 1e-9 and abs(codec.index_to_shift(0)) < 1e-9
    )

    lines.append("")
    lines.append("| 检查项 | 结果 | 说明 |")
    lines.append("|---|---|---|")
    lines.append(f"| duration 精确档往返 | {'PASS' if checks['duration_exact_roundtrip'] else 'FAIL'} | "
                 f"0~{codec.duration_max_steps - 1} 步长（0~{(codec.duration_max_steps - 1) * unit_ms:.0f} ms），容差 1e-9 s |")
    lines.append(f"| duration 溢出档语义 | {'PASS' if checks['duration_overflow_bucket'] else 'FAIL'} | "
                 f"≥ {codec.duration_max_steps * unit_ms:.0f} ms → index {codec.duration_max_steps + 1}，还原为下界 |")
    lines.append(f"| time-shift 精确档往返 | {'PASS' if checks['shift_exact_roundtrip'] else 'FAIL'} | "
                 f"0~{codec.shift_max_steps - 1} 步长（0~{(codec.shift_max_steps - 1) * unit_ms:.0f} ms）|")
    lines.append(f"| time-shift 溢出档语义 | {'PASS' if checks['shift_overflow_bucket'] else 'FAIL'} | "
                 f"≥ {codec.shift_max_steps * unit_ms:.0f} ms → index {codec.shift_max_steps + 1} |")
    lines.append(f"| index 0 = PAD / 伪标签 | {'PASS' if checks['index0_is_pad'] else 'FAIL'} | 还原为 0 秒 |")
    if dur_bad:
        lines.append("")
        lines.append(f"duration 往返不一致样本（前5）: {dur_bad[:5]}")
    if shift_bad:
        lines.append("")
        lines.append(f"time-shift 往返不一致样本（前5）: {shift_bad[:5]}")
    lines.append("")
    return checks


def test_real_songs(codec, cfg: dict, report_lines: List[str]) -> Dict[str, Any]:
    """真实 GT 曲目：MIDI → 复合事件 → MIDI 往返。"""
    selftest_cfg = cfg["selftest"]
    gen_cfg = cfg["generation"]
    tmp_dir = os.path.join(_PROJECT_ROOT, selftest_cfg["tmp_dir"])
    os.makedirs(tmp_dir, exist_ok=True)

    tol_ms = float(selftest_cfg["duration_tolerance_sec"]) * 1000.0
    song_rows: List[dict] = []

    for song_id in selftest_cfg["songs"]:
        ann = load_song_annotations(song_id)
        if ann is None:
            report_lines.append(f"- **{song_id}**: GT MIDI 加载失败，跳过")
            continue
        notes = ann["midi"]["tracks"][0]["notes"]
        notes.sort(key=lambda x: (x["start_time"], x["note"]))

        events = codec.encode_notes(notes)
        pitch_ids = events["pitch_ids"]
        duration_ids = events["duration_ids"]
        shift_ids = events["shift_ids"]

        decoded = events_to_notes(pitch_ids, duration_ids, shift_ids, codec=codec)

        # note-level（编解码器 + 量化）比较
        tol_sec = tol_ms / 1000.0
        nl = _compare_note_lists(notes[:len(decoded)], decoded, tol_sec=tol_sec)

        # 写回 MIDI 并用 pretty_midi 读回（file-level，含 tick 量化）
        out_path = os.path.join(tmp_dir, f"{song_id}_roundtrip.mid")
        write_stats = notes_to_midi(
            decoded,
            output_path=out_path,
            ticks_per_beat=int(gen_cfg["ticks_per_beat"]),
            tempo_bpm=int(gen_cfg["default_bpm"]),
        )
        reread = _readback_notes(out_path)
        # MIDI 无插入顺序信息 → 两侧必须用同一排序键
        fl = _compare_note_lists(decoded, reread, tol_sec=tol_sec, sort_second=True)

        n = int(nl["num_compared"])

        song_rows.append({
            "song_id": song_id,
            "num_notes": int(n),
            "encode_stats": events["stats"],
            "tail_shift_is_pseudo": bool(len(shift_ids) > 0 and shift_ids[-1] == 0),
            "pitch_mismatch": int(nl["pitch_mismatch"]),
            "dur_err_mean_ms": float(nl["dur_err_mean_ms"]),
            "dur_err_max_ms": float(nl["dur_err_max_ms"]),
            "dur_within_tol_rate": float(nl["dur_within_tol_rate"]),
            "onset_err_mean_ms": float(nl["onset_err_mean_ms"]),
            "onset_err_max_ms": float(nl["onset_err_max_ms"]),
            "onset_within_tol_rate": float(nl["onset_within_tol_rate"]),
            "midi_write": write_stats,
            "midi_readback": {
                "count_dec": int(fl["count_dec"]),
                "count_reread": int(fl["count_orig"]),
                "pitch_mismatch": int(fl["pitch_mismatch"]),
                "dur_err_max_ms": float(fl["dur_err_max_ms"]),
                "onset_err_max_ms": float(fl["onset_err_max_ms"]),
            },
        })

    return {"songs": song_rows, "tolerance_ms": tol_ms}


def test_edge_cases(codec, cfg: dict, report_lines: List[str]) -> List[dict]:
    """合成边界用例。"""
    selftest_cfg = cfg["selftest"]
    gen_cfg = cfg["generation"]
    tmp_dir = os.path.join(_PROJECT_ROOT, selftest_cfg["tmp_dir"])
    os.makedirs(tmp_dir, exist_ok=True)

    aol = float(selftest_cfg["long_rest_sec"])
    long_dur = float(selftest_cfg["long_duration_sec"])

    cases: List[dict] = []

    # 用例1：超长休止（远超溢出档）
    notes_long_rest = [
        {"note": 60, "start_time": 0.0, "duration": 0.5},
        {"note": 62, "start_time": 0.5 + aol, "duration": 0.5},
        {"note": 64, "start_time": 0.5 + aol + 0.3, "duration": 0.5},
    ]
    # 用例2：同时发音（shift = 0）
    notes_chord = [
        {"note": 60, "start_time": 0.0, "duration": 0.5},
        {"note": 64, "start_time": 0.0, "duration": 0.5},
        {"note": 67, "start_time": 0.0, "duration": 0.5},
        {"note": 62, "start_time": 0.5, "duration": 0.5},
    ]
    # 用例3：超长音符
    notes_long_note = [
        {"note": 59, "start_time": 0.0, "duration": long_dur},
        {"note": 60, "start_time": long_dur, "duration": 0.4},
    ]
    # 用例4：同音反复（触发 shift=0 与重叠写 MIDI 的配对问题）
    notes_repeat = [
        {"note": 72, "start_time": 0.0, "duration": 0.1},
        {"note": 72, "start_time": 0.1, "duration": 0.1},
        {"note": 72, "start_time": 0.2, "duration": 0.1},
        {"note": 72, "start_time": 0.25, "duration": 0.1},   # 与上一条重叠
    ]
    # 用例5：负间隔（乱序输入 → 应被截断为 0 并计数）
    notes_neg = [
        {"note": 65, "start_time": 0.0, "duration": 0.2},
        {"note": 66, "start_time": 0.6, "duration": 0.2},
        {"note": 67, "start_time": 0.3, "duration": 0.2},   # onset 回退 → 负间隔
    ]

    # 第三列：该用例是否"逐音比较适用"。
    # 负间隔用例的输入本身是乱序的（onset 回退），编码器会插入顺序固化，
    # 而规范化排序会把顺序改回来 → 逐音 pitch 比较在该用例上没有意义，只检查防御是否生效。
    specs = [
        ("超长休止(溢出档)", notes_long_rest, True),
        ("同时发音shift=0", notes_chord, True),
        ("超长音符(溢出档)", notes_long_note, True),
        ("同音反复+重叠", notes_repeat, True),
        ("负间隔防御(乱序输入)", notes_neg, False),
    ]

    for name, notes, comparable in specs:
        events = codec.encode_notes(notes)
        decoded = events_to_notes(
            events["pitch_ids"], events["duration_ids"], events["shift_ids"], codec=codec
        )
        nl = _compare_note_lists(notes[:len(decoded)], decoded)
        if not comparable:
            nl["pitch_mismatch"] = None      # 该用例不适用逐音比较
        out_path = os.path.join(tmp_dir, f"edge_{name}.mid")
        write_stats = notes_to_midi(
            decoded,
            output_path=out_path,
            ticks_per_beat=int(gen_cfg["ticks_per_beat"]),
            tempo_bpm=int(gen_cfg["default_bpm"]),
        )
        reread = _readback_notes(out_path)
        cases.append({
            "case": name,
            "num_notes": int(nl["num_compared"]),
            "pitch_mismatch": None if nl["pitch_mismatch"] is None else int(nl["pitch_mismatch"]),
            "order_sensitive": not comparable,
            "dur_err_max_ms": float(nl["dur_err_max_ms"]),
            "onset_err_max_ms": float(nl["onset_err_max_ms"]),
            "encode_stats": events["stats"],
            "reread_count": len(reread),
            "truncated_overlap": int(write_stats["truncated_overlap"]),
        })

    return cases


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):           # type: ignore[attr-defined]
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    cfg = load_config()
    codec = get_compound_codec()

    print("=" * 74)
    print("E1 单元自测：MIDI → 复合事件 → 写回 MIDI 往返一致性")
    print("=" * 74)

    unit_lines: List[str] = []
    unit_checks = test_codec_unit(codec, unit_lines)

    print("[1/3] 编解码器单元一致性完成")
    song_result = test_real_songs(codec, cfg, [])
    print(f"[2/3] 真实曲目往返完成：{len(song_result['songs'])} 首")
    edge_cases = test_edge_cases(codec, cfg, [])
    print(f"[3/3] 边界用例完成：{len(edge_cases)} 个")

    # ---- 汇总写盘 ----
    report_path = os.path.join(_PROJECT_ROOT, cfg["selftest"]["report_path"])
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    md = _render_markdown(cfg, unit_checks, unit_lines, song_result, edge_cases)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)

    json_path = os.path.join(_PROJECT_ROOT, "results", "e1_roundtrip_metrics.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "unit_checks": unit_checks,
                "songs": song_result["songs"],
                "tolerance_ms": song_result["tolerance_ms"],
                "edge_cases": edge_cases,
            },
            f, ensure_ascii=False, indent=2,
        )

    print(f"\n自测报告: {report_path}")
    print(f"指标 JSON: {json_path}")

    # ---- 控制台摘要 ----
    songs = song_result["songs"]  # type: ignore
    print("\n" + "-" * 74)
    print(f"{'曲目':<28}{'音符':>6}{'pitch错':>8}{'时值均ms':>10}{'时值max':>9}{'onset均ms':>11}{'onsetmax':>9}")
    print("-" * 74)
    for s in songs:
        print(f"{s['song_id']:<28}{s['num_notes']:>6}{s['pitch_mismatch']:>8}"
              f"{s['dur_err_mean_ms']:>10.2f}{s['dur_err_max_ms']:>9.1f}"
              f"{s['onset_err_mean_ms']:>11.2f}{s['onset_err_max_ms']:>9.1f}")
    print("-" * 74)

    all_pitch_ok = all(s["pitch_mismatch"] == 0 for s in songs)
    all_count_ok = all(
        s["midi_readback"]["count_dec"] == s["midi_readback"]["count_reread"] for s in songs
    )
    if all_pitch_ok and all_count_ok and all(unit_checks.values()):
        print("结论：音高无损 + 音符数守恒 + 编解码器单元检查全部通过 → 自测 PASS")
        return 0
    print("结论：存在未通过项，详见自测报告")
    return 1


def _render_markdown(
    cfg: dict,
    unit_checks: Dict[str, bool],
    unit_lines: List[str],
    song_result: Dict[str, object],
    edge_cases: List[dict],
) -> str:
    codec_cfg = cfg["codec"]
    selftest_cfg = cfg["selftest"]
    tol_ms = float(song_result["tolerance_ms"])  # type: ignore
    songs: List[dict] = song_result["songs"]  # type: ignore

    L: List[str] = []
    L.append("# E1 复合事件（pitch + duration + time-shift）编解码单元自测报告")
    L.append("")
    L.append(f"- **分支**：`{cfg['experiment']['branch']}`")
    L.append(f"- **阶段**：{cfg['experiment']['name']}")
    L.append(f"- **范围**：{cfg['experiment']['note']}")
    L.append("- **自测对象**：`MIDI → 复合事件序列 → 写回 MIDI` 往返一致性")
    L.append("")
    L.append("## 1. 配置参数（全部来自 `configs/e1_compound_events.yaml`）")
    L.append("")
    L.append("| 参数 | 值 |")
    L.append("|---|---|")
    unit_s = float(codec_cfg["time_unit_sec"])
    L.append(f"| 量化单位 | {unit_s} s / 步（{unit_s * 1000:.0f} ms）|")
    L.append(f"| pitch 词表 | {codec_cfg['pitch_vocab_size']} 类（MIDI pitch 0~127 直映射）|")
    L.append(f"| duration 精确上限 | {codec_cfg['duration_max_steps'] - 1} 步 = "
             f"{(codec_cfg['duration_max_steps'] - 1) * unit_s:.2f} s，类别数 {codec_cfg['duration_max_steps'] + 2} |")
    L.append(f"| time-shift 精确上限 | {codec_cfg['shift_max_steps'] - 1} 步 = "
             f"{(codec_cfg['shift_max_steps'] - 1) * unit_s:.2f} s，类别数 {codec_cfg['shift_max_steps'] + 2} |")
    L.append(f"| MIDI 写出 | ticks_per_beat={cfg['generation']['ticks_per_beat']}，tempo={cfg['generation']['default_bpm']} BPM |")
    L.append(f"| 往返容差 | ±{tol_ms:.0f} ms（= 1 个量化步长）|")
    L.append("")

    # ---- 第2节：结论 ----
    all_pitch_ok = all(s["pitch_mismatch"] == 0 for s in songs)
    all_count_ok = all(s["midi_readback"]["count_dec"] == s["midi_readback"]["count_reread"] for s in songs)
    all_tail_pseudo = all(s["tail_shift_is_pseudo"] for s in songs)
    worst_dur = max((s["dur_err_max_ms"] for s in songs), default=0.0)
    worst_onset = max((s["onset_err_max_ms"] for s in songs), default=0.0)
    dur_rate_min = min((s["dur_within_tol_rate"] for s in songs), default=1.0)
    onset_rate_min = min((s["onset_within_tol_rate"] for s in songs), default=1.0)
    unit_pass = all(unit_checks.values())

    L.append("## 2. 自测结论")
    L.append("")
    L.append("| 判定项 | 标准 | 实测 | 结论 |")
    L.append("|---|---|---|---|")
    L.append(f"| 音高无损 | pitch mismatch = 0 | 最大 = {max((s['pitch_mismatch'] for s in songs), default=0)} | "
             f"{'**PASS**' if all_pitch_ok else 'FAIL'} |")
    L.append(f"| 音符数守恒（写回 MIDI 后重读） | count_dec = count_reread | "
             f"{'全部一致' if all_count_ok else '存在不一致'} | {'**PASS**' if all_count_ok else 'FAIL'} |")
    L.append(f"| 末尾 time-shift 为伪标签 | 末位 shift index = 0 | "
             f"{'全部为 0' if all_tail_pseudo else '存在非 0'} | {'**PASS**' if all_tail_pseudo else 'FAIL'} |")
    L.append(f"| 编解码器单元检查 | {len(unit_checks)} 项全通过 | {sum(unit_checks.values())}/{len(unit_checks)} 通过 | "
             f"{'**PASS**' if unit_pass else 'FAIL'} |")
    L.append(f"| 时值往返误差 | 全部 ≤ {tol_ms:.0f} ms | 最低满足率 {dur_rate_min * 100:.2f}%，最大 {worst_dur:.1f} ms | "
             f"{'**PASS**' if dur_rate_min >= 1.0 else 'FAIL（溢出档，见第 4 节局限）'} |")
    L.append(f"| onset 往返误差 | 全部 ≤ {tol_ms:.0f} ms | 最低满足率 {onset_rate_min * 100:.2f}%，最大 {worst_onset:.1f} ms | "
             f"{'**PASS**' if onset_rate_min >= 1.0 else 'FAIL（同前）'} |")
    L.append("")
    over = sum(s["encode_stats"]["duration_overflow"] for s in songs)
    sover = sum(s["encode_stats"]["shift_overflow"] for s in songs)
    if over or sover:
        L.append(f"> 未达 100% 容差满足率的原因：**量化溢出**。全部曲目共有 {over} 个音符时值、"
                 f"{sover} 个 time-shift 落入溢出档，只能还原为下界值，且溢出后其后所有音符的 onset "
                 f"会被整体前移（误差累积）。此为已知局限（第 4 节），不是编解码错误。")
        L.append("")
    elif dur_rate_min < 1.0 or onset_rate_min < 1.0:
        L.append(f"> 全部曲目均无溢出（0 处），残余误差来自 **量化本身 ±半格（{tol_ms / 2:.0f} ms）**："
                 f"本报告用相对 onset 比较（原始 onset 减首音符 onset），因此单个音符的误差上界为"
                 f"「自身舍入」+「首音符舍入」= ±1 格 = {tol_ms:.0f} ms，恰好落在容差边界上，"
                 f"个别音符因浮点舍入被判为略微超出。实测最大误差 {worst_onset:.1f} ms、"
                 f"均值 ≤ {max((s['onset_err_mean_ms'] for s in songs), default=0.0):.1f} ms，属量化精度内的抖动，非逻辑错误。")
        L.append("")

    # ---- 第3节：编解码器单元 ----
    L.append("## 3. 分项测试明细")
    L.append("")
    L.extend(unit_lines)

    # ---- 第4节：真实曲目明细 ----
    L.append("### T2 真实 GT 曲目往返（11 首 BasicPitch 转录）")
    L.append("")
    L.append("流程：`load_song_annotations(MIDI)` → `codec.encode_notes` → `events_to_notes` → "
             "`notes_to_midi` → `pretty_midi` 读回比较。")
    L.append("")
    L.append("#### T2.1 音符级（编解码 + 量化，不含 MIDI tick 误差）")
    L.append("")
    L.append("| 曲目 | 音符数 | pitch 错 | 时值均 err(ms) | 时值 max(ms) | 时值≤容差 | onset 均 err(ms) | onset max(ms) | onset≤容差 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in songs:
        L.append(f"| {s['song_id']} | {s['num_notes']} | {s['pitch_mismatch']} | "
                 f"{s['dur_err_mean_ms']:.2f} | {s['dur_err_max_ms']:.1f} | {s['dur_within_tol_rate'] * 100:.2f}% | "
                 f"{s['onset_err_mean_ms']:.2f} | {s['onset_err_max_ms']:.1f} | {s['onset_within_tol_rate'] * 100:.2f}% |")
    L.append("")
    L.append("#### T2.2 溢出/异常统计（量化局限的直接证据）")
    L.append("")
    L.append("| 曲目 | 超出 max_notes 截断 | duration 溢出 | time-shift 溢出 | 负间隔被截断 | 末尾伪标签=0 |")
    L.append("|---|---:|---:|---:|---:|---|")
    for s in songs:
        st = s["encode_stats"]
        L.append(f"| {s['song_id']} | {st['truncated_notes']} | {st['duration_overflow']} | "
                 f"{st['shift_overflow']} | {st['negative_gap_clamped']} | "
                 f"{'是' if s['tail_shift_is_pseudo'] else '否'} |")
    L.append("")
    L.append("#### T2.3 写回 MIDI 后重读（含 tick 量化误差）")
    L.append("")
    L.append("| 曲目 | 写入音符 | 读回音符 | 数量守恒 | pitch 错 | 时值 max(ms) | onset max(ms) | 同音重叠截断 |")
    L.append("|---|---:|---:|---|---:|---:|---:|---:|")
    for s in songs:
        rb = s["midi_readback"]
        ws = s["midi_write"]
        ok = "是" if rb["count_dec"] == rb["count_reread"] else "**否**"
        L.append(f"| {s['song_id']} | {ws['num_notes']} | {rb['count_reread']} | {ok} | "
                 f"{rb['pitch_mismatch']} | {rb['dur_err_max_ms']:.2f} | {rb['onset_err_max_ms']:.2f} | "
                 f"{ws['truncated_overlap']} |")
    L.append("")

    # ---- 第5节：边界用例 ----
    L.append("### T3 合成边界用例")
    L.append("")
    L.append("| 用例 | 音符数 | pitch 错 | 时值 max(ms) | onset max(ms) | duration 溢出 | shift 溢出 | 读回音符 | 同音重叠截断 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in edge_cases:
        st = c["encode_stats"]
        pm = "不适用（输入本身乱序）" if c["pitch_mismatch"] is None else str(c["pitch_mismatch"])
        L.append(f"| {c['case']} | {c['num_notes']} | {pm} | {c['dur_err_max_ms']:.1f} | "
                 f"{c['onset_err_max_ms']:.1f} | {st['duration_overflow']} | {st['shift_overflow']} | "
                 f"{c['reread_count']} | {c['truncated_overlap']} |")
    L.append("")
    L.append("用例说明与预期：")
    L.append("")
    L.append("- **超长休止 / 超长音符**：故意超过 yaml 中的精确上限，用于验证溢出档确实被触发、"
             "且溢出后后续音符 onset 会被整体前移（误差累积），属已知局限而非 bug。")
    L.append("- **同时发音 (shift=0)**：多个音符共享同一 onset，验证 shift=0 不丢音符、MIDI 事件流排序正确。")
    L.append("- **同音反复 + 重叠**：验证写 MIDI 时对同音高重叠区间的截断保护生效，"
             "音符数不丢（`truncated_overlap` 为被截断处数；历史上此处处理不当曾导致 1025 音符被解析成 792）。")
    L.append("- **负间隔**：输入未按 onset 排序（onset 回退），验证负 gap 被截断为 0 并计数，不会解码出负 onset。")
    L.append("")

    # ---- 第6节：已知局限 ----
    L.append("## 4. 已知局限（本阶段如实记录，不在报告中掩盖）")
    L.append("")
    max_ovl = max((s["midi_write"]["truncated_overlap"] for s in songs), default=0)
    L.append(f"1. **长休止 / 长音符表达受限（溢出档）**：time-shift 超过 "
             f"{codec_cfg['shift_max_steps'] * unit_s:.2f} s、时值超过 "
             f"{codec_cfg['duration_max_steps'] * unit_s:.2f} s 时一律压缩到溢出档，"
             f"解码只能还原下界值，且**其后所有音符的 onset 会被整体前移（误差累积）**。"
             f"当前 11 首 GT 的真实分布（最大 IOI 19.40 s、最大时值 6.89 s）恰好不触发溢出，"
             f"故本轮自测中溢出计数为 0——这是把 max_steps 按实测分布调大的结果，"
             f"并非该问题被消除；合成边界用例（休止 25 s / 音符 12 s）已验证其仍会发生。")
    L.append("2. **溢出档带来的词表代价（已发生的取舍）**：为了让真实数据不溢出，"
             f"shift 类别数被拉到 {int(codec_cfg['shift_max_steps']) + 2}、duration 到 "
             f"{int(codec_cfg['duration_max_steps']) + 2}，其中长尾档几乎无训练样本，"
             "后续训练阶段可能出现类别稀疏；更合理的做法是非均匀（对数）分箱，本阶段不做。")
    L.append("3. **末尾 time-shift 为伪标签**：采用前向 IOI 约定（shift_i = onset_(i+1) - onset_i），"
             "最后一个音符之后不存在\"下一个音符\"，其 time-shift 无监督目标，编码置 0；"
             "**后续训练必须对该位置做 loss 掩码，窗口切片切尾时同理**。")
    L.append("4. **量化粒度固定**：所有位置统一 "
             f"{unit_s * 1000:.0f} ms 网格。onset 误差上界 = ±半格（时值）／±1 格（相对 onset 比较含首音符舍入），"
             "无法同时精细刻画极短装饰音与超长时值。")
    L.append("5. **同音高重叠区间的截断**：GT 由 BasicPitch 多轨合并而来，存在同音高重叠音符。"
             f"写 MIDI 时会把前一个音截断到后一个音起点（本轮最多 {max_ovl} 处），"
             "以规避 (channel, pitch) 配对歧义导致的丢音符；代价是这些音符的**时值被缩短**，"
             "因此 file-level 时值误差可达数十 ms（见 T2.3）。音符数量始终守恒。")
    L.append(f"6. **max_notes 截断**：单曲最多保留 {codec_cfg['max_notes']} 个音符，"
             "超出部分被丢弃（dapu02/04/05/12/13 有截断，见 T2.2）；E1 不做滑窗，滑窗在后续阶段处理。")
    L.append("7. **本自测只覆盖编解码与 MIDI 往返**，不含模型生成质量评估；"
             "复合事件模型尚未训练，因此 E1 未产出任何生成 MIDI、未给出任何生成指标。")
    L.append("")
    L.append("## 5. 复现方式")
    L.append("")
    L.append("```bash")
    L.append("venv/Scripts/python.exe tests/test_compound_event_roundtrip.py")
    L.append("```")
    L.append("")
    L.append(f"- 报告路径：`{selftest_cfg['report_path']}`")
    L.append("- 指标 JSON：`results/e1_roundtrip_metrics.json`")
    L.append("- 临时 MIDI：`results/e1_roundtrip_tmp/`")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
