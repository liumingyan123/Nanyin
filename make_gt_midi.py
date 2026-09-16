"""从 BasicPitch AMT 转录（output/BasicPitch/{song}.mid）生成 GT 主干旋律 .mid 文件。

GT(原始南音) 参照 = 主干旋律 midi，即投入训练/评估的 BasicPitch 单轨转录
（BasicPitch 对 dapu 大谱音频的自动转录，含低音与装饰音的全部主干音符），
不是 structured_data 后处理产物，也不是 v4 派生四声部合奏 midi。

输入: output/BasicPitch/{song}.mid   (BasicPitch 转录的单轨 midi)
输出: results/gt_midi/{song}/{song}_gt.mid   (主干单轨，program=105 琵琶)

数据源规范（2026-08-31）：
  - 训练与评估的唯一数据源 = BasicPitch 转录 midi（output/BasicPitch/*.mid）
  - structured_data/*_midi.json 为后处理产物，判定不准确，整体弃用
"""
import os

import pretty_midi

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, "output", "BasicPitch")
OUT = os.path.join(SCRIPT_DIR, "results", "gt_midi")
# 评估/合奏仅需 2 首验证集曲目
SONGS = ["dapu14kouhuangtian", "dapu15wujinjiao"]


def build(song_id: str) -> str:
    mid_path = os.path.join(BASE, f"{song_id}.mid")
    if not os.path.exists(mid_path):
        print(f"[缺失] {song_id}: 未找到 BasicPitch 转录 {mid_path}")
        return ""

    src = pretty_midi.PrettyMIDI(mid_path)

    # BasicPitch 转录为单轨；若有多个轨道，合并全部音符作为主干
    notes = []
    for inst in src.instruments:
        notes.extend(inst.notes)
    if not notes:
        print(f"[缺失] {song_id}: BasicPitch midi 无音符")
        return ""

    # 按起始时间排序，保证时间轴有序
    notes.sort(key=lambda n: (n.start, n.pitch))

    mid = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(
        program=105,
        is_drum=False,
        name="pipa",  # 用 ASCII 名称，避免 mido latin-1 编码错误
    )
    for n in notes:
        inst.notes.append(
            pretty_midi.Note(
                velocity=max(1, int(n.velocity)),
                pitch=int(n.pitch),
                start=float(n.start),
                end=float(n.end),
            )
        )
    mid.instruments.append(inst)

    out_dir = os.path.join(OUT, song_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{song_id}_gt.mid")
    mid.write(out_path)
    print(
        f"[OK] {song_id}: {len(notes)} 音符, 时长 {mid.get_end_time():.2f}s, "
        f"pitch[{min(n.pitch for n in notes)},{max(n.pitch for n in notes)}] -> {out_path}"
    )
    return out_path


if __name__ == "__main__":
    for s in SONGS:
        build(s)
