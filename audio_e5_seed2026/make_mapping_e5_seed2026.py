# -*- coding: utf-8 -*-
"""生成 E5（len256 / seed2026）合奏音频听测索引表。

只读 results/e5/generated_midi（不改原 MIDI），扫描 audio_e5_seed2026 下已产出的 wav，
输出 CSV + Markdown 对照表。用于听测索引，不参与任何训练/评估。
"""
import csv
import os
import re
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
SRC = PROJ / "results" / "e5" / "generated_midi"
OUT = PROJ / "audio_e5_seed2026"
TRK = OUT / "tracks"
MIX = OUT / "mix"

PAT = re.compile(r"^(?P<song>[a-z0-9]+)_w(?P<win>\d+)_(?P<model>boyatcn|musicmamba)"
                 r"_seed(?P<seed>\d+)_len(?P<len>\d+)$")
INSTS = ["pipa", "dongxiao", "sanxian", "erxian"]


def main():
    rows = []
    for f in sorted(SRC.glob("*_seed2026_len256.mid")):
        m = PAT.match(f.stem)
        if not m:
            continue
        stem = f.stem
        mix = MIX / f"{stem}_ensemble.wav"
        tracks = {i: TRK / stem / f"ens_{i}.wav" for i in INSTS}
        ok_tracks = sum(1 for p in tracks.values() if p.exists() and p.stat().st_size > 0)
        rows.append({
            "model": m.group("model"),
            "seed": int(m.group("seed")),
            "target_len": int(m.group("len")),
            "song": m.group("song"),
            "window": "w" + m.group("win"),
            "source_midi": f.name,
            "mix_wav": mix.name if mix.exists() else "",
            "pipa_wav": tracks["pipa"].name if tracks["pipa"].exists() else "",
            "dongxiao_wav": tracks["dongxiao"].name if tracks["dongxiao"].exists() else "",
            "sanxian_wav": tracks["sanxian"].name if tracks["sanxian"].exists() else "",
            "erxian_wav": tracks["erxian"].name if tracks["erxian"].exists() else "",
            "tracks_done": f"{ok_tracks}/4",
            "status": "OK" if (mix.exists() and ok_tracks == 4) else "PENDING",
        })

    rows.sort(key=lambda r: (r["model"], r["song"], r["window"]))

    csv_path = OUT / "mapping_e5_seed2026.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md_path = OUT / "mapping_e5_seed2026.md"
    head = ["model", "seed", "target_len", "song", "window", "source_midi",
            "mix_wav", "tracks_done", "status"]
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# E5 合奏音频听测索引（len256 / seed2026 / 10 窗口 / 2 模型）\n\n")
        fh.write("- 渲染管线：V4 原逻辑 `derive_ensemble_v4.py --late-join 0` → "
                 "`ts_render_full.py --seed 2024 --top_k 10 --win 5.0` → `mix_ensemble.py`\n")
        fh.write("- 声部与音量配比（V4 写死，未改动）：pipa 1.0 / dongxiao 0.9 / sanxian 0.7 / erxian 0.8\n")
        fh.write("- 混音文件目录：`audio_e5_seed2026/mix/`，分轨目录：`audio_e5_seed2026/tracks/<stem>/`\n\n")
        fh.write("| " + " | ".join(head) + " |\n")
        fh.write("|" + "---|" * len(head) + "\n")
        for r in rows:
            fh.write("| " + " | ".join(str(r[k]) for k in head) + " |\n")

    ok = sum(1 for r in rows if r["status"] == "OK")
    print(f"rows={len(rows)} ok={ok} pending={len(rows) - ok}")
    print("csv ->", csv_path)
    print("md  ->", md_path)


if __name__ == "__main__":
    main()
