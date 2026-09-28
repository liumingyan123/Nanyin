#!/bin/bash
# E5 · len256 / seed2026 · 四声部合奏音频渲染（纯后处理，不训练/不评估/不改模型）
# 管线严格沿用 V4（与 /root/tokensynth/render_ens_v4_newmidi.sh、render_all_v4d.sh 完全一致）：
#   1) derive_ensemble_v4.py --midi <主干> --outdir <四声部目录> --late-join 0
#      基础主声部 = pipa；后置派生 dongxiao / erxian / sanxian（V4 原逻辑，参数未改动）
#   2) ts_render_full.py --ref refs/ref_<inst>.wav --midi <inst>.mid --out ens_<inst>.wav
#      --seed 2024 --top_k 10 --win 5.0 --device cuda     ← 与 V4 同渲染器/同参考音色/同种子/同窗口
#      （用 ts_render_full 而非 ts_render_multi：V4 为"每轨独立进程、每轨从 seed 2024 重新播种"，
#        multi 版是单次播种、RNG 跨任务连续，与 V4 不等价，故此处不采用）
#   3) mix_ensemble.py --indir <tracks> --out <mix>.wav --prefix ens_
#      （V4 音量配比写死在脚本内：pipa 1.0 / dongxiao 0.9 / sanxian 0.7 / erxian 0.8，未改动）
# 输入（只读）  : results/e5/generated_midi/*_seed2026_len256.mid
# 输出（wav 不入 git）: audio_e5_seed2026/{ensemble_midi,tracks,mix}
set -u
cd /root/tokensynth
export PYTHONPATH=/root/tokensynth
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1

PY=/root/tokensynth_env/bin/python
PROJ=/mnt/d/科研/NanyinPingGu
SRC=$PROJ/results/e5/generated_midi
OUT=$PROJ/audio_e5_seed2026
ENS=$OUT/ensemble_midi
TRK=$OUT/tracks
MIX=$OUT/mix
LOG=$OUT/render_e5_seed2026.log

mkdir -p "$ENS" "$TRK" "$MIX"

for f in "$SRC"/*_seed2026_len256.mid; do
  stem=$(basename "$f" .mid)
  d="$ENS/$stem"
  mkdir -p "$d" "$TRK/$stem"

  # 1) V4 派生（无晚入，与 V4 生成侧一致）
  if [ -f "$d/pipa.mid" ] && [ -f "$d/sanxian.mid" ] && [ -f "$d/dongxiao.mid" ] && [ -f "$d/erxian.mid" ]; then
    echo "[SKIP DERIVE] $stem" | tee -a "$LOG"
  else
    echo "===== $(date) DERIVE $stem =====" | tee -a "$LOG"
    $PY derive_ensemble_v4.py --midi "$f" --outdir "$d" --late-join 0 >> "$LOG" 2>&1
    echo "[derive rc=$?] $stem" | tee -a "$LOG"
  fi

  # 2) 逐轨渲染（顺序与 V4 render_ens_v4_newmidi.sh 一致：dongxiao → erxian → sanxian → pipa）
  for spec in "dongxiao ref_dongxiao" "erxian ref_erxian" "sanxian ref_sanxian" "pipa ref_pipa"; do
    set -- $spec
    name=$1
    ref=$2
    inst_midi="$d/${name}.mid"
    outw="$TRK/$stem/ens_${name}.wav"
    [ -f "$inst_midi" ] || { echo "[缺inst midi] $inst_midi" | tee -a "$LOG"; continue; }
    if [ -f "$outw" ] && [ -s "$outw" ]; then
      echo "[SKIP RENDER] $stem/$name" | tee -a "$LOG"
      continue
    fi
    echo "===== $(date) RENDER $stem/$name =====" | tee -a "$LOG"
    $PY ts_render_full.py \
        --ref "refs/${ref}.wav" --midi "$inst_midi" --out "$outw" \
        --seed 2024 --top_k 10 --win 5.0 --device cuda >> "$LOG" 2>&1
    echo "===== $(date) DONE $stem/$name rc=$? =====" | tee -a "$LOG"
  done

  # 3) 混音
  outm="$MIX/${stem}_ensemble.wav"
  if [ -f "$outm" ] && [ -s "$outm" ]; then
    echo "[SKIP MIX] $stem" | tee -a "$LOG"
  elif ls "$TRK/$stem"/ens_*.wav >/dev/null 2>&1; then
    echo "===== $(date) MIX $stem =====" | tee -a "$LOG"
    $PY mix_ensemble.py --indir "$TRK/$stem" --out "$outm" --prefix ens_ >> "$LOG" 2>&1
    echo "[mix rc=$?] $stem" | tee -a "$LOG"
  else
    echo "[缺轨] $stem" | tee -a "$LOG"
  fi
done

echo "ALL_DONE_AT $(date)" | tee -a "$LOG"
