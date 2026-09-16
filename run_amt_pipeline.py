# -*- coding: utf-8 -*-
"""
run_amt_pipeline.py —— 南音 AMT 转录评估完整流水线（主入口）
============================================================

需求步骤：
  步骤1 音频转录渲染：
    - test_audio/ 存放全部原始 dapu 无人声音频
    - 批量调用 Basic-Pitch / YourMT3+ 转录 MIDI
      输出: output/BasicPitch, output/yourmt3plus
    - 用统一合成器把全部 MIDI 渲染成 wav -> 各模型目录 render_wav/
  步骤2 评估指标计算（5 项，文献严格对齐）：
    clews_cosine_mean / chroma_dtw_similarity / pitch_contour_pearson /
    onset_alignment_rate(±50ms) / beat_sync_chroma_similarity
    -> 每模型每首曲目保存为 {Model}_result.csv
  步骤3 模型对比与统计分析：
    排名(按 CLEWS 均值)、稳定性(方差)、两两配对 t-检验
  步骤4 结果输出：
    model_summary.csv / stat_test_result.txt / amt_report.txt

阶段:
  prepare     —— 从结构化数据复制 dapu 无人声音频到 test_audio/
  transcribe  —— 批量转录（三模型）
  render      —— 统一渲染 MIDI -> render_wav/
  evaluate    —— 计算 5 项指标
  stats       —— 统计分析
  all         —— 依次执行以上全部

用法:
  venv\\Scripts\\python.exe run_amt_pipeline.py --stage all
  venv\\Scripts\\python.exe run_amt_pipeline.py --stage transcribe --models basicpitch
  venv\\Scripts\\python.exe run_amt_pipeline.py --stage render --force
  venv\\Scripts\\python.exe run_amt_pipeline.py --stage evaluate --models yourmt3plus
"""
import argparse
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from amt_eval_pipeline import config          # noqa: E402
from amt_eval_pipeline import render as render_mod      # noqa: E402
from amt_eval_pipeline import metrics as metrics_mod    # noqa: E402
from amt_eval_pipeline import statistics as stats_mod   # noqa: E402

logger = logging.getLogger("amt.pipeline")


def _utf8(stream):
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


sys.stdout = _utf8(sys.stdout)
sys.stderr = _utf8(sys.stderr)


def setup_logging():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if root.handlers:
        return
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    fh = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


# ------------------------------------------------------------------------------
# 辅助
# ------------------------------------------------------------------------------

MODEL_ALIASES = {
    "basicpitch": "BasicPitch",
    "yourmt3plus": "yourmt3plus",
    "yourmt3": "yourmt3plus",
}


def resolve_models(models_arg):
    if not models_arg or models_arg == "all":
        return list(config.MODEL_DIRS.keys())
    out = []
    for m in str(models_arg).replace(" ", "").split(","):
        key = m.lower()
        if key not in MODEL_ALIASES:
            logger.warning(f"未知模型: {m}，忽略")
            continue
        if MODEL_ALIASES[key] not in out:
            out.append(MODEL_ALIASES[key])
    if not out:
        out = list(config.MODEL_DIRS.keys())
    return out


def get_test_songs():
    """test_audio/ 下全部 wav 的主干名（作为测试乐曲）。"""
    d = config.TEST_AUDIO_DIR
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.wav"))


def win_to_wsl(path_str):
    """Windows 绝对路径 -> WSL 可访问路径 (/mnt/d/...)。"""
    ap = os.path.abspath(str(path_str))
    drive, rest = os.path.splitdrive(ap)
    drive = drive.lower().rstrip(":")
    rest = rest.replace("\\", "/").lstrip("/")
    return f"/mnt/{drive}/{rest}"


# ------------------------------------------------------------------------------
# 阶段 0：prepare —— 准备 test_audio
# ------------------------------------------------------------------------------

def stage_prepare(force=False, limit=0):
    src = config.STRUCTURED_DATA_DIR
    if not src.is_dir():
        logger.error(f"结构化数据目录不存在: {src}")
        return
    dapu_dirs = sorted(p for p in src.iterdir() if p.is_dir() and p.name.startswith("dapu"))
    if limit:
        dapu_dirs = dapu_dirs[:limit]
    if not dapu_dirs:
        logger.warning(f"{src} 下没有 dapu* 目录")
        return
    config.TEST_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for d in dapu_dirs:
        audio = d / "audio.wav"
        if not audio.is_file():
            logger.warning(f"{d.name} 缺少 audio.wav，跳过")
            continue
        dst = config.TEST_AUDIO_DIR / f"{d.name}.wav"
        if dst.exists() and not force:
            continue
        shutil.copy2(audio, dst)
        logger.info(f"prepare: test_audio/{dst.name} ({audio.stat().st_size/1024/1024:.1f}MB)")
    songs = get_test_songs()
    logger.info(f"prepare: test_audio 共 {len(songs)} 首: {songs}")


# ------------------------------------------------------------------------------
# 阶段 1：transcribe —— 批量转录
# ------------------------------------------------------------------------------

def _run_subprocess(cmd, timeout, tag):
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
        out = (r.stdout or b"").decode("utf-8", "replace")
        err = (r.stderr or b"").decode("utf-8", "replace")
        for line in out.splitlines():
            if line.strip():
                logger.info(f"[{tag}] {line.strip()}")
        if err.strip():
            logger.warning(f"[{tag}][stderr] {err.strip()[-600:]}")
        return r.returncode
    except subprocess.TimeoutExpired:
        logger.error(f"[{tag}] 超时(>{timeout}s)")
        return -1
    except Exception as e:
        logger.error(f"[{tag}] 调用失败: {type(e).__name__}: {e}")
        return -1


def _transcribe_basicpitch(songs, force):
    py = config.PROJECT_ROOT / "venv38" / "Scripts" / "python.exe"
    script = BASE / "amt_eval_pipeline" / "transcribe_basicpitch.py"
    if not py.is_file():
        logger.error(f"venv38 Python 不存在: {py}")
        return
    out_dir = config.MODEL_DIRS["BasicPitch"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = [s for s in songs if force or not (out_dir / f"{s}.mid").is_file()]
    if not pending:
        logger.info("[BasicPitch] 全部已转录，跳过")
        return
    for song in pending:
        wav = config.TEST_AUDIO_DIR / f"{song}.wav"
        midi = out_dir / f"{song}.mid"
        logger.info(f"[BasicPitch] 转录 {song} ...")
        rc = _run_subprocess(
            [str(py), str(script), "--input", str(wav), "--output", str(midi)],
            timeout=7200, tag="BasicPitch",
        )
        if rc == 0 and midi.is_file():
            logger.info(f"[BasicPitch] OK {song} -> {midi.name}")
        else:
            logger.error(f"[BasicPitch] 失败 {song} (exit={rc})")


def _transcribe_yourmt3(songs, force):
    py = config.PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
    script = BASE / "amt_eval_pipeline" / "transcribe_yourmt3.py"
    if not py.is_file():
        logger.error(f"venv Python 不存在: {py}")
        return
    out_dir = config.MODEL_DIRS["yourmt3plus"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = [s for s in songs if force or not (out_dir / f"{s}.mid").is_file()]
    if not pending:
        logger.info("[YourMT3+] 全部已转录，跳过")
        return
    for song in pending:
        wav = config.TEST_AUDIO_DIR / f"{song}.wav"
        midi = out_dir / f"{song}.mid"
        logger.info(f"[YourMT3+] 转录 {song} ...")
        rc = _run_subprocess(
            [str(py), str(script), "--input", str(wav), "--output", str(midi)],
            timeout=12 * 3600, tag="YourMT3+",
        )
        if rc == 0 and midi.is_file():
            logger.info(f"[YourMT3+] OK {song} -> {midi.name}")
        else:
            logger.error(f"[YourMT3+] 失败 {song} (exit={rc})")


def stage_transcribe(models, force, limit=0):
    songs = get_test_songs()
    if limit:
        songs = songs[:limit]
    if not songs:
        logger.error("test_audio 为空，请先执行 prepare 或手动放入 wav")
        return
    logger.info(f"transcribe: 共 {len(songs)} 首: {songs}")
    if "BasicPitch" in models:
        _transcribe_basicpitch(songs, force)
    if "yourmt3plus" in models:
        _transcribe_yourmt3(songs, force)


# ------------------------------------------------------------------------------
# 阶段 2：render —— 统一渲染
# ------------------------------------------------------------------------------

def stage_render(models, force):
    for model in models:
        midi_dir = config.MODEL_DIRS[model]
        if not midi_dir.is_dir():
            logger.warning(f"[render] {midi_dir} 不存在，跳过")
            continue
        wav_dir = midi_dir / config.RENDER_SUBDIR
        wav_dir.mkdir(parents=True, exist_ok=True)
        midis = sorted(midi_dir.glob("*.mid"))
        if not midis:
            logger.warning(f"[render] {model} 目录下无 MIDI")
            continue
        for mp in midis:
            wp = wav_dir / (mp.stem + ".wav")
            if wp.exists() and not force:
                continue
            try:
                render_mod.render_midi_to_wav(mp, wp)
                logger.info(f"[render] {model}/{wp.name}")
            except Exception as e:
                logger.error(f"[render] {model}/{mp.name} 失败: {type(e).__name__}: {e}")


# ------------------------------------------------------------------------------
# 阶段 3：evaluate —— 指标计算
# ------------------------------------------------------------------------------

def stage_evaluate(models, force, limit=0):
    import pandas as pd

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    songs = get_test_songs()
    if limit:
        songs = songs[:limit]
    if not songs:
        logger.error("test_audio 为空")
        return

    embedder = None
    for model in models:
        wav_dir = config.MODEL_DIRS[model] / config.RENDER_SUBDIR
        if not wav_dir.is_dir():
            logger.warning(f"[evaluate] {model} 无 render_wav 目录，跳过")
            continue
        rows = []
        for song in songs:
            ref = config.TEST_AUDIO_DIR / f"{song}.wav"
            rend = wav_dir / f"{song}.wav"
            cache = config.CACHE_DIR / f"{model}_{song}.json"
            if cache.is_file() and not force:
                rows.append(json.loads(cache.read_text(encoding="utf-8")))
                continue
            if not rend.is_file():
                logger.warning(f"[evaluate] {model}/{song} 渲染音频缺失，跳过")
                continue
            try:
                if embedder is None:
                    embedder = metrics_mod.CLEWSEmbedder(config.CLEWS_CKPT)
                    logger.info(f"[evaluate] CLEWS 嵌入器就绪: {embedder.name} "
                                f"(device={embedder.device})")
                m = metrics_mod.compute_all_metrics(ref, rend, clews_embedder=embedder)
                row = {"曲名": song}
                for k, v in m.items():
                    row[k] = round(v, 6) if v is not None else None
                cache.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
                rows.append(row)
                vals = "  ".join(f"{k}={v}" for k, v in row.items() if k != "曲名")
                logger.info(f"[evaluate] {model}/{song}: {vals}")
            except Exception as e:
                logger.error(f"[evaluate] {model}/{song} 失败: {type(e).__name__}: {e}")
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(config.RESULT_CSVS[model], index=False, encoding="utf-8-sig")
            logger.info(f"[evaluate] 写入 {config.RESULT_CSVS[model].name} ({len(df)} 行)")


# ------------------------------------------------------------------------------
# 阶段 4：stats —— 统计分析
# ------------------------------------------------------------------------------

def stage_stats():
    import pandas as pd

    tables = {}
    for model, csvp in config.RESULT_CSVS.items():
        if not csvp.is_file():
            logger.warning(f"[stats] 缺少 {csvp.name}，请先运行 evaluate")
            continue
        tables[model] = pd.read_csv(csvp, encoding="utf-8-sig")
    if len(tables) < 2:
        logger.error("[stats] 有效结果表不足(需>=2个模型)，无法对比")
        return
    summary, results, stat_text, report_text = stats_mod.run_all(
        tables,
        config.SUMMARY_CSV,
        config.STAT_TEST_TXT,
        config.REPORT_TXT,
        models=list(tables),
    )
    # 控制台打印简要报告
    print("\n" + report_text)


# ------------------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="南音 AMT 转录评估流水线")
    ap.add_argument("--stage", default="all",
                    choices=["prepare", "transcribe", "render", "evaluate", "stats", "all"],
                    help="执行阶段 (默认 all)")
    ap.add_argument("--models", default="all",
                    help="模型列表，逗号分隔: basicpitch,yourmt3plus (默认 all)")
    ap.add_argument("--force", action="store_true",
                    help="强制重跑（忽略已有产物/缓存）")
    ap.add_argument("--limit", type=int, default=0,
                    help="仅处理前 N 首（调试用，默认全部）")
    args = ap.parse_args()

    setup_logging()
    t0 = time.time()
    models = resolve_models(args.models)
    logger.info(f"流水线启动 stage={args.stage} models={models} force={args.force} limit={args.limit}")

    stage = args.stage
    if stage in ("prepare", "all"):
        stage_prepare(force=args.force, limit=args.limit)
    if stage in ("transcribe", "all"):
        stage_transcribe(models, force=args.force, limit=args.limit)
    if stage in ("render", "all"):
        stage_render(models, force=args.force)
    if stage in ("evaluate", "all"):
        stage_evaluate(models, force=args.force, limit=args.limit)
    if stage in ("stats", "all"):
        stage_stats()

    logger.info(f"流水线完成，总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
