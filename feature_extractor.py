import os
import json
import librosa
import numpy as np
import time
import sys

# 配置路径
BASE_PATH = r"D:\科研\NanyinPingGu\南音数据集\nanyin_final"
STRUCTURED_DIR = os.path.join(BASE_PATH, "structured_data")
LOG_FILE = os.path.join(BASE_PATH, "feature_extraction.log")

def log_message(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] {message}"
    print(msg)
    sys.stdout.flush()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def extract_audio_features(audio_path):
    """提取音频核心特征 (优化版)"""
    start_time = time.time()
    log_message(f"  Analyzing audio: {os.path.basename(audio_path)}")
    
    # 1. 加载音频
    log_message("    - Loading audio...")
    y_full, sr_full = librosa.load(audio_path, sr=22050)
    
    # 22050 -> 11025 resample to speed up
    y_fast = librosa.resample(y_full, orig_sr=22050, target_sr=11025)
    sr_fast = 11025
    
    # 2. 提取 F0
    log_message("    - Extracting F0 (YIN)...")
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C7')
    f0 = librosa.yin(y_fast, fmin=fmin, fmax=fmax, sr=sr_fast, hop_length=1024)
    f0_mean = np.nanmean(f0) if np.any(~np.isnan(f0)) else 0
    
    # 3. 提取节拍
    log_message("    - Analyzing Tempo...")
    tempo, beats = librosa.beat.beat_track(y=y_fast, sr=sr_fast)
    if isinstance(tempo, np.ndarray): tempo = tempo[0]
    
    # 4. 提取调式
    log_message("    - Identifying Key...")
    chroma = librosa.feature.chroma_cqt(y=y_fast, sr=sr_fast)
    chroma_mean = np.mean(chroma, axis=1)
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    estimated_key = keys[np.argmax(chroma_mean)]
    
    # 5. 提取 MFCC
    log_message("    - Extracting MFCC...")
    mfccs = librosa.feature.mfcc(y=y_full, sr=sr_full, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1).tolist()
    
    duration = librosa.get_duration(y=y_full, sr=sr_full)
    elapsed = time.time() - start_time
    log_message(f"    Success: Completed in {elapsed:.2f}s")
    
    return {
        "f0_mean_hz": round(float(f0_mean), 2),
        "tempo_bpm": round(float(tempo), 2),
        "estimated_key": estimated_key,
        "mfcc_mean": [round(float(m), 3) for m in mfcc_mean],
        "sample_rate": 22050,
        "duration_sec": round(duration, 2)
    }

def process_all():
    log_message("Starting batch feature extraction...")
    song_ids = [d for d in os.listdir(STRUCTURED_DIR) if os.path.isdir(os.path.join(STRUCTURED_DIR, d))]
    total = len(song_ids)
    
    for i, song_id in enumerate(song_ids, 1):
        song_path = os.path.join(STRUCTURED_DIR, song_id)
        audio_file = os.path.join(song_path, "audio.wav")
        
        if not os.path.exists(audio_file):
            continue
            
        feat_files = [f for f in os.listdir(song_path) if f.endswith("_features.json")]
        if not feat_files: continue
        feat_path = os.path.join(song_path, feat_files[0])
        
        try:
            with open(feat_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if "audio_features" in data and data["audio_features"]:
                log_message(f"[{i}/{total}] Skipping processed: {song_id}")
                continue
        except:
            pass

        log_message(f"[{i}/{total}] Processing: {song_id}")
        
        try:
            features = extract_audio_features(audio_file)
            
            with open(feat_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            data["audio_features"] = features
            
            with open(feat_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            log_message(f"    Results saved for {song_id}")
            
        except Exception as e:
            log_message(f"    Error processing {song_id}: {str(e)}")

if __name__ == "__main__":
    process_all()
