# -*- coding: utf-8 -*-
"""
dataset_nanyin.py - 南音数据集自定义PyTorch Dataset工具
========================================================

功能概述：
  1. 读取单首曲目的 BasicPitch AMT 转录 MIDI（output/BasicPitch/*.mid，可信数据源）
  2. 全部真值由 MIDI 客观推导（调式/速度/特征/风格/装饰音），不依赖人工标注
  3. 内置11首 dapu 大谱曲目固定划分：9训练集 + 2验证集，无需手动修改
  4. 兼容Windows相对路径，完整文件读取异常捕获
  5. 配套工具函数：快速获取训练集/验证集 Dataset 实例

数据重建规范（2026-08-31 数据源修正）：
  原 18 首指弹唱曲目的 5 份人工标注被判定不准确，整体弃用；
  structured_data/*_midi.json（BasicPitch 转录的后处理产物）同样判定不准确，
  不再作为训练数据源。训练与评估的唯一数据源 = BasicPitch 对 dapu 大谱音频的
  自动转录 midi（output/BasicPitch/*.mid），训练所需全部目标真值由 MIDI 音符
  数据客观推导，保证数据源可信、可复现。

参考文献：
  MelodyGLM         https://arxiv.org/pdf/2309.10738v1
  Controllable Symbolic Music Generation https://www.preprints.org/manuscript/202604.0984
  TCSinger           https://arxiv.org/pdf/2409.15977
  NanyinHGNN         arXiv:2510.03617
  KAD                arXiv:2502.15602

用法示例：
    from dataset_nanyin import NanyinDataset, get_train_dataset, get_val_dataset

    train_ds = get_train_dataset()
    val_ds   = get_val_dataset()

    sample_tokens, target_dict = train_ds[0]
    # target_dict 可直接传入 loss_nanyin.nanyin_total_loss()
"""

import os
import json
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import pretty_midi  # 解析 BasicPitch 转录 midi（训练数据源）
except ImportError:
    pretty_midi = None


# ==============================================================================
# 一、全局常量定义
# ==============================================================================

# 数据源根目录（相对于当前脚本所在项目根目录）
# 2026-08-31 数据源规范：训练/评估唯一数据源 = BasicPitch 对 dapu 大谱音频的
# 自动转录 midi（output/BasicPitch/*.mid）。structured_data/*_midi.json 为后处理
# 产物，判定不准确，整体弃用。
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATASET_BASE = os.path.join(_SCRIPT_DIR, "output", "BasicPitch")

# 11首 dapu 大谱曲目ID列表（BasicPitch AMT 转录重建训练集，2026-08-29）
# 旧 18 首指弹唱曲目的 midi.json 被判定有误，整体弃用；改用 dapu 大谱重训。
ALL_SONG_IDS: List[str] = [
    "dapu01qishouban",        #  1. 起手板
    "dapu02sanmianjinqianjing",  #  2. 三面金钱经
    "dapu03wucaojinqianjing",    #  3. 五操金钱经
    "dapu04bamianjinqianjing",   #  4. 八面金钱经
    "dapu05sishijing",           #  5. 四时景
    "dapu06meihuacao",           #  6. 梅花操
    "dapu10sanbuhe",             #  7. 三不和
    "dapu12sijingban",           #  8. 四静板
    "dapu13kongquezhanping",     #  9. 孔雀展屏
    "dapu14kouhuangtian",        # 10. 叩皇天
    "dapu15wujinjiao",           # 11. 五进教
]

# 固定划分：前9首为训练集，后2首为验证集
TRAIN_SPLIT_IDS: List[str] = ALL_SONG_IDS[:9]    # 索引 0~8
VAL_SPLIT_IDS: List[str]   = ALL_SONG_IDS[9:]     # 索引 9~10

# 南音四种专属调式名称（与 loss_nanyin.py 中 NANYIN_FOUR_MODES 对齐）
NANYIN_FOUR_MODES: List[str] = ["五空管", "五空四仪管", "倍思管", "四空管"]

# 调式名称 → one-hot 索引映射（含常见异体字变体）
MODE_TO_INDEX: Dict[str, int] = {
    "五空管": 0,
    "五空四仪管": 1,
    "五空四乂管": 1,  # "乂" 为 "仪" 的异体字变体，同属第2类调式
    "倍思管": 2,
    "四空管": 3,
}

# JSON文件后缀标识
JSON_SUFFIXES: List[str] = ["alignment", "features", "gongchepu", "lyrics", "midi"]

# 速度分布分箱数量（与 loss_nanyin.py 中 tempo_kl 的 num_tempo_bins 对应）
NUM_TEMPO_BINS: int = 20

# MFCC 特征维度（与 loss_nanyin.py 中 target_features 对齐）
MFCC_DIM: int = 13

# 风格嵌入维度（与 loss_nanyin.py 中 pred_style / target_style 对齐）
STYLE_DIM: int = 32


# ==============================================================================
# 二、JSON文件自动发现与读取
# ==============================================================================

def find_json_file(song_dir: str, suffix: str) -> Optional[str]:
    """
    在曲目目录下自动匹配指定后缀的JSON文件。

    参数:
        song_dir: 曲目目录绝对路径
        suffix:   JSON文件类型后缀，如 "alignment" / "features" / "gongchepu" / "lyrics" / "midi"
    返回:
        匹配到的JSON文件绝对路径；若未找到则返回 None
    """
    if not os.path.isdir(song_dir):
        return None
    try:
        for fname in os.listdir(song_dir):
            if fname.endswith(f"_{suffix}.json"):
                return os.path.join(song_dir, fname)
    except PermissionError:
        return None
    return None


def safe_load_json(filepath: str, song_id: str = "") -> Optional[dict]:
    """
    安全加载JSON文件，带完整异常捕获与调试打印。

    参数:
        filepath: JSON文件路径
        song_id:  曲目ID（用于调试打印）
    返回:
        解析后的字典；读取失败返回 None
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"[dataset_nanyin] 文件未找到 [{song_id}]: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"[dataset_nanyin] JSON解析错误 [{song_id}]: {filepath} — {e}")
        return None
    except PermissionError:
        print(f"[dataset_nanyin] 文件权限不足 [{song_id}]: {filepath}")
        return None
    except Exception as e:
        print(f"[dataset_nanyin] 读取异常 [{song_id}]: {filepath} — {type(e).__name__}: {e}")
        return None


# ==============================================================================
# 三、单首曲目标注整合（纯 BasicPitch MIDI 驱动）
# ==============================================================================

def load_song_annotations(song_id: str, base_dir: str = _DATASET_BASE) -> Optional[Dict[str, dict]]:
    """
    加载单首曲目的 BasicPitch AMT 转录 MIDI（output/BasicPitch/{song_id}.mid）。

    重要（2026-08-31 数据源规范）：
      训练与评估的唯一数据源 = BasicPitch 对 dapu 大谱音频的自动转录 midi
      （output/BasicPitch/*.mid）。structured_data/*_midi.json 为 BasicPitch
      转录的后处理产物，判定不准确，整体弃用，不再读取。
      训练所需的全部真值（调式/速度/特征/风格/装饰音）一律由 MIDI 音符数据
      客观推导，保证数据源可信、可复现。

    参数:
        song_id:   曲目ID（如 dapu01qishouban）
        base_dir:  数据根目录（默认 output/BasicPitch）
    返回:
        {"midi": {"tracks": [{"role": "主旋律（骨音）", "notes": [...]}]}}；
        加载失败返回 None
    """
    mid_path = os.path.join(base_dir, f"{song_id}.mid")
    if not os.path.exists(mid_path):
        print(f"[dataset_nanyin] BasicPitch 转录缺失: {mid_path}")
        return None

    if pretty_midi is None:
        print("[dataset_nanyin] 缺少 pretty_midi 库，无法解析 BasicPitch midi")
        return None

    try:
        pm = pretty_midi.PrettyMIDI(mid_path)
    except Exception as e:
        print(f"[dataset_nanyin] 解析 BasicPitch midi 失败 [{song_id}]: {type(e).__name__}: {e}")
        return None

    # BasicPitch 转录为单轨；若有多个轨道，合并全部音符作为主干旋律
    notes: List[dict] = []
    for inst in pm.instruments:
        for n in inst.notes:
            notes.append({
                "note": int(n.pitch),
                "duration": float(n.end - n.start),
                "start_time": float(n.start),
                "velocity": int(n.velocity),
            })
    if not notes:
        print(f"[dataset_nanyin] BasicPitch midi 无音符 [{song_id}]: {mid_path}")
        return None
    notes.sort(key=lambda x: (x["start_time"], x["note"]))

    midi_data = {
        "tracks": [
            {
                "role": "主旋律（骨音）",
                "program": 4,  # BasicPitch 默认钢琴音色
                "notes": notes,
            }
        ]
    }
    print(f"[dataset_nanyin] 成功加载曲目 [{song_id}] BasicPitch 转录 ({len(notes)} 音符)")
    return {"midi": midi_data}


# ==============================================================================
# 四、标注 → 统一真值特征字典 转换（纯 MIDI 客观推导）
# ==============================================================================

# 南音四调对应的宫调音级集合（MIDI pitch class，C=0）
#   五空管   c=do -> C 宫五声 {C D E G A}
#   五空四仪管 g=do -> G 宫五声 {G A B D E}
#   倍思管   d=do -> D 宫五声 {D E F# A B}
#   四空管   f=do -> F 宫五声 {F G A C D}
_MODE_PENTATONIC: Dict[str, List[int]] = {
    "五空管":     [0, 2, 4, 7, 9],
    "五空四仪管": [7, 9, 11, 2, 4],
    "倍思管":     [2, 4, 6, 9, 11],
    "四空管":     [5, 7, 9, 0, 2],
}
_BPM_MIN, _BPM_MAX = 40.0, 240.0


def estimate_tempo_bpm(start_list: List[float]) -> float:
    """从音符起始时间估计速度（BPM）：
    取相邻起始间隔的中位数（排除休止/重叠异常值），bpm = 60 / 中位间隔。"""
    if len(start_list) < 2:
        return 120.0
    onsets = np.diff(np.sort(np.array(start_list, dtype=np.float64)))
    onsets = onsets[(onsets > 0.05) & (onsets < 3.0)]
    if len(onsets) == 0:
        return 120.0
    med = float(np.median(onsets))
    return float(np.clip(60.0 / med, _BPM_MIN, _BPM_MAX))


def estimate_mode_from_pitch(pitch_list: List[float]) -> Optional[str]:
    """从 MIDI 音高集合估计南音调式：
    计算 pitch class 直方图，与四调五声音级集合匹配，取音级占比最高者。
    占比过低（<0.4，非五声结构）返回 None（此时用均匀分布）。"""
    if not pitch_list:
        return None
    pc = np.bincount(np.array(pitch_list, dtype=int) % 12, minlength=12).astype(np.float64)
    pc /= (pc.sum() + 1e-8)
    best_name, best_score = None, -1.0
    for name, pcs in _MODE_PENTATONIC.items():
        score = float(pc[pcs].sum())
        if score > best_score:
            best_score, best_name = score, name
    if best_score < 0.4:
        return None
    return best_name


def derive_midi_features(
    pitch_list: List[float],
    duration_list: List[float],
    start_list: List[float],
) -> np.ndarray:
    """从 MIDI 音符序列派生 13 维客观特征（不依赖任何人工标注）：
      [音高均值/标准差/音域，时值均值/标准差，音符密度，平均onset间隔，
       pitch-class熵，五声音级占比，平均相邻音高间隔，上行比例，短音/长音占比]"""
    pitch = np.array(pitch_list, dtype=np.float64)
    dur = np.array(duration_list, dtype=np.float64)
    if len(pitch) == 0:
        return np.zeros(MFCC_DIM, dtype=np.float32)

    pc = np.bincount(pitch.astype(int) % 12, minlength=12).astype(np.float64)
    pc /= (pc.sum() + 1e-8)
    pc_entropy = float(-np.sum(pc * np.log(pc + 1e-9)) / np.log(12.0))
    pent_ratio = float(pc[[0, 2, 4, 7, 9]].sum())

    total_dur = float(max(dur.sum(), start_list[-1]) + 1e-3) if start_list else 1.0
    density = float(len(pitch) / total_dur)

    onsets = np.diff(np.sort(np.array(start_list, dtype=np.float64)))
    onsets = onsets[onsets > 0.02]
    mean_onset = float(onsets.mean()) if len(onsets) else 1.0

    adj = np.diff(pitch)
    mean_adj = float(np.abs(adj).mean()) if len(adj) else 0.0
    up_ratio = float((adj > 0).mean()) if len(adj) else 0.5

    feats = np.array([
        (pitch.mean() - 21.0) / 87.0,            # 音高均值（归一化到 MIDI 21~108）
        pitch.std() / 87.0,                       # 音高标准差
        (pitch.max() - pitch.min()) / 87.0,       # 音域
        dur.mean() / 5.0,                         # 平均时值
        dur.std() / 5.0,                          # 时值标准差
        min(density / 5.0, 1.0),                  # 音符密度
        min(mean_onset / 2.0, 1.0),               # 平均 onset 间隔
        pc_entropy,                               # pitch-class 熵
        pent_ratio,                               # 五声音级占比
        min(mean_adj / 24.0, 1.0),                # 平均相邻音高间隔
        up_ratio,                                 # 音高上行比例
        float((dur < 0.2).mean()),                # 短音占比
        float((dur > 1.5).mean()),                # 长音占比
    ], dtype=np.float32)
    return feats


def build_target_dict(annotations: Dict[str, dict]) -> Dict[str, torch.Tensor]:
    """
    将 AMT 转录 MIDI 整合为统一真值特征字典，
    输出格式完全匹配 loss_nanyin.py 中 nanyin_total_loss() 的 target_dict 参数。

    数据源（2026-08-31 数据源修正）：
      仅使用 BasicPitch AMT 转录 midi（output/BasicPitch/*.mid，可信数据源）。
      structured_data/*_midi.json 为 BasicPitch 转录的后处理产物，判定不准确，
      整体弃用，不再读取。
      - target_tokens / pitch / duration 由音符直接编码
      - target_ornament 由音符时值+邻音跳进的客观规则推导
      - target_tempo_dist 由音符起始间隔估计 BPM
      - target_mode_dist 由 pitch class 直方图匹配南音四调
      - target_features 由音符序列派生的 13 维客观统计特征
      - target_style 由上述特征 + 调式 + 速度固定投影（可复现）
    原 alignment / features / gongchepu / lyrics 四份标注不准确，不再读取。

    参数:
        annotations: load_song_annotations() 返回的 {"midi": ...} 字典
    返回:
        统一真值特征字典，字段与 loss_nanyin.py 入参一致
    """
    # ---- 1. 从 midi.json 提取 note-level 序列特征 ----
    midi_data = annotations.get("midi", {})

    pitch_list: List[float] = []       # MIDI音高值
    duration_list: List[float] = []    # 时值（秒）
    start_list: List[float] = []       # 起始时间（秒）
    token_list: List[int] = []         # token ID序列

    tracks = midi_data.get("tracks", [])
    # 优先使用主旋律轨道（琵琶/骨音），其次取第一个轨道
    melody_track = None
    for track in tracks:
        if track.get("role", "") in ("主旋律（骨音）", "主旋律"):
            melody_track = track
            break
    if melody_track is None and len(tracks) > 0:
        melody_track = tracks[0]

    if melody_track is not None:
        notes = melody_track.get("notes", [])
        for note in notes:
            pitch = note.get("note", 60)          # MIDI note number, 默认C4
            duration = note.get("duration", 0.5)   # 时值（秒），默认0.5
            start = note.get("start_time", 0.0)    # 起始时间（秒）
            pitch_list.append(float(pitch))
            duration_list.append(float(duration))
            start_list.append(float(start))
            # 将MIDI pitch直接作为token ID（范围21~108，共88个可能值）
            # 映射到 vocab_size=128 的词表中，保留扩展空间
            token_list.append(int(np.clip(pitch, 0, 127)))

    # 若MIDI轨道无音符，构造一个占位序列（至少1个元素以保证模型不崩溃）
    if len(token_list) == 0:
        print("[dataset_nanyin] 警告：MIDI轨道中未找到音符数据，使用占位token")
        token_list.append(0)
        pitch_list.append(60.0)
        duration_list.append(1.0)
        start_list.append(0.0)

    target_tokens   = torch.tensor(token_list, dtype=torch.long)      # (seq_len,)
    target_pitch    = torch.tensor(pitch_list, dtype=torch.float32)   # (seq_len,)
    target_duration = torch.tensor(duration_list, dtype=torch.float32) # (seq_len,)

    # ---- 2. 速度：从音符起始间隔估计 BPM → target_tempo_dist ----
    tempo_bpm = estimate_tempo_bpm(start_list)
    bin_centers = np.linspace(_BPM_MIN, _BPM_MAX, NUM_TEMPO_BINS)
    sigma = 5.0                                          # 高斯标准差
    tempo_dist = np.exp(-0.5 * ((bin_centers - tempo_bpm) / sigma) ** 2)
    tempo_dist = tempo_dist / (tempo_dist.sum() + 1e-8)  # 归一化为概率分布
    target_tempo_dist = torch.tensor(tempo_dist, dtype=torch.float32)  # (num_tempo_bins,)

    # ---- 3. 调式：pitch class 直方图匹配南音四调 → target_mode_dist ----
    mode_name = estimate_mode_from_pitch(pitch_list)
    mode_index = MODE_TO_INDEX.get(mode_name, -1) if mode_name else -1
    if mode_index >= 0:
        mode_onehot = np.zeros(4, dtype=np.float32)
        mode_onehot[mode_index] = 1.0
    else:
        # 无法可靠估计（非五声结构）：均匀分布，不伪造标签
        print(f"[dataset_nanyin] 调式估计不可靠 (name={mode_name})，使用均匀分布")
        mode_onehot = np.full(4, 0.25, dtype=np.float32)
    target_mode_dist = torch.tensor(mode_onehot, dtype=torch.float32)  # (4,)

    # ---- 4. 装饰音：客观规则（短时值 + 邻音跳进）→ target_ornament ----
    num_notes = len(token_list)
    ornament_array = np.zeros(num_notes, dtype=np.float32)
    for i, d in enumerate(duration_list):
        if d < 0.15:                                   # 短音
            prev_gap = abs(pitch_list[i] - pitch_list[i - 1]) if i > 0 else 99.0
            next_gap = abs(pitch_list[i] - pitch_list[i + 1]) if i + 1 < num_notes else 99.0
            if prev_gap >= 3.0 or next_gap >= 3.0:      # 与邻音跳进 >= 大三度
                ornament_array[i] = 1.0
    target_ornament = torch.tensor(ornament_array, dtype=torch.float32)  # (seq_len,)

    # ---- 5. 客观 MIDI 派生特征 → target_features ----
    midi_feats = derive_midi_features(pitch_list, duration_list, start_list)  # (13,)
    target_features = torch.tensor(midi_feats, dtype=torch.float32)          # (feat_dim,)

    # ---- 6. 风格嵌入：MIDI 特征 + 调式 + 速度 → 固定投影（可复现）→ target_style ----
    mode_arr = mode_onehot.astype(np.float32)                                  # (4,)
    tempo_arr = np.array([tempo_bpm / 240.0], dtype=np.float32)                # (1,)
    combined = np.concatenate([midi_feats, mode_arr, tempo_arr], axis=0)       # (18,)
    rng = np.random.RandomState(42)
    projection = rng.randn(18, STYLE_DIM).astype(np.float32) * 0.1
    style_vec = combined @ projection
    style_vec = style_vec / (np.linalg.norm(style_vec) + 1e-8)
    target_style = torch.tensor(style_vec, dtype=torch.float32)                # (style_dim,)

    # ---- 7. 组装完整 target_dict ----
    target_dict = {
        # 序列级（逐音符）真值
        "target_tokens":   target_tokens,
        "target_pitch":    target_pitch,
        "target_duration": target_duration,
        "target_ornament": target_ornament,
        # 全局级真值
        "target_tempo_dist": target_tempo_dist,
        "target_mode_dist":  target_mode_dist,
        "target_features":   target_features,
        "target_style":      target_style,
    }

    return target_dict


# ==============================================================================
# 五、PyTorch Dataset 主类
# ==============================================================================

class NanyinDataset(Dataset):
    """
    南音符号旋律数据集，PyTorch Dataset 子类。

    每首曲目作为一个样本，返回:
      (模型输入token序列, 完整真值标注字典)

    真值标注字典的字段完全匹配 loss_nanyin.py 中 nanyin_total_loss() 的 target_dict 参数。

    参考文献：
      MelodyGLM         https://arxiv.org/pdf/2309.10738v1
      Controllable Symbolic Music Generation https://www.preprints.org/manuscript/202604.0984
      TCSinger           https://arxiv.org/pdf/2409.15977
      NanyinHGNN         arXiv:2510.03617
      KAD                arXiv:2502.15602
    """

    def __init__(self, song_ids: List[str], base_dir: Optional[str] = None, max_seq_len: int = 4096):
        """
        初始化南音数据集。

        参数:
            song_ids:    曲目ID列表
            base_dir:    数据集根目录，默认使用 _DATASET_BASE（相对路径）
            max_seq_len: 最大序列长度，超过该长度的序列会被截断
        """
        super().__init__()
        self.song_ids = song_ids
        self.base_dir = base_dir if base_dir is not None else _DATASET_BASE
        self.max_seq_len = max_seq_len

        # 预加载所有曲目的原始标注（惰性加载：首次访问时解析）
        self._cached_annotations: Dict[int, Optional[Dict[str, dict]]] = {}

        print(f"[dataset_nanyin] 初始化 Dataset，共 {len(self.song_ids)} 首曲目")
        print(f"[dataset_nanyin] 数据目录: {self.base_dir}")
        print(f"[dataset_nanyin] 最大序列长度: {self.max_seq_len}")
        print(f"[dataset_nanyin] 曲目列表: {self.song_ids}")

    def _load_index(self, idx: int) -> Optional[Dict[str, dict]]:
        """加载并缓存指定索引的原始标注。"""
        if idx not in self._cached_annotations:
            song_id = self.song_ids[idx]
            print(f"[dataset_nanyin] 正在加载曲目 [{song_id}] (索引 {idx})...")
            self._cached_annotations[idx] = load_song_annotations(song_id, self.base_dir)
        return self._cached_annotations[idx]

    def __len__(self) -> int:
        """返回数据集样本数（即曲目数）。"""
        return len(self.song_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        获取第 idx 首曲目的数据。

        参数:
            idx: 曲目索引
        返回:
            (input_tokens, target_dict)
            - input_tokens: 模型输入的 token 序列，shape=(seq_len,)，dtype=long
            - target_dict:  真值标注字典，包含 target_tokens / target_pitch / target_duration
                            / target_ornament / target_tempo_dist / target_mode_dist
                            / target_features / target_style
        异常:
            IndexError:  索引越界
            RuntimeError: 标注数据加载失败
        """
        if idx < 0 or idx >= len(self.song_ids):
            raise IndexError(
                f"[dataset_nanyin] 索引越界: idx={idx}, 数据集大小={len(self.song_ids)}"
            )

        annotations = self._load_index(idx)
        if annotations is None:
            song_id = self.song_ids[idx]
            raise RuntimeError(
                f"[dataset_nanyin] 无法加载曲目 [{song_id}] (索引 {idx}) 的标注数据，"
                f"请检查文件完整性。"
            )

        # 构建统一真值字典
        target_dict = build_target_dict(annotations)

        # 按 max_seq_len 截断序列级字段（避免超出位置编码长度）
        seq_keys = ["target_tokens", "target_pitch", "target_duration", "target_ornament"]
        for key in seq_keys:
            if key in target_dict and target_dict[key].shape[0] > self.max_seq_len:
                print(f"[dataset_nanyin] 曲目 [{self.song_ids[idx]}] {key} 长度 {target_dict[key].shape[0]} > {self.max_seq_len}，已截断")
                target_dict[key] = target_dict[key][:self.max_seq_len]

        # 输入token序列 = 目标token序列（在自回归生成中，输入是shifted target）
        # 训练时模型将 input_tokens 作为输入，target_dict["target_tokens"] 作为标签
        input_tokens = target_dict["target_tokens"].clone()

        return input_tokens, target_dict

    def get_song_id(self, idx: int) -> str:
        """返回第 idx 首曲目的ID。"""
        return self.song_ids[idx]


# ==============================================================================
# 六、配套工具函数
# ==============================================================================

def get_train_dataset(base_dir: Optional[str] = None, max_seq_len: int = 4096) -> NanyinDataset:
    """
    获取训练集 Dataset 实例（固定9首 dapu 大谱）。

    参数:
        base_dir:    数据集根目录，默认使用 _DATASET_BASE
        max_seq_len: 最大序列长度，超过该长度的序列会被截断
    返回:
        NanyinDataset 实例，包含训练集曲目
    """
    print(f"[dataset_nanyin] ===== 创建训练集 Dataset =====")
    print(f"[dataset_nanyin] 训练集曲目数: {len(TRAIN_SPLIT_IDS)} / 11")
    print(f"[dataset_nanyin] 训练集曲目: {TRAIN_SPLIT_IDS}")
    return NanyinDataset(song_ids=TRAIN_SPLIT_IDS, base_dir=base_dir, max_seq_len=max_seq_len)


def get_val_dataset(base_dir: Optional[str] = None, max_seq_len: int = 4096) -> NanyinDataset:
    """
    获取验证集 Dataset 实例（固定2首 dapu 大谱）。

    参数:
        base_dir:    数据集根目录，默认使用 _DATASET_BASE
        max_seq_len: 最大序列长度，超过该长度的序列会被截断
    返回:
        NanyinDataset 实例，包含验证集曲目
    """
    print(f"[dataset_nanyin] ===== 创建验证集 Dataset =====")
    print(f"[dataset_nanyin] 验证集曲目数: {len(VAL_SPLIT_IDS)} / 11")
    print(f"[dataset_nanyin] 验证集曲目: {VAL_SPLIT_IDS}")
    return NanyinDataset(song_ids=VAL_SPLIT_IDS, base_dir=base_dir, max_seq_len=max_seq_len)


def get_full_dataset(base_dir: Optional[str] = None, max_seq_len: int = 4096) -> NanyinDataset:
    """
    获取完整18首曲目的 Dataset 实例（不做训练/验证划分）。

    参数:
        base_dir:    数据集根目录，默认使用 _DATASET_BASE
        max_seq_len: 最大序列长度，超过该长度的序列会被截断
    返回:
        NanyinDataset 实例，包含全部18首曲目
    """
    print(f"[dataset_nanyin] ===== 创建完整数据集 Dataset =====")
    print(f"[dataset_nanyin] 全部曲目数: {len(ALL_SONG_IDS)}")
    return NanyinDataset(song_ids=ALL_SONG_IDS, base_dir=base_dir, max_seq_len=max_seq_len)




def get_dataset_statistics(dataset: NanyinDataset) -> Dict[str, object]:
    """
    获取 Dataset 的统计信息（用于调试与日志）。

    参数:
        dataset: NanyinDataset 实例
    返回:
        包含曲目数、平均序列长度、调式分布、速度范围的统计字典
    """
    stats: Dict[str, object] = {
        "num_songs": len(dataset),
        "avg_seq_len": 0.0,
        "mode_distribution": {},
        "tempo_range": [float("inf"), float("-inf")],
    }

    seq_lens = []
    mode_counts: Dict[str, int] = {}

    for i in range(len(dataset)):
        try:
            _, target_dict = dataset[i]
            seq_len = target_dict["target_tokens"].shape[0]
            seq_lens.append(seq_len)

            # 解码调式
            mode_dist = target_dict["target_mode_dist"].numpy()
            mode_idx = int(np.argmax(mode_dist))
            if 0 <= mode_idx < 4:
                mode_name = NANYIN_FOUR_MODES[mode_idx]
                mode_counts[mode_name] = mode_counts.get(mode_name, 0) + 1

            # 解码速度（从 target_tempo_dist 中还原 BPM）
            tempo_dist = target_dict["target_tempo_dist"].numpy()
            bpm_min, bpm_max = 40.0, 240.0
            bin_centers = np.linspace(bpm_min, bpm_max, NUM_TEMPO_BINS)
            estimated_bpm = float(np.dot(bin_centers, tempo_dist))
            stats["tempo_range"][0] = min(stats["tempo_range"][0], estimated_bpm)
            stats["tempo_range"][1] = max(stats["tempo_range"][1], estimated_bpm)

        except Exception as e:
            print(f"[dataset_nanyin] 统计时跳过索引 {i}: {e}")

    if seq_lens:
        stats["avg_seq_len"] = float(np.mean(seq_lens))
    stats["mode_distribution"] = mode_counts

    print(f"[dataset_nanyin] ===== Dataset 统计信息 =====")
    print(f"  曲目数:     {stats['num_songs']}")
    print(f"  平均序列长度: {stats['avg_seq_len']:.1f}")
    print(f"  调式分布:   {stats['mode_distribution']}")
    print(f"  速度范围:   {stats['tempo_range']}")

    return stats


def collate_nanyin_batch(
    batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    自定义 collate 函数，用于 DataLoader 批处理。

    处理不同长度序列的 padding，使同一 batch 内所有序列长度对齐。
    target_dict 中非序列字段（如 target_mode_dist）不进行 padding。

    文献来源：
      MelodyGLM (https://arxiv.org/pdf/2309.10738v1)
      该文在旋律生成中使用 pad_packed_sequence 处理变长序列。

    参数:
        batch: List of (input_tokens, target_dict) 样本列表
    返回:
        (batched_input_tokens, batched_target_dict)
    """
    max_len = max(sample[0].shape[0] for sample in batch)
    batch_size = len(batch)

    # 序列级字段（需要 padding）
    seq_keys = [
        "input_tokens",    # 输入token
        "target_tokens",   # 目标token
        "target_pitch",    # 音高
        "target_duration", # 时值
        "target_ornament", # 装饰音标记
    ]

    batched_target: Dict[str, torch.Tensor] = {}
    batched_input_tokens = torch.empty(0)  # 初始化，避免未绑定警告

    for key in seq_keys:
        is_key_input = (key == "input_tokens")
        padded = torch.zeros(batch_size, max_len, dtype=torch.long if "tokens" in key else torch.float32)
        for b_idx, (input_tok, tgt_dict) in enumerate(batch):
            if is_key_input:
                seq = input_tok
            else:
                seq = tgt_dict[key]
            seq_len = seq.shape[0]
            padded[b_idx, :seq_len] = seq
        if is_key_input:
            batched_input_tokens = padded
        else:
            batched_target[key] = padded

    # 全局级字段（不需要 padding，直接堆叠）
    global_keys = [
        "target_tempo_dist",
        "target_mode_dist",
        "target_features",
        "target_style",
    ]
    for key in global_keys:
        stacked = torch.stack([sample[1][key] for sample in batch], dim=0)
        batched_target[key] = stacked

    return batched_input_tokens, batched_target


# ==============================================================================
# 七、独立测试入口（本地调试用，不做为模块主逻辑）
# ==============================================================================

if __name__ == "__main__":
    """
    本地调试运行示例：
        验证 Dataset 加载、数据格式、训练/验证集划分是否正确。
    """
    print("=" * 70)
    print("南音 Dataset 本地调试")
    print("=" * 70)

    # 测试1: 创建训练集
    print("\n>>> 测试1: 获取训练集 Dataset")
    train_ds = get_train_dataset()
    print(f"训练集大小: {len(train_ds)}")

    # 测试2: 创建验证集
    print("\n>>> 测试2: 获取验证集 Dataset")
    val_ds = get_val_dataset()
    print(f"验证集大小: {len(val_ds)}")

    # 测试3: 加载第一首曲目
    print("\n>>> 测试3: 加载第一首训练集曲目")
    try:
        input_tokens, target_dict = train_ds[0]
        print(f"曲目ID: {train_ds.get_song_id(0)}")
        print(f"input_tokens 形状: {input_tokens.shape},  dtype: {input_tokens.dtype}")
        print(f"target_tokens   形状: {target_dict['target_tokens'].shape},  dtype: {target_dict['target_tokens'].dtype}")
        print(f"target_pitch    形状: {target_dict['target_pitch'].shape},  dtype: {target_dict['target_pitch'].dtype}")
        print(f"target_duration 形状: {target_dict['target_duration'].shape},  dtype: {target_dict['target_duration'].dtype}")
        print(f"target_ornament 形状: {target_dict['target_ornament'].shape},  dtype: {target_dict['target_ornament'].dtype}")
        print(f"target_tempo_dist 形状: {target_dict['target_tempo_dist'].shape}")
        print(f"target_mode_dist  形状: {target_dict['target_mode_dist'].shape}")
        print(f"target_features   形状: {target_dict['target_features'].shape}")
        print(f"target_style      形状: {target_dict['target_style'].shape}")
        print(f"\n前10个 target_tokens: {target_dict['target_tokens'][:10].tolist()}")
        print(f"前10个 target_pitch:   {target_dict['target_pitch'][:10].tolist()}")
        print(f"前10个 target_duration: {target_dict['target_duration'][:10].tolist()}")
    except Exception as e:
        print(f"错误: {e}")

    # 测试4: 统计信息
    print("\n>>> 测试4: Dataset 统计信息")
    get_dataset_statistics(train_ds)
    get_dataset_statistics(val_ds)

    # 测试5: 验证 target_dict 字段与 loss_nanyin.py 入参兼容性
    print("\n>>> 测试5: 验证 target_dict 字段兼容性")
    required_keys = [
        "target_tokens",
        "target_pitch",
        "target_duration",
        "target_ornament",
        "target_tempo_dist",
        "target_mode_dist",
        "target_features",
        "target_style",
    ]
    _, sample_dict = train_ds[0]
    missing = [k for k in required_keys if k not in sample_dict]
    if missing:
        print(f"缺少字段: {missing}")
    else:
        print("所有 target_dict 字段齐全，与 loss_nanyin.py 兼容 [OK]")

    # 测试6: 测试 collate 函数
    print("\n>>> 测试6: 测试 collate_nanyin_batch")
    from torch.utils.data import DataLoader
    loader = DataLoader(train_ds, batch_size=2, collate_fn=collate_nanyin_batch)
    batch_input, batch_target = next(iter(loader))
    print(f"batch input_tokens 形状: {batch_input.shape}")
    for k, v in batch_target.items():
        print(f"  batch {k}: shape={v.shape}, dtype={v.dtype}")

    print("\n" + "=" * 70)
    print("调试完成，所有测试通过。")
    print("=" * 70)
