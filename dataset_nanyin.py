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

⚠ GT 数据噪声来源（E2 硬性要求，论文实验必须记录，勿删除本段）：
  训练/评估用的 GT = **BasicPitch 对南音 dapu 大谱音频的自动转录**
  （output/BasicPitch/*.mid），不是人工校对的谱面，存在**同音重叠音符**：
  转录会对同一音高产出相互重叠（或首尾相接）的多个音符。本数据集在编码阶段
  按 (start_time, pitch) 排序后逐音编码，时值保留转录原值；但在**写回 MIDI**
  时（generate_music.notes_to_midi）为保证音符数量不丢（规避 (channel, pitch)
  配对歧义导致丢音，历史 bug：1025 音符被解析成 792），会把**前一个音的时值截断
  到后一个音的起点**。
  ⇒ 该截断会**引入局部时值噪声**：这些音符的时值被人为缩短，且模型学到的时值
    分布会继承这部分偏差。论文实验须明确记录该噪声来源及其量级（见
    reports/E1_selftest_compound_events.md 与 reports/E2_window_training.md）。
"""

import os
import json
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import pretty_midi  # 解析 BasicPitch 转录 midi（训练数据源）
except ImportError:
    pretty_midi = None

from nanyin_config import load_config, get_section, E2_CONFIG_PATH


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
    # 排序键：(start_time, pitch)。同音重叠音符（BasicPitch 转录噪声）在此被相邻排列，
    # 编码阶段保留转录原时值；写回 MIDI 时前一音会被截断到后一音起点（噪声来源，
    # 见文件头"GT 数据噪声来源"注释，论文实验必须记录）。
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
# 三之二、复合事件编解码器（E1：pitch + duration + time-shift）
# ==============================================================================

class CompoundEventCodec:
    """音符级复合事件（pitch + duration + time-shift）编解码器。

    事件组织方式：**每个音符一个三元组**，序列长度 = 音符数

        第 i 步事件 = (pitch_id_i, duration_id_i, shift_id_i)

      - pitch_id    : MIDI 音高，直接作为类别 index（词表大小 128，见 yaml: codec.pitch_vocab_size）
      - duration_id : 该音符时值（秒）量化到 time_unit_sec 网格后的类别 index
      - shift_id    : 该音符 onset 到**下一个音符** onset 的间隔（time-shift，秒）类别 index
                      ⇒ onset_{i+1} = onset_i + shift_i，即**前向 IOI**（见下方"time-shift 方向约定"）

    duration / shift 的类别 index 约定（统一公式，见 yaml: codec.*_max_steps）：

        ==========================================================
         index                     含义
        ==========================================================
         0                         PAD / 未定义（伪标签位）
         s + 1                     s 个量化步长，秒数 = s * time_unit_sec，s < max_steps
         max_steps + 1             溢出档（>= max_steps * time_unit_sec）
        ==========================================================

        故类别数 = max_steps + 2。溢出档解码时只能还原为下界值 max_steps * time_unit_sec。

    time-shift 方向约定（重要，决定了哪个位置是伪标签）：
        采用**前向 IOI**：shift_i = onset_{i+1} - onset_i。
        由此保证"末尾位置的 time-shift 无真实监督目标"这一性质成立
        （最后一个音符之后没有下一个音符），且解码时先落音符再累加 Shift，
        天然与自回归生成的顺序一致：第 i 步采样出的 shift 决定第 i+1 个音的位置，
        不存在未来信息泄漏。

    ⚠ 已知局限（本阶段明确记录，后续阶段再处理，勿在此处悄悄绕过）：
      1) **序列末尾 time-shift 为伪标签**：最后一个音符之后不存在"下一个音符"，
         其 time-shift 没有真实监督目标，统一置 0（未定义位）。
         ⇒ 计算 time-shift 损失时必须对该位置置掩码，否则会引入错误的监督信号。
         ⇒ 窗口/切片切尾时同理：窗口末尾假想的"再下一个音符"并不存在，
            该 time-shift 一律为伪标签。
      2) **对长休止表达存在局限**：相邻 onset 间隔 >= max_steps * time_unit_sec
         （当前 yaml: 1000 * 0.02 = 20.00 s）时，全部被压缩到同一个溢出档，解码只能还原
         为 20.00 s 下界，无法表达确切休止长度；音符时值同理（>= 8.00 s 溢出）。
         ⇒ 一旦发生溢出，其后所有音符的 onset 会被永久性地向前平移（误差累积），
            具体误差分布见 reports/E1_selftest_compound_events.md。

    特殊 token（E2 起，用于窗口样本的 BOS / EOS 标记）：
        每路词表在真实类别之上追加 ``num_special_tokens``（当前 2）个 id：
            pitch    : BOS = pitch_vocab_size(128),      EOS = 129
            duration : BOS = duration_num_classes(402),  EOS = 403
            shift    : BOS = shift_num_classes(1002),    EOS = 1003
        三路词表**总大小** = 真实类别数 + 2（模型 embedding / 分类头必须使用总大小）。
        BOS / EOS 只是序列边界标记，不携带音高/时值语义；窗口样本中它们的位置
        由 ``build_window_sample()`` 给出。
    """

    def __init__(self, cfg: Optional[dict] = None):
        codec_cfg = dict(cfg if cfg is not None else get_section("codec"))
        self.time_unit_sec: float = float(codec_cfg["time_unit_sec"])
        self.pitch_vocab_size: int = int(codec_cfg["pitch_vocab_size"])
        self.pitch_min: int = int(codec_cfg["pitch_min"])
        self.pitch_max: int = int(codec_cfg["pitch_max"])
        self.duration_max_steps: int = int(codec_cfg["duration_max_steps"])
        self.shift_max_steps: int = int(codec_cfg["shift_max_steps"])
        self.max_notes: int = int(codec_cfg["max_notes"])
        self.num_special_tokens: int = int(codec_cfg.get("num_special_tokens", 0))

        # 0 号保留给 PAD / 未定义，末档为溢出档
        self.duration_num_classes: int = self.duration_max_steps + 2
        self.shift_num_classes: int = self.shift_max_steps + 2

        # ---- 特殊 token id（紧跟真实类别之后）----
        self.pitch_bos_id: int = self.pitch_vocab_size
        self.pitch_eos_id: int = self.pitch_vocab_size + 1
        self.duration_bos_id: int = self.duration_num_classes
        self.duration_eos_id: int = self.duration_num_classes + 1
        self.shift_bos_id: int = self.shift_num_classes
        self.shift_eos_id: int = self.shift_num_classes + 1
        self.pad_id: int = 0

        # ---- 词表总大小（含特殊 token），供模型 embedding / 分类头使用 ----
        self.pitch_vocab_total: int = self.pitch_vocab_size + self.num_special_tokens
        self.duration_vocab_total: int = self.duration_num_classes + self.num_special_tokens
        self.shift_vocab_total: int = self.shift_num_classes + self.num_special_tokens

    # ---------- 量化 ↔ 秒(或步) ----------

    def seconds_to_steps(self, seconds: float) -> int:
        """秒 → 量化步数（四舍五入到最近网格，负值截断为 0）。"""
        return max(0, int(round(float(seconds) / self.time_unit_sec)))

    def steps_to_seconds(self, steps: int) -> float:
        """量化步数 → 秒。"""
        return float(steps) * self.time_unit_sec

    def _class_from_steps(self, steps: int, max_steps: int) -> int:
        """量化步数 → 类别 index（0=PAD；1..max_steps 为精确档；max_steps+1=溢出）。"""
        steps = max(0, int(steps))
        if steps >= max_steps:
            return max_steps + 1          # 溢出档
        return steps + 1

    def _value_to_class(self, seconds: float, max_steps: int) -> int:
        """秒 → 类别 index（0=PAD；1..max_steps 为 s+1 映射；max_steps+1=溢出）。"""
        return self._class_from_steps(self.seconds_to_steps(seconds), max_steps)

    def _class_to_steps(self, cls: int, max_steps: int) -> int:
        """类别 index → 量化步数（溢出档还原为下界，PAD 还原为 0）。"""
        if cls <= 0:
            return 0
        return int(min(cls - 1, max_steps))

    def duration_to_index(self, duration_sec: float) -> int:
        """时值（秒）→ duration 类别 index。"""
        return self._value_to_class(duration_sec, self.duration_max_steps)

    def index_to_duration(self, idx: int) -> float:
        """duration 类别 index → 时值（秒）。"""
        return self.steps_to_seconds(self._class_to_steps(int(idx), self.duration_max_steps))

    def shift_to_index(self, shift_sec: float) -> int:
        """time-shift（秒）→ shift 类别 index。0 秒的气口/同时发音 = index 1。"""
        return self._value_to_class(shift_sec, self.shift_max_steps)

    def index_to_shift(self, idx: int) -> float:
        """shift 类别 index → time-shift（秒）。"""
        return self.steps_to_seconds(self._class_to_steps(int(idx), self.shift_max_steps))

    # ---------- 序列级编解码 ----------

    def encode_notes(self, notes: List[dict]) -> Dict[str, Any]:
        """音符列表 → 三路复合事件 ID 序列。

        参数:
            notes: [{"note": int, "duration": float, "start_time": float, "velocity": int}, ...]
                   需按 (start_time, pitch) 升序（load_song_annotations 已保证）
        返回:
            {
              "pitch_ids":    List[int],   # 长度 = 音符数
              "duration_ids": List[int],
              "shift_ids":    List[int],   # 前向 IOI：shift_i = onset_{i+1} - onset_i；末音符 = 0（伪标签）
              "stats": {...}               # 溢出等统计，便于自测报告量化局限
            }
        """
        pitch_ids: List[int] = []
        duration_ids: List[int] = []
        shift_ids: List[int] = []

        truncated = 0
        duration_overflow = 0
        shift_overflow = 0
        negative_gap = 0

        use_notes = notes[: self.max_notes]
        truncated = len(notes) - len(use_notes)

        # 前向 IOI：先算出每个音符量化后的**绝对 onset**，再按 shift_i = q_{i+1} - q_i 作差。
        #   这样做的原因：若对每个原始间隔独立量化再分类，各处 ±半格的舍入误差会在解码
        #   累加时形成单向漂移（实测千音符级曲目的累计 onset 误差可达数秒）；用"量化后
        #   绝对位置作差"可让每个音符的 onset 误差被限制在半格以内，且不随序列长度累积。
        quant_starts = [
            self.seconds_to_steps(float(n.get("start_time", 0.0))) for n in use_notes
        ]

        for i, note in enumerate(use_notes):
            pitch = int(np.clip(int(note.get("note", 60)), self.pitch_min, self.pitch_max))
            duration = float(note.get("duration", 0.5))

            if i == len(use_notes) - 1:
                # ⚠ 伪标签：末尾音符之后不存在"下一个音符"，time-shift 无真实监督目标，
                #   统一置 0（未定义位）。训练时必须对该位置做掩码，不计入损失。
                shift_index = 0
            else:
                gap_steps = quant_starts[i + 1] - quant_starts[i]
                if gap_steps < 0:
                    # 多轨合并/音符重叠可能产生负间隔，截断为 0 并计数
                    negative_gap += 1
                    gap_steps = 0
                # 用"步数"而非"秒"分类，避免二次量化误差
                shift_index = self._class_from_steps(gap_steps, self.shift_max_steps)

            duration_index = self.duration_to_index(duration)
            if duration_index == self.duration_max_steps + 1:
                duration_overflow += 1
            if shift_index == self.shift_max_steps + 1:
                shift_overflow += 1

            pitch_ids.append(pitch)
            duration_ids.append(duration_index)
            shift_ids.append(shift_index)

        stats = {
            "num_notes": len(pitch_ids),
            "truncated_notes": truncated,
            "duration_overflow": duration_overflow,
            "shift_overflow": shift_overflow,
            "negative_gap_clamped": negative_gap,
        }

        return {
            "pitch_ids": pitch_ids,
            "duration_ids": duration_ids,
            "shift_ids": shift_ids,
            "stats": stats,
        }

    def decode_events(
        self,
        pitch_ids: List[int],
        duration_ids: List[int],
        shift_ids: List[int],
        start_offset_sec: float = 0.0,
    ) -> List[dict]:
        """复合事件序列 → 音符列表（onset 由 time-shift 累加得到）。

        onset 递推（前向 IOI 约定）：
            onset_0    = start_offset_sec
            onset_{i+1} = onset_i + index_to_shift(shift_ids[i])
        即**先落当前音符，再用当前位置的 shift 推进到下一个音符**。
        末尾 shift_ids[-1] 为伪标签 0，不参与 onset 计算（最后一个音符已经落地）。

        参数:
            pitch_ids / duration_ids / shift_ids: 三路等长 ID 序列
            start_offset_sec: 首个音符的 onset（秒），默认 0
        返回:
            [{"note": int, "duration": float, "start_time": float}, ...]
        """
        n = min(len(pitch_ids), len(duration_ids), len(shift_ids))
        decoded: List[dict] = []
        cur_start = float(start_offset_sec)
        for i in range(n):
            decoded.append({
                "note": int(np.clip(int(pitch_ids[i]), self.pitch_min, self.pitch_max)),
                "duration": self.index_to_duration(int(duration_ids[i])),
                "start_time": cur_start,
            })
            cur_start += self.index_to_shift(int(shift_ids[i]))
        return decoded


_DEFAULT_CODEC: Optional["CompoundEventCodec"] = None


def get_compound_codec(cfg: Optional[dict] = None, reload: bool = False) -> CompoundEventCodec:
    """获取默认复合事件编解码器（参数来自 configs/e1_compound_events.yaml）。

    参数 cfg 既可以是 **codec 小节**（如 {"time_unit_sec": ...}），
    也可以是**完整配置字典**（含 "codec" 键，内部自动取小节）。
    """
    global _DEFAULT_CODEC
    if isinstance(cfg, dict) and "codec" in cfg:
        cfg = cfg["codec"]
    if reload or _DEFAULT_CODEC is None or cfg is not None:
        codec = CompoundEventCodec(cfg)
        if cfg is None:
            _DEFAULT_CODEC = codec
        return codec
    return _DEFAULT_CODEC


# ==============================================================================
# 三之三、E2：复合事件序列的窗口切分（256 事件 / 窗口）
# ==============================================================================
#
# 设计要点：
#   1. 窗口长度单位是**事件数**（= 复合序列长度），不是音符时间长度；
#   2. 曲目 ID 隔离：窗口严格在单曲内部切，训练集 / 验证集曲目 ID 不相交，
#      同曲切出的窗口不会跨集合；
#   3. 训练集步长 128（50% 重叠），验证/测试集步长 = 窗口长度（非重叠）；
#   4. 不足窗口长度的尾段**直接丢弃**，不做 padding；
#   5. 每个窗口前后加 BOS / EOS，输入 = [BOS] + 窗口事件，目标 = 窗口事件 + [EOS]；
#   6. **窗口最后一个事件的 time-shift 是伪标签**：window 尾部假想的"再下一个音符"
#      不存在（生成时该事件之后就是 EOS），因此统一置 0，并在 target 中用
#      IGNORE_INDEX 屏蔽，禁止参与反向传播（E1 自测报告硬性约束）。
#
# ⚠ 该掩码的代价（如实记录）：对于**非末尾窗口**（窗口中段切出的窗口），
#   最后一个事件到下一个音符的真实 time-shift 在整曲序列中是可知的，
#   本实现按 E1 约定一律视为未定义并屏蔽 → 每个窗口舍弃 1 个真实可得的
#   shift 监督样本。选择一致处理而非"按位置区别对待"，是为了让模型在窗口
#   尾部学到的语义始终是"该停了"，与自回归生成时"第 256 个事件后接 EOS"一致。
# ==============================================================================

# PyTorch CrossEntropyLoss 默认 ignore_index：标记"该位置不参与损失"
IGNORE_INDEX: int = -100


def encode_song_events(
    song_id: str,
    base_dir: Optional[str] = None,
    codec: Optional[CompoundEventCodec] = None,
) -> Optional[Dict[str, Any]]:
    """整曲 → 复合事件 ID 序列（窗口切分的输入）。

    参数:
        song_id:  曲目 ID
        base_dir: 数据根目录，默认 _DATASET_BASE
        codec:    编解码器，默认 get_compound_codec()
    返回:
        {"song_id", "pitch_ids", "duration_ids", "shift_ids", "num_events", "stats"}；
        加载失败返回 None
    """
    codec = codec if codec is not None else get_compound_codec()
    annotations = load_song_annotations(song_id, base_dir if base_dir is not None else _DATASET_BASE)
    if annotations is None:
        return None

    notes = annotations["midi"]["tracks"][0]["notes"]
    notes = sorted(notes, key=lambda x: (float(x["start_time"]), int(x["note"])))
    events = codec.encode_notes(notes)
    return {
        "song_id": song_id,
        "pitch_ids": events["pitch_ids"],
        "duration_ids": events["duration_ids"],
        "shift_ids": events["shift_ids"],
        "num_events": len(events["pitch_ids"]),
        "stats": events["stats"],
    }


def slice_event_windows(
    num_events: int, window_size: int, stride: int
) -> List[Tuple[int, int]]:
    """按 [start, end) 切出等长窗口；不足 window_size 的尾段直接丢弃（不 padding）。

    参数:
        num_events:  序列事件总数
        window_size: 窗口长度（事件数）
        stride:      滑动步长；stride == window_size 时为非重叠取样
    返回:
        [(start, end), ...]，每个窗口长度恒为 window_size
    """
    if num_events < window_size or window_size <= 0 or stride <= 0:
        return []
    return [(s, s + window_size) for s in range(0, num_events - window_size + 1, stride)]


def build_window_sample(
    song_events: Dict[str, Any],
    start: int,
    end: int,
    codec: Optional[CompoundEventCodec] = None,
) -> Dict[str, torch.Tensor]:
    """由整曲事件序列切出一个窗口样本（含 BOS / EOS、attention_mask、ignore_index 掩码）。

    序列布局（长度 = window_size + 1）：
        输入 : [BOS] + e_0 ... e_{W-1}
        目标 : e_0 ... e_{W-1} + [EOS]
    time-shift 掩码：
        目标序列中**最后一个事件位**与 **EOS 位**的 shift 均置 IGNORE_INDEX(-100)，
        因为窗口最后一个事件之后不存在"下一个音符"（伪标签），EOS 也不是音符事件。

    参数:
        song_events: encode_song_events() 的返回
        start/end:   窗口在整曲事件序列上的 [start, end) 区间
        codec:       编解码器
    返回:
        样本字典（全部为 torch.long 张量）
    """
    codec = codec if codec is not None else get_compound_codec()

    pitch_ids = list(song_events["pitch_ids"][start:end])
    duration_ids = list(song_events["duration_ids"][start:end])
    shift_ids = list(song_events["shift_ids"][start:end])

    # ---- 窗口尾部 time-shift = 伪标签：置 0（未定义），训练阶段用 IGNORE_INDEX 屏蔽 ----
    # （整曲编码时末尾 shift 已经是 0；此处对窗口切尾统一处理，保证任意窗口都成立）
    shift_ids[-1] = codec.pad_id

    seq_len = len(pitch_ids) + 1     # +1 = BOS（输入侧）/ EOS（目标侧）

    input_pitch_ids = torch.tensor([codec.pitch_bos_id] + pitch_ids, dtype=torch.long)
    input_duration_ids = torch.tensor([codec.duration_bos_id] + duration_ids, dtype=torch.long)
    input_shift_ids = torch.tensor([codec.shift_bos_id] + shift_ids, dtype=torch.long)

    target_pitch_ids = torch.tensor(pitch_ids + [codec.pitch_eos_id], dtype=torch.long)
    target_duration_ids = torch.tensor(duration_ids + [codec.duration_eos_id], dtype=torch.long)

    # shift 目标：最后一个事件位 + EOS 位 → IGNORE_INDEX
    shift_target = list(shift_ids)
    shift_target[-1] = IGNORE_INDEX
    shift_target.append(IGNORE_INDEX)
    target_shift_ids = torch.tensor(shift_target, dtype=torch.long)

    attention_mask = torch.ones(seq_len, dtype=torch.long)

    return {
        "input_pitch_ids": input_pitch_ids,
        "input_duration_ids": input_duration_ids,
        "input_shift_ids": input_shift_ids,
        "target_pitch_ids": target_pitch_ids,
        "target_duration_ids": target_duration_ids,
        "target_shift_ids": target_shift_ids,
        "attention_mask": attention_mask,
    }


class CompoundEventWindowDataset(Dataset):
    """E2 复合事件窗口数据集：样本 = 一个固定长度（默认 256 事件）的窗口。

    与旧 NanyinDataset（每首曲目一个样本）的区别：
      - 样本粒度 = 窗口，不再是整曲；
      - 每个样本自带 BOS/EOS、attention_mask 与 time-shift 伪标签掩码；
      - 不再提供 target_tempo_dist / target_mode_dist 等全局级字段
        （E2 只做三路分类 CE，全局级真值待 E3 需要时再接入）。

    __getitem__ 返回:
        {
          "input_*_ids"/"target_*_ids"/"attention_mask": torch.long 张量,
          "song_id": str, "split": str, "start": int, "end": int, "window_index": int
        }
    """

    def __init__(
        self,
        song_ids: List[str],
        split: str,
        cfg: Optional[dict] = None,
        base_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        参数:
            song_ids: 曲目 ID 列表（训练集与验证集必须不相交 → 曲目 ID 隔离）
            split:    "train" | "val"（决定滑动步长）
            cfg:      配置字典，默认加载 configs/e2_compound_window.yaml
            base_dir: 数据根目录
            verbose:  是否打印窗口统计与截断曲目清单
        """
        super().__init__()
        self.cfg = cfg if cfg is not None else load_config(E2_CONFIG_PATH)
        self.window_cfg = self.cfg["window"]
        self.window_size: int = int(self.window_cfg["window_size_events"])
        self.drop_tail: bool = bool(self.window_cfg["drop_tail"])
        if split == "train":
            self.stride: int = int(self.window_cfg["train_stride_events"])
        elif split in ("val", "test", "eval"):
            self.stride = int(self.window_cfg["eval_stride_events"])
        else:
            raise ValueError(f"[dataset_nanyin] 未知 split: {split}（应为 train/val）")

        self.split = split
        self.song_ids = list(song_ids)
        self.base_dir = base_dir if base_dir is not None else _DATASET_BASE
        self.codec = get_compound_codec(self.cfg)

        # 窗口索引表 + 整曲事件缓存（11 首曲目，一次性编码，避免每 sample 重复读 MIDI）
        self.records: List[Dict[str, Any]] = []
        self._song_events: Dict[str, Dict[str, Any]] = {}
        self._song_stats_rows: List[Dict[str, Any]] = []

        for song_id in self.song_ids:
            song_events = encode_song_events(song_id, self.base_dir, self.codec)
            if song_events is None:
                print(f"[dataset_nanyin] 窗口构建跳过（GT 加载失败）: {song_id}")
                continue
            self._song_events[song_id] = song_events

            num_events = int(song_events["num_events"])
            starts = slice_event_windows(num_events, self.window_size, self.stride)
            covered_tail = starts[-1][1] if starts else 0
            discarded_tail = num_events - covered_tail      # 未被任何窗口覆盖的尾部事件数

            for w_idx, (s, e) in enumerate(starts):
                self.records.append({
                    "song_id": song_id,
                    "split": self.split,
                    "start": int(s),
                    "end": int(e),
                    "window_index": int(w_idx),
                })

            encode_stats = song_events["stats"]
            self._song_stats_rows.append({
                "song_id": song_id,
                "split": self.split,
                "num_events": num_events,
                "truncated_notes": int(encode_stats["truncated_notes"]),
                "duration_overflow": int(encode_stats["duration_overflow"]),
                "shift_overflow": int(encode_stats["shift_overflow"]),
                "negative_gap_clamped": int(encode_stats["negative_gap_clamped"]),
                "num_windows": len(starts),
                "discarded_tail_events": int(discarded_tail),
            })

        if verbose:
            self.print_statistics()

    # ---------- 基本接口 ----------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        song_events = self._song_events[rec["song_id"]]
        sample = build_window_sample(
            song_events, rec["start"], rec["end"], codec=self.codec
        )
        sample.update({
            "song_id": rec["song_id"],
            "split": rec["split"],
            "start": int(rec["start"]),
            "end": int(rec["end"]),
            "window_index": int(rec["window_index"]),
        })
        return sample

    # ---------- 统计 / 台账 ----------

    def song_stats_rows(self) -> List[Dict[str, Any]]:
        """逐曲统计行（用于窗口数量统计清单 CSV）。"""
        return list(self._song_stats_rows)

    def truncated_songs(self) -> List[Dict[str, Any]]:
        """被 codec.max_notes 截断的曲目清单（E2 硬性要求：必须如实打印记录）。"""
        return [r for r in self._song_stats_rows if r["truncated_notes"] > 0]

    def window_size_events(self) -> int:
        return self.window_size

    def print_statistics(self) -> None:
        """打印窗口切分统计（含被 max_notes 截断的曲目清单）。"""
        total_windows = len(self.records)
        total_events = sum(r["num_events"] for r in self._song_stats_rows)
        total_disc = sum(r["discarded_tail_events"] for r in self._song_stats_rows)
        print(f"[dataset_nanyin] ===== 复合事件窗口统计 [{self.split}] =====")
        print(f"  窗口长度={self.window_size} 事件, 步长={self.stride} "
              f"({'非重叠' if self.stride >= self.window_size else '滑动重叠'}), drop_tail={self.drop_tail}")
        print(f"  曲目数={len(self._song_stats_rows)}, 窗口数={total_windows}, "
              f"事件总数={total_events}, 尾部丢弃事件数={total_disc}")
        print(f"  {'曲目':<28}{'事件数':>8}{'窗口数':>8}{'尾丢弃':>8}{'max_notes截断':>14}")
        for r in self._song_stats_rows:
            print(f"  {r['song_id']:<28}{r['num_events']:>8}{r['num_windows']:>8}"
                  f"{r['discarded_tail_events']:>8}{r['truncated_notes']:>14}")
        truncated = self.truncated_songs()
        if truncated:
            print(f"  ⚠ 被 max_notes={self.codec.max_notes} 截断的曲目清单（共 {len(truncated)} 首）：")
            for r in truncated:
                print(f"      - {r['song_id']}: 丢弃 {r['truncated_notes']} 个音符 "
                      f"（保留 {r['num_events']}）")
        else:
            print(f"  未被 max_notes={self.codec.max_notes} 截断的曲目（0 首截断）")


def get_compound_window_datasets(
    cfg: Optional[dict] = None,
    base_dir: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[CompoundEventWindowDataset, CompoundEventWindowDataset]:
    """构建 E2 训练集 / 验证集窗口数据集（曲目 ID 固定划分，与 E0~exp3 一致）。"""
    cfg = cfg if cfg is not None else load_config(E2_CONFIG_PATH)
    train_ds = CompoundEventWindowDataset(
        song_ids=TRAIN_SPLIT_IDS, split="train", cfg=cfg, base_dir=base_dir, verbose=verbose
    )
    val_ds = CompoundEventWindowDataset(
        song_ids=VAL_SPLIT_IDS, split="val", cfg=cfg, base_dir=base_dir, verbose=verbose
    )

    # 曲目 ID 隔离自检（防止划分被误改导致同曲窗口跨集）
    overlap = set(TRAIN_SPLIT_IDS) & set(VAL_SPLIT_IDS)
    if overlap:
        raise RuntimeError(f"[dataset_nanyin] 训练/验证曲目 ID 存在交集，违反曲目隔离: {overlap}")
    return train_ds, val_ds


def collate_compound_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """E2 窗口样本 collate：张量堆叠，字符串/标量元数据收集为 list。"""
    tensor_keys = [
        "input_pitch_ids", "input_duration_ids", "input_shift_ids",
        "target_pitch_ids", "target_duration_ids", "target_shift_ids",
        "attention_mask",
    ]
    meta_keys = ["song_id", "split", "start", "end", "window_index"]
    collated: Dict[str, Any] = {}
    for key in tensor_keys:
        collated[key] = torch.stack([b[key] for b in batch], dim=0)
    for key in meta_keys:
        collated[key] = [b[key] for b in batch]
    return collated


def compute_vocab_class_stats(
    dataset: CompoundEventWindowDataset,
    top_k: int = 10,
    long_tail_threshold: int = 5,
) -> Dict[str, Any]:
    """统计 duration / time-shift 词表各类别的样本出现频次（监控长尾类别）。

    统计口径：**只统计真正参与损失的监督位置**
      - duration：目标序列全部位置（含 EOS 位）；
      - shift  ：跳过 IGNORE_INDEX 位置（窗口尾伪标签 + EOS）。

    参数:
        dataset: 窗口数据集
        top_k:   打印频次最高的 K 个类别
        long_tail_threshold: 频次 <= 该值计为长尾类别
    返回:
        {"num_windows", "num_events", "duration": {...}, "shift": {...}}
    """
    codec = dataset.codec
    duration_counts = np.zeros(codec.duration_vocab_total, dtype=np.int64)
    shift_counts = np.zeros(codec.shift_vocab_total, dtype=np.int64)
    num_events = 0

    for i in range(len(dataset)):
        sample = dataset[i]
        tgt_dur = sample["target_duration_ids"].numpy()
        tgt_shift = sample["target_shift_ids"].numpy()
        duration_counts += np.bincount(tgt_dur, minlength=codec.duration_vocab_total)
        valid_shift = tgt_shift[tgt_shift != IGNORE_INDEX]
        shift_counts += np.bincount(valid_shift, minlength=codec.shift_vocab_total)
        num_events += int(sample["attention_mask"].shape[0]) - 1   # 窗口事件数（去掉 BOS/EOS 侧）

    def _summarize(counts: np.ndarray) -> Dict[str, Any]:
        total = int(counts.sum())
        used = int((counts > 0).sum())
        zero = int((counts == 0).sum())
        tail_mask = (counts > 0) & (counts <= long_tail_threshold)
        tail_classes = int(tail_mask.sum())
        tail_tokens = int(counts[tail_mask].sum())
        top_idx = np.argsort(-counts)[:top_k]
        top = [{"class_id": int(i), "count": int(counts[i])} for i in top_idx if counts[i] > 0]
        return {
            "num_classes": int(counts.shape[0]),
            "used_classes": used,
            "zero_count_classes": zero,
            "total_tokens": total,
            "long_tail_threshold": int(long_tail_threshold),
            "long_tail_classes": tail_classes,
            "long_tail_token_share": float(tail_tokens / total) if total else 0.0,
            "top_classes": top,
            "counts": [int(c) for c in counts],
        }

    return {
        "num_windows": len(dataset),
        "num_events": int(num_events),
        "duration": _summarize(duration_counts),
        "shift": _summarize(shift_counts),
    }


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
    token_list: List[int] = []         # token ID序列（旧：仅 pitch 单 token）
    note_items: List[dict] = []        # 原始音符 dict 列表（供复合事件编码器使用）

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
            note_items.append({
                "note": int(np.clip(pitch, 0, 127)),
                "duration": float(duration),
                "start_time": float(start),
            })

    # 若MIDI轨道无音符，构造一个占位序列（至少1个元素以保证模型不崩溃）
    if len(token_list) == 0:
        print("[dataset_nanyin] 警告：MIDI轨道中未找到音符数据，使用占位token")
        token_list.append(0)
        pitch_list.append(60.0)
        duration_list.append(1.0)
        start_list.append(0.0)
        note_items.append({"note": 60, "duration": 1.0, "start_time": 0.0})

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

    # ---- 6.5 复合事件（E1）：(pitch, duration, time-shift) 三路类别序列 ----
    #   每音符一个三元组，长度与 token 序列一致；target_tokens 仅为兼容旧链路保留。
    codec = get_compound_codec()
    events = codec.encode_notes(note_items)
    target_pitch_ids = torch.tensor(events["pitch_ids"], dtype=torch.long)      # (seq_len,)
    target_duration_ids = torch.tensor(events["duration_ids"], dtype=torch.long)  # (seq_len,)
    target_shift_ids = torch.tensor(events["shift_ids"], dtype=torch.long)      # (seq_len,)

    # ---- 7. 组装完整 target_dict ----
    target_dict = {
        # 序列级（逐音符）真值
        "target_tokens":   target_tokens,
        "target_pitch":    target_pitch,
        "target_duration": target_duration,
        "target_ornament": target_ornament,
        # 复合事件三路类别 ID（E1 新增；末尾 shift 为伪标签，后续训练需掩码）
        "target_pitch_ids":    target_pitch_ids,
        "target_duration_ids": target_duration_ids,
        "target_shift_ids":    target_shift_ids,
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
        seq_keys = [
            "target_tokens", "target_pitch", "target_duration", "target_ornament",
            "target_pitch_ids", "target_duration_ids", "target_shift_ids",
        ]
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
    # dtype：类别 ID 一律 long，连续值（音高/时值/装饰音）一律 float32
    seq_keys = [
        "input_tokens",    # 输入token（旧：仅 pitch 单 token）
        "target_tokens",   # 目标token
        "target_pitch",    # 音高（连续值）
        "target_duration", # 时值（连续值）
        "target_ornament", # 装饰音标记
        "target_pitch_ids",     # 复合事件：pitch 类别 ID
        "target_duration_ids",  # 复合事件：duration 类别 ID
        "target_shift_ids",     # 复合事件：time-shift 类别 ID
    ]
    long_keys = {
        "input_tokens", "target_tokens",
        "target_pitch_ids", "target_duration_ids", "target_shift_ids",
    }

    batched_target: Dict[str, torch.Tensor] = {}
    batched_input_tokens = torch.empty(0)  # 初始化，避免未绑定警告

    for key in seq_keys:
        is_key_input = (key == "input_tokens")
        padded = torch.zeros(batch_size, max_len, dtype=torch.long if key in long_keys else torch.float32)
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
