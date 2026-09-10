# -*- coding: utf-8 -*-
"""
train_boyatcn.py - 南音BoYaTCN模型训练脚本
============================================

功能概述：
  1. 对接 dataset_nanyin.py（数据集）与 loss_nanyin.py（复合损失），
     加载固定划分的训练集（15首）与验证集（3首）；
  2. 前向传播计算 NOEF 复合 TotalLoss，执行反向传播更新网络参数；
  3. 每个epoch结束后，在验证集批量计算全套NOEF评估指标；
  4. 保留print输出epoch编号、各类分项损失、全部指标得分，
     所有数据同步追加写入本地CSV文件 ./experiment_record_boyatcn.csv；
  5. 权重保存：验证集综合NOEF得分最高时，模型存入 ./checkpoints/boyatcn/，
     支持断点续训自动加载；
  6. 分层动态权重调度：
       epoch 0~49  : 所有 λ=0.1（弱约束预热期）
       epoch 50~149: λ 线性上调至目标值（过渡期）
       epoch 150+  : λ 固定为最优权重（稳定收敛期）
     默认最优λ: λ1=1.5, λ2=1.0, λ3=1.0, λ4=1.5, λ5=0.5
     权重搜索候选区间: {0.1, 0.5, 1.0, 2.0, 5.0}

参考文献：
  MelodyGLM         https://arxiv.org/pdf/2309.10738v1
  Controllable Symbolic Music Generation https://www.preprints.org/manuscript/202604.0984
  TCSinger           https://arxiv.org/pdf/2409.15977
  NanyinHGNN         arXiv:2510.03617
  KAD                arXiv:2502.15602
"""

import os
import sys
import csv
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---- 导入自定义模块 ----
from models import NanyinBoYaTCN
from loss_nanyin import nanyin_total_loss
from loss_nanyin import (
    pitch_accuracy_MSE,
    pitch_stability_L1,
    rhythm_consistency_MSE,
    tempo_kl,
    ornament_score_MSE,
    ornament_density_MSE,
    mode_kl,
    fad_diff,
    style_cosine_loss,
    NANYIN_FOUR_MODES,
)
from dataset_nanyin import (
    get_train_dataset,
    get_val_dataset,
    collate_nanyin_batch,
)
from nanyin_metrics import NOEF_CATEGORY_WEIGHTS


# ==============================================================================
# 辅助：批量计算训练集各分项损失（用于统计tracking，非反向传播用）
# ==============================================================================

@torch.no_grad()
def compute_component_losses(preds: dict, target_dict: dict) -> Dict[str, float]:
    """
    在no_grad下计算各维度独立损失，用于epoch统计打印和CSV记录。

    参数:
        preds:       模型预测输出字典
        target_dict: 真值标注字典
    返回:
        各分项损失标量值的字典
    """
    # 序列分类交叉熵
    pred_flat = preds["pred_tokens"].reshape(-1, preds["pred_tokens"].shape[-1])
    tgt_flat = target_dict["target_tokens"].reshape(-1)
    ce_loss = F.cross_entropy(pred_flat, tgt_flat).item()

    # 音高损失
    p_mse = pitch_accuracy_MSE(preds["pred_pitch"], target_dict["target_pitch"]).item()
    p_stab = pitch_stability_L1(preds["pred_pitch"], target_dict["target_pitch"]).item()
    pitch_loss = p_mse + p_stab

    # 节奏损失
    r_mse = rhythm_consistency_MSE(preds["pred_duration"], target_dict["target_duration"]).item()
    t_kl = tempo_kl(preds["pred_tempo_dist"], target_dict["target_tempo_dist"]).item()
    rhythm_loss = r_mse + t_kl

    # 装饰音损失
    o_score = ornament_score_MSE(
        preds["pred_ornament"], target_dict["target_ornament"],
        preds["pred_pitch"], target_dict["target_pitch"],
        preds["pred_duration"], target_dict["target_duration"],
    ).item()
    o_density = ornament_density_MSE(preds["pred_ornament"], target_dict["target_ornament"]).item()
    ornament_loss = o_score + o_density

    # 调式损失
    m_kl = mode_kl(preds["pred_mode_logits"], target_dict["target_mode_dist"]).item()
    mode_loss = m_kl

    # 风格损失
    f_val = fad_diff(preds["pred_features"], target_dict["target_features"]).item()
    s_cos = style_cosine_loss(preds["pred_style"], target_dict["target_style"]).item()
    style_loss = f_val + s_cos

    return {
        "ce": ce_loss,
        "pitch": pitch_loss,
        "rhythm": rhythm_loss,
        "ornament": ornament_loss,
        "mode": mode_loss,
        "style": style_loss,
    }

# ==============================================================================
# 一、路径与超参配置
# ==============================================================================

# 当前脚本目录
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# CSV 日志路径
CSV_LOG_PATH = os.path.join(_SCRIPT_DIR, "experiment_record_boyatcn.csv")

# 模型权重保存路径
CHECKPOINT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints", "boyatcn")

# ---- 训练超参 ----
NUM_EPOCHS = 200                 # 总训练轮次（λ权重需150轮到位）
BATCH_SIZE = 1                   # 批大小（max_seq_len=4096时减为1以避免OOM）
LEARNING_RATE = 1e-3             # 初始学习率
WEIGHT_DECAY = 1e-5              # 正则化系数
EARLY_STOP_PATIENCE = 60         # 早停耐心：验证得分连续N轮不提升则终止

# ---- 最优 λ 权重（epoch 150+ 固定值） ----
# 搜索候选区间: {0.1, 0.5, 1.0, 2.0, 5.0}
FINAL_LAMBDA1 = 1.5   # 音高损失权重
FINAL_LAMBDA2 = 1.0   # 节奏损失权重
FINAL_LAMBDA3 = 1.0   # 装饰音损失权重
FINAL_LAMBDA4 = 1.5   # 调式损失权重
FINAL_LAMBDA5 = 0.5   # 风格损失权重

# ---- 动态权重调度节点 ----
WARMUP_EPOCHS = 50     # 预热期：λ 全部=0.1
RAMP_EPOCHS = 100      # 过渡期：线性上调，从 epoch 50 到 150
FIXED_EPOCH = WARMUP_EPOCHS + RAMP_EPOCHS  # epoch 150 起固定

# ---- NOEF 五大类指标权重（与 nanyin_metrics.py 对齐） ----
NOEF_WEIGHTS: Dict[str, float] = {
    "pitch":    0.25,
    "rhythm":   0.20,
    "ornament": 0.25,
    "mode":     0.20,
    "style":    0.10,
}

# ---- 模型结构参数 ----
VOCAB_SIZE = 128          # MIDI token 词表大小（0~127）
D_MODEL = 256             # 隐藏维度
NUM_HEADS = 8             # 注意力头数


# ==============================================================================
# 二、分层动态权重调度
# ==============================================================================

def get_dynamic_lambdas(epoch: int,
                       finals: Optional[Tuple[float, float, float, float, float]] = None
                       ) -> Tuple[float, float, float, float, float]:
    """
    根据当前epoch计算分层动态λ权重。

    调度策略：
      - epoch 0 ~ WARMUP_EPOCHS-1 (0~49): 全部λ = 0.1（弱约束，让模型先学token分类）
      - epoch WARMUP_EPOCHS ~ FIXED_EPOCH-1 (50~149): λ 从0.1线性上调至FINAL值
      - epoch >= FIXED_EPOCH (150+): λ 固定为FINAL值

    参数:
        epoch:  当前训练轮次
        finals: 外部传入的最终λ五元组，若为None则使用模块默认值

    文献来源：
      TCSinger (https://arxiv.org/pdf/2409.15977)
      该文在歌唱合成训练中采用渐进式权重调度策略，前期弱化辅助损失，
      逐步增大约束权重以避免训练初期的不稳定。
    """
    if finals is None:
        finals = (FINAL_LAMBDA1, FINAL_LAMBDA2, FINAL_LAMBDA3, FINAL_LAMBDA4, FINAL_LAMBDA5)
    f1, f2, f3, f4, f5 = finals

    if epoch < WARMUP_EPOCHS:
        # ---- 预热期：全部弱约束 ----
        return (0.1, 0.1, 0.1, 0.1, 0.1)

    elif epoch < FIXED_EPOCH:
        # ---- 过渡期：线性上调 ----
        progress = (epoch - WARMUP_EPOCHS) / RAMP_EPOCHS  # [0, 1)
        lam1 = 0.1 + (f1 - 0.1) * progress
        lam2 = 0.1 + (f2 - 0.1) * progress
        lam3 = 0.1 + (f3 - 0.1) * progress
        lam4 = 0.1 + (f4 - 0.1) * progress
        lam5 = 0.1 + (f5 - 0.1) * progress
        return (lam1, lam2, lam3, lam4, lam5)

    else:
        # ---- 稳定期：固定最优权重 ----
        return finals


# ==============================================================================
# 三、验证集 NOEF 指标计算
# ==============================================================================

@torch.no_grad()
def validate_noef(
    model: NanyinBoYaTCN,
    val_loader: DataLoader,
    device: torch.device,
    lambdas: Tuple[float, ...],
) -> Dict[str, float]:
    """
    在验证集上批量计算全套 NOEF 评估指标。

    指标计算基于 loss_nanyin.py 中各子损失函数的"可微近似"版本，
    将损失值转化为 [0, 1] 范围内的得分（越高越好）。

    NOEF综合得分加权公式：
      综合Score = 0.25×音高分 + 0.20×节奏分 + 0.25×装饰音分 + 0.20×调式分 + 0.10×整体风格分

    返回:
        包含全部 NOEF 指标得分和综合得分的字典
    """
    model.eval()

    # 累积统计量
    total_samples = 0
    total_ce_loss = 0.0
    total_pitch_mse = 0.0
    total_pitch_stab = 0.0
    total_rhythm_mse = 0.0
    total_tempo_kl = 0.0
    total_orn_score = 0.0
    total_orn_density = 0.0
    total_mode_kl = 0.0
    total_fad = 0.0
    total_style_cos = 0.0

    for input_tokens, target_dict in val_loader:
        input_tokens = input_tokens.to(device)

        # 移动 target_dict 中 tensor 到设备
        target_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in target_dict.items()
        }

        # 前向传播
        preds = model(input_tokens)
        batch_size = input_tokens.shape[0]
        total_samples += batch_size

        # ---- 逐项计算各维度损失（不做加权，独立评估） ----
        # CE
        pred_flat = preds["pred_tokens"].reshape(-1, VOCAB_SIZE)
        tgt_flat = target_on_device["target_tokens"].reshape(-1)
        ce_val = F.cross_entropy(pred_flat, tgt_flat).item()
        total_ce_loss += ce_val * batch_size

        # 音高
        p_mse = pitch_accuracy_MSE(preds["pred_pitch"], target_on_device["target_pitch"]).item()
        p_stab = pitch_stability_L1(preds["pred_pitch"], target_on_device["target_pitch"]).item()
        total_pitch_mse += p_mse * batch_size
        total_pitch_stab += p_stab * batch_size

        # 节奏
        r_mse = rhythm_consistency_MSE(preds["pred_duration"], target_on_device["target_duration"]).item()
        t_kl = tempo_kl(preds["pred_tempo_dist"], target_on_device["target_tempo_dist"]).item()
        total_rhythm_mse += r_mse * batch_size
        total_tempo_kl += t_kl * batch_size

        # 装饰音
        o_score = ornament_score_MSE(
            preds["pred_ornament"], target_on_device["target_ornament"],
            preds["pred_pitch"], target_on_device["target_pitch"],
            preds["pred_duration"], target_on_device["target_duration"],
        ).item()
        o_density = ornament_density_MSE(preds["pred_ornament"], target_on_device["target_ornament"]).item()
        total_orn_score += o_score * batch_size
        total_orn_density += o_density * batch_size

        # 调式
        m_kl = mode_kl(preds["pred_mode_logits"], target_on_device["target_mode_dist"]).item()
        total_mode_kl += m_kl * batch_size

        # 风格
        f_val = fad_diff(preds["pred_features"], target_on_device["target_features"]).item()
        s_cos = style_cosine_loss(preds["pred_style"], target_on_device["target_style"]).item()
        total_fad += f_val * batch_size
        total_style_cos += s_cos * batch_size

    # ---- 平均化 ----
    n = max(total_samples, 1)
    avg_ce = total_ce_loss / n
    avg_pitch_mse = total_pitch_mse / n
    avg_pitch_stab = total_pitch_stab / n
    avg_rhythm_mse = total_rhythm_mse / n
    avg_tempo_kl = total_tempo_kl / n
    avg_orn_score = total_orn_score / n
    avg_orn_density = total_orn_density / n
    avg_mode_kl = total_mode_kl / n
    avg_fad = total_fad / n
    avg_style_cos = total_style_cos / n

    # ---- 损失 → 得分映射 [0, 1]，越高越好 ----
    def loss_to_score(loss_val: float, scale: float = 1.0) -> float:
        """将损失值映射为 [0, 1] 得分。f(x) = 1 / (1 + scale * x)"""
        return float(1.0 / (1.0 + scale * loss_val))

    # 各维度子指标得分
    pitch_acc_score     = loss_to_score(avg_pitch_mse, scale=0.1)
    pitch_stab_score    = loss_to_score(avg_pitch_stab, scale=1.0)
    rhythm_cons_score   = loss_to_score(avg_rhythm_mse, scale=0.5)
    tempo_acc_score     = loss_to_score(avg_tempo_kl, scale=1.0)
    ornament_ors_score  = loss_to_score(avg_orn_score, scale=0.5)
    ornament_den_score  = loss_to_score(avg_orn_density, scale=1.0)
    mode_cons_score     = loss_to_score(avg_mode_kl, scale=1.0)
    # FAD: 越低越好，style_cosine: 越低越好
    fad_score           = loss_to_score(avg_fad, scale=0.01)
    style_cons_score    = loss_to_score(avg_style_cos, scale=1.0)

    # ---- NOEF 五大类汇总 ----
    pitch_cat_score    = 0.5 * pitch_acc_score + 0.5 * pitch_stab_score
    rhythm_cat_score   = 0.5 * rhythm_cons_score + 0.5 * tempo_acc_score
    ornament_cat_score = 0.5 * ornament_ors_score + 0.5 * ornament_den_score
    mode_cat_score     = mode_cons_score
    style_cat_score    = 0.5 * fad_score + 0.5 * style_cons_score

    # ---- 综合得分 ----
    composite = (
        NOEF_WEIGHTS["pitch"]    * pitch_cat_score +
        NOEF_WEIGHTS["rhythm"]   * rhythm_cat_score +
        NOEF_WEIGHTS["ornament"] * ornament_cat_score +
        NOEF_WEIGHTS["mode"]     * mode_cat_score +
        NOEF_WEIGHTS["style"]    * style_cat_score
    )

    # ---- 组装返回 ----
    metrics = {
        # 子指标
        "pitch_accuracy":       pitch_acc_score,
        "pitch_stability":      pitch_stab_score,
        "rhythm_consistency":   rhythm_cons_score,
        "tempo_accuracy":       tempo_acc_score,
        "ornament_score":       ornament_ors_score,
        "ornament_density":     ornament_den_score,
        "mode_consistency":     mode_cons_score,
        "fad":                  avg_fad,               # 原始FAD值（越低越好）
        "style_consistency":    style_cons_score,
        # 原始损失均值（供CSV记录）
        "_ce_loss":             avg_ce,
        "_pitch_mse_loss":      avg_pitch_mse,
        "_pitch_stab_loss":     avg_pitch_stab,
        "_rhythm_mse_loss":     avg_rhythm_mse,
        "_tempo_kl_loss":       avg_tempo_kl,
        "_orn_score_loss":      avg_orn_score,
        "_orn_density_loss":    avg_orn_density,
        "_mode_kl_loss":        avg_mode_kl,
        "_fad_loss":            avg_fad,
        "_style_cos_loss":      avg_style_cos,
        # 五大类得分
        "_pitch_cat":           pitch_cat_score,
        "_rhythm_cat":          rhythm_cat_score,
        "_ornament_cat":        ornament_cat_score,
        "_mode_cat":            mode_cat_score,
        "_style_cat":           style_cat_score,
        # 综合得分
        "noef_composite":       composite,
    }
    return metrics


# ==============================================================================
# 四、CSV 日志记录
# ==============================================================================

def init_csv_log(csv_path: str):
    """初始化CSV文件，写入表头（若文件不存在）。"""
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)

    # 定义CSV列名
    headers = [
        "epoch",
        "train_total_loss", "train_ce_loss",
        "train_pitch_loss", "train_rhythm_loss",
        "train_ornament_loss", "train_mode_loss", "train_style_loss",
        "lambda1", "lambda2", "lambda3", "lambda4", "lambda5",
        "val_pitch_accuracy", "val_pitch_stability",
        "val_rhythm_consistency", "val_tempo_accuracy",
        "val_ornament_score", "val_ornament_density",
        "val_mode_consistency", "val_fad", "val_style_consistency",
        "val_noef_composite",
    ]

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists or os.path.getsize(csv_path) == 0:
            writer.writerow(headers)


def append_csv_row(csv_path: str, epoch: int,
                   train_losses: Dict[str, float],
                   lambdas: Tuple[float, ...],
                   val_metrics: Dict[str, float]):
    """追加一行训练/验证记录到CSV文件。"""
    row = [
        epoch,
        f"{train_losses.get('total', 0.0):.6f}",
        f"{train_losses.get('ce', 0.0):.6f}",
        f"{train_losses.get('pitch', 0.0):.6f}",
        f"{train_losses.get('rhythm', 0.0):.6f}",
        f"{train_losses.get('ornament', 0.0):.6f}",
        f"{train_losses.get('mode', 0.0):.6f}",
        f"{train_losses.get('style', 0.0):.6f}",
        f"{lambdas[0]:.2f}", f"{lambdas[1]:.2f}",
        f"{lambdas[2]:.2f}", f"{lambdas[3]:.2f}", f"{lambdas[4]:.2f}",
        f"{val_metrics.get('pitch_accuracy', 0.0):.6f}",
        f"{val_metrics.get('pitch_stability', 0.0):.6f}",
        f"{val_metrics.get('rhythm_consistency', 0.0):.6f}",
        f"{val_metrics.get('tempo_accuracy', 0.0):.6f}",
        f"{val_metrics.get('ornament_score', 0.0):.6f}",
        f"{val_metrics.get('ornament_density', 0.0):.6f}",
        f"{val_metrics.get('mode_consistency', 0.0):.6f}",
        f"{val_metrics.get('fad', 0.0):.6f}",
        f"{val_metrics.get('style_consistency', 0.0):.6f}",
        f"{val_metrics.get('noef_composite', 0.0):.6f}",
    ]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# ==============================================================================
# 五、模型保存与加载
# ==============================================================================

def save_checkpoint(model: NanyinBoYaTCN,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    best_val_score: float,
                    lambdas: Tuple[float, ...],
                    save_dir: str,
                    filename: str = "best_model.pt"):
    """
    保存训练检查点。

    参数:
        model:      模型实例
        optimizer:  优化器实例
        epoch:      当前epoch
        best_val_score: 当前最佳验证集综合NOEF得分
        lambdas:    当前λ权重
        save_dir:   保存目录
        filename:   文件名
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_score": best_val_score,
        "lambdas": lambdas,
    }, filepath)
    print(f"[checkpoint] 模型已保存至: {filepath}")
    print(f"[checkpoint] 当前最佳验证NOEF得分: {best_val_score:.6f}")


def load_checkpoint(model: NanyinBoYaTCN,
                    optimizer: torch.optim.Optimizer,
                    save_dir: str,
                    filename: str = "best_model.pt",
                    device: torch.device = torch.device("cpu")
                    ) -> Tuple[int, float, Tuple[float, ...]]:
    """
    加载训练检查点，实现断点续训。

    返回:
        (start_epoch, best_val_score, lambdas)
        若检查点不存在则返回 (0, 0.0, (0.1,)*5)
    """
    filepath = os.path.join(save_dir, filename)
    if not os.path.exists(filepath):
        print("[checkpoint] 未找到检查点文件，从头开始训练。")
        return 0, 0.0, (0.1, 0.1, 0.1, 0.1, 0.1)

    try:
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1   # 从下一个epoch继续
        best_val_score = checkpoint.get("best_val_score", 0.0)
        saved_lambdas = checkpoint.get("lambdas", (0.1, 0.1, 0.1, 0.1, 0.1))
        print(f"[checkpoint] 成功加载检查点: {filepath}")
        print(f"[checkpoint] 续训起始epoch: {start_epoch}, 最佳NOEF得分: {best_val_score:.6f}")
        return start_epoch, best_val_score, saved_lambdas
    except Exception as e:
        print(f"[checkpoint] 加载检查点失败: {e}，从头开始训练。")
        return 0, 0.0, (0.1, 0.1, 0.1, 0.1, 0.1)


# ==============================================================================
# 六、训练主循环
# ==============================================================================

def train(finals: Optional[Tuple[float, float, float, float, float]] = None,
           csv_log_path: Optional[str] = None,
           checkpoint_subdir: Optional[str] = None):
    """
    BoYaTCN 南音旋律生成训练主函数。

    参数（网格搜索/消融实验调度时可外部传入，不传则使用默认值）：
        finals:            最终λ五元组 (λ1,λ2,λ3,λ4,λ5)，覆盖模块默认值
        csv_log_path:      自定义CSV日志文件路径
        checkpoint_subdir: checkpoints/ 下的子目录名
    """
    # ---- 若外部传参，覆盖对应路径 ----
    _csv_path = csv_log_path if csv_log_path is not None else CSV_LOG_PATH
    if checkpoint_subdir is not None:
        _ckpt_dir = os.path.join(_SCRIPT_DIR, "checkpoints", checkpoint_subdir)
    else:
        _ckpt_dir = CHECKPOINT_DIR

    # ---- 0. 设备检测 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] 计算设备: {device}")

    # ---- 1. 加载数据集 ----
    print("\n" + "=" * 60)
    print("[train] 加载训练集（15首）与验证集（3首）")
    print("=" * 60)

    train_ds = get_train_dataset()
    val_ds = get_val_dataset()

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_nanyin_batch, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_nanyin_batch, drop_last=False
    )

    print(f"[train] 训练集批数: {len(train_loader)}, 验证集批数: {len(val_loader)}")

    # ---- 2. 模型、优化器初始化 ----
    print("\n" + "=" * 60)
    print("[train] 初始化 NanyinBoYaTCN 模型")
    print("=" * 60)

    model = NanyinBoYaTCN(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    # 学习率调度器：每50个epoch衰减为原来的0.5倍
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] 模型总参数量: {total_params:,}")
    print(f"[train] 可训练参数量: {trainable_params:,}")

    # ---- 3. 断点续训加载 ----
    start_epoch, best_val_score, _ = load_checkpoint(
        model, optimizer, _ckpt_dir, "best_model.pt", device
    )
    early_stop_counter = 0  # 早停计数器：连续N轮未提升则终止

    # ---- 4. 初始化CSV日志 ----
    init_csv_log(_csv_path)
    print(f"[train] CSV日志路径: {_csv_path}")

    # ---- 5. 打印训练配置 ----
    print("\n" + "=" * 60)
    print("[train] 训练配置")
    print("=" * 60)
    print(f"  总epoch数:     {NUM_EPOCHS}")
    print(f"  批次大小:      {BATCH_SIZE}")
    print(f"  学习率:        {LEARNING_RATE}")
    print(f"  权重衰减:      {WEIGHT_DECAY}")
    print(f"  预热期:        0 ~ {WARMUP_EPOCHS-1} (λ=0.1)")
    print(f"  过渡期:        {WARMUP_EPOCHS} ~ {FIXED_EPOCH-1} (线性上调)")
    print(f"  稳定期:        {FIXED_EPOCH}+ (λ固定)")
    print(f"  最优λ:        λ1={FINAL_LAMBDA1}, λ2={FINAL_LAMBDA2}, "
          f"λ3={FINAL_LAMBDA3}, λ4={FINAL_LAMBDA4}, λ5={FINAL_LAMBDA5}")
    if finals is not None:
        print(f"  ★ 调度覆盖λ:  λ1={finals[0]}, λ2={finals[1]}, λ3={finals[2]}, "
              f"λ4={finals[3]}, λ5={finals[4]}")
    print(f"  λ搜索候选区间: {{0.1, 0.5, 1.0, 2.0, 5.0}}")
    print(f"  权重保存路径:  {_ckpt_dir}")
    print(f"  CSV日志路径:   {_csv_path}")
    print()

    # ---- 6. 训练循环 ----
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()

        # ---- 6a. 计算当前epoch的动态λ权重 ----
        lam1, lam2, lam3, lam4, lam5 = get_dynamic_lambdas(epoch, finals)

        # 累积统计
        epoch_total_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_pitch_loss = 0.0
        epoch_rhythm_loss = 0.0
        epoch_ornament_loss = 0.0
        epoch_mode_loss = 0.0
        epoch_style_loss = 0.0
        num_batches = 0

        for _, (input_tokens, target_dict) in enumerate(train_loader):
            input_tokens = input_tokens.to(device)

            # 移动 target_dict 中 tensor 到设备
            target_on_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in target_dict.items()
            }

            # ---- 梯度清零 ----
            optimizer.zero_grad()

            # ---- 前向传播 ----
            preds = model(input_tokens)

            # ---- 组装 loss_nanyin.py 所需的 target_dict ----
            # 合并真值（target_*）与预测（pred_*）到同一个字典
            loss_dict = {
                # 序列级字段
                "target_tokens":   target_on_device["target_tokens"],
                "target_pitch":    target_on_device["target_pitch"],
                "pred_pitch":      preds["pred_pitch"],
                "target_duration": target_on_device["target_duration"],
                "pred_duration":   preds["pred_duration"],
                "target_ornament": target_on_device["target_ornament"],
                "pred_ornament":   preds["pred_ornament"],
                # 全局级字段
                "target_tempo_dist": target_on_device["target_tempo_dist"],
                "pred_tempo_dist":   preds["pred_tempo_dist"],
                "target_mode_dist":  target_on_device["target_mode_dist"],
                "pred_mode_logits":  preds["pred_mode_logits"],
                "target_features":   target_on_device["target_features"],
                "pred_features":     preds["pred_features"],
                "target_style":      target_on_device["target_style"],
                "pred_style":        preds["pred_style"],
            }

            # ---- 计算复合TotalLoss ----
            loss = nanyin_total_loss(
                pred_tokens=preds["pred_tokens"],
                target_dict=loss_dict,
                lambda1=lam1, lambda2=lam2, lambda3=lam3,
                lambda4=lam4, lambda5=lam5,
            )

            # ---- 同时计算各分项损失（用于tracking统计）----
            comp_losses = compute_component_losses(preds, target_on_device)

            # ---- 反向传播 ----
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # ---- 累积统计 ----
            epoch_total_loss += loss.item()
            epoch_ce_loss += comp_losses["ce"]
            epoch_pitch_loss += comp_losses["pitch"]
            epoch_rhythm_loss += comp_losses["rhythm"]
            epoch_ornament_loss += comp_losses["ornament"]
            epoch_mode_loss += comp_losses["mode"]
            epoch_style_loss += comp_losses["style"]
            num_batches += 1

        # 计算epoch平均
        avg_total_loss = epoch_total_loss / max(num_batches, 1)
        avg_ce_loss = epoch_ce_loss / max(num_batches, 1)
        avg_pitch_loss = epoch_pitch_loss / max(num_batches, 1)
        avg_rhythm_loss = epoch_rhythm_loss / max(num_batches, 1)
        avg_ornament_loss = epoch_ornament_loss / max(num_batches, 1)
        avg_mode_loss = epoch_mode_loss / max(num_batches, 1)
        avg_style_loss = epoch_style_loss / max(num_batches, 1)

        # ---- 学习率调度 ----
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ---- 6b. 每个epoch结束后：验证集评估 ----
        print("\n" + "-" * 60)
        print(f"[train] ===== Epoch {epoch+1}/{NUM_EPOCHS} =====")
        print(f"[train] 当前λ权重: λ1={lam1:.2f}, λ2={lam2:.2f}, λ3={lam3:.2f}, "
              f"λ4={lam4:.2f}, λ5={lam5:.2f}")
        print(f"[train] 学习率: {current_lr:.6f}")
        print(f"[train] 训练集 TotalLoss: {avg_total_loss:.4f}")
        print(f"[train]   分项损失 -> CE={avg_ce_loss:.4f}, Pitch={avg_pitch_loss:.4f}, "
              f"Rhythm={avg_rhythm_loss:.4f}, Ornament={avg_ornament_loss:.4f}, "
              f"Mode={avg_mode_loss:.4f}, Style={avg_style_loss:.4f}")

        # ---- 验证集NOEF评估 ----
        val_metrics = validate_noef(model, val_loader, device, (lam1, lam2, lam3, lam4, lam5))

        print(f"\n[train] ----- 验证集 NOEF 指标 -----")
        print(f"  [音高类]   pitch_accuracy={val_metrics['pitch_accuracy']:.4f}  "
              f"pitch_stability={val_metrics['pitch_stability']:.4f}")
        print(f"  [节奏类]   rhythm_consistency={val_metrics['rhythm_consistency']:.4f}  "
              f"tempo_accuracy={val_metrics['tempo_accuracy']:.4f}")
        print(f"  [装饰音类] ornament_score={val_metrics['ornament_score']:.4f}  "
              f"ornament_density={val_metrics['ornament_density']:.4f}")
        print(f"  [调式类]   mode_consistency={val_metrics['mode_consistency']:.4f}")
        print(f"  [风格类]   fad={val_metrics['fad']:.4f}  "
              f"style_consistency={val_metrics['style_consistency']:.4f}")
        print(f"  >>> NOEF综合得分: {val_metrics['noef_composite']:.6f} <<<")

        # ---- 模型权重保存：仅在验证集NOEF得分提升时 ----
        current_val_score = val_metrics["noef_composite"]
        if current_val_score > best_val_score:
            best_val_score = current_val_score
            save_checkpoint(
                model, optimizer,
                epoch=epoch,
                best_val_score=best_val_score,
                lambdas=(lam1, lam2, lam3, lam4, lam5),
                save_dir=_ckpt_dir,
                filename="best_model.pt",
            )
            print(f"[train] *** 验证集NOEF得分提升至 {best_val_score:.6f}，已保存最佳权重 ***")
        else:
            # 每10个epoch周期性保存一次（作为备份）
            if (epoch + 1) % 10 == 0:
                save_checkpoint(
                    model, optimizer,
                    epoch=epoch,
                    best_val_score=best_val_score,
                    lambdas=(lam1, lam2, lam3, lam4, lam5),
                    save_dir=_ckpt_dir,
                    filename=f"checkpoint_epoch_{epoch+1}.pt",
                )

        # ---- 写入CSV日志 ----
        train_losses = {
            "total": avg_total_loss,
            "ce": avg_ce_loss,
            "pitch": avg_pitch_loss,
            "rhythm": avg_rhythm_loss,
            "ornament": avg_ornament_loss,
            "mode": avg_mode_loss,
            "style": avg_style_loss,
        }
        append_csv_row(_csv_path, epoch + 1, train_losses,
                       (lam1, lam2, lam3, lam4, lam5), val_metrics)

        # ---- 早停检查 ----
        if current_val_score > best_val_score:
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"[train] NOEF未提升累计: {early_stop_counter}/{EARLY_STOP_PATIENCE}")
        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f"\n[train] *** 早停触发：连续{EARLY_STOP_PATIENCE}轮NOEF未提升，终止训练 ***")
            break

    # ---- 7. 训练完成 ----
    print("\n" + "=" * 60)
    print(f"[train] BoYaTCN 训练完成！共 {NUM_EPOCHS} 个epoch")
    print(f"[train] 最佳验证集NOEF综合得分: {best_val_score:.6f}")
    print(f"[train] 最佳权重保存于: {os.path.join(_ckpt_dir, 'best_model.pt')}")
    print("=" * 60)


# ==============================================================================
# 七、脚本入口
# ==============================================================================

if __name__ == "__main__":
    train()
