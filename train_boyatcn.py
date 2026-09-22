# -*- coding: utf-8 -*-
"""
train_boyatcn.py - 南音BoYaTCN模型训练脚本
============================================

运行模式（--mode）：
  * ``noef``（默认旧路径，E0/exp 主线）：整曲 + NOEF 复合损失，见 train()；
  * ``compound``（E2 新增）：复合事件窗口训练，见 train_compound()。
      venv/Scripts/python.exe train_boyatcn.py --mode compound
    - 样本：256 事件窗口（训练集步长 128 滑动 / 验证集非重叠），
      窗口自带 BOS/EOS、attention_mask，末尾 time-shift 伪标签用 ignore_index 屏蔽；
    - 损失：total = 1.0*pitch_CE + 0.7*duration_CE + 0.7*shift_CE（**无 L0-L5 消融**）；
    - 早停：监控验证集 total loss，保存 best-val checkpoint；
    - 必打日志：窗口数量清单、max_notes 截断曲目清单、duration/shift 词表长尾频次统计。

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
import json
import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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
    # ---- E3：可微分布损失（全部基于生成头 logits）----
    build_pitch_class_matrix,
    build_value_bin_matrix,
    build_mode_distribution_matrix,
    pitch_class_distribution_loss,
    mode_distribution_loss,
    duration_distribution_loss,
    shift_ioi_distribution_loss,
)
from dataset_nanyin import (
    IGNORE_INDEX,
    get_train_dataset,
    get_val_dataset,
    collate_nanyin_batch,
    collate_compound_batch,
    compute_vocab_class_stats,
    get_compound_codec,
    get_compound_window_datasets,
)
from nanyin_config import load_config, E2_CONFIG_PATH
from nanyin_metrics import NOEF_CATEGORY_WEIGHTS


# ==============================================================================
# 辅助：因果自回归 target 对齐（训练目标 = 预测下一个 token）
# ==============================================================================

# 序列级真值字段：与模型输出逐位置对齐（pred[i] 对应 target[i+1]）
_SEQ_LEVEL_TARGET_FIELDS = ("target_tokens", "target_pitch", "target_duration", "target_ornament")


def shift_seq_targets(td: dict) -> dict:
    """
    将序列级真值右移一位（丢弃第一个 token），与因果自回归模型的 next-token
    预测输出（长度 L-1，输入 x[:-1]）对齐；全局级真值（与位置无关）保持不变。

    对齐关系：模型输入 x[:-1]，位置 i 的输出 = "给定 x[:i+1] 预测 x[i+1]"，
    对应的真值应为 x[i+1]，即 target_tokens[:, 1:]。
    若不做此对齐（旧行为 pred[i] vs target[i]），训练目标退化为双向完形/自复制，
    与自回归生成错位，导致续写 pitch_acc 极低。
    """
    return {
        k: (v[:, 1:] if k in _SEQ_LEVEL_TARGET_FIELDS else v)
        for k, v in td.items()
    }


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

    # 风格损失（E3：两个独立辅助头默认停用 → 前向返回 None，此时风格损失记 0）
    if preds.get("pred_features") is None or preds.get("pred_style") is None:
        style_loss = 0.0
    else:
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
EARLY_STOP_PATIENCE = 160        # 早停耐心：验证得分连续N轮不提升则终止。
                                 # 注意：验证集是模型从未见过的曲目，NOEF 很早就会饱和，
                                 # 若耐心过小会提前截断训练（λ权重尚未到位），导致模型欠拟合。
                                 # 160 > 剩余epoch，实际等价于跑满全部 epoch。

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
        # 因果自回归对齐：序列级 target 右移一位（对应 next-token 预测）
        target_on_device = shift_seq_targets(target_on_device)

        # 前向传播（输入去掉最后一个 token，预测其后的每个 token）
        preds = model(input_tokens[:, :-1])
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

        # 风格（E3：两个独立辅助头默认停用 → 前向返回 None，此时跳过统计）
        if preds.get("pred_features") is None or preds.get("pred_style") is None:
            pass
        else:
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
            # 因果自回归对齐：序列级 target 右移一位（对应 next-token 预测）
            target_on_device = shift_seq_targets(target_on_device)

            # ---- 梯度清零 ----
            optimizer.zero_grad()

            # ---- 前向传播（输入去掉最后一个 token，预测其后的每个 token）----
            preds = model(input_tokens[:, :-1])

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
# 七、E2：复合事件窗口训练（三输出头）
# ==============================================================================
#
# 与上方 NOEF 老路径完全隔离：老路径（train()）保持不动，E2 走 train_compound()。
# 损失口径（E2 当前阶段**不做 L0-L5 消融**，消融留给 E3）：
#     total_loss = 1.0 * pitch_CE + 0.7 * duration_CE + 0.7 * shift_CE
# shift_CE 使用 ignore_index=IGNORE_INDEX，屏蔽窗口最后一个事件位（伪标签）与 EOS 位。
# ==============================================================================


def set_seed(seed: int) -> None:
    """设置 E2 训练所需的全部随机种子（与老路径无关，E2 独立控制）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_distribution_matrices(codec, cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """E3：构造四种分布损失的常量聚合矩阵（不参与梯度）。

    矩阵由 codec 的物理量映射（index_to_duration / index_to_shift）与配置的
    直方图分箱边界算出：
      pc       (pitch_vocab_size, 12)  MIDI 音高 → 音级
      mode     (4, 12)                 南音四调五声音级先验分布
      duration (duration_num_classes, K_d)  duration 类别 → 秒级直方图 bin
      shift    (shift_num_classes, K_s)     shift 类别 → IOI 秒级直方图 bin
    """
    dcfg = cfg.get("distribution", {})
    if not dcfg:
        raise KeyError("[train_boyatcn] 配置缺少 distribution 小节（E3 损失需要分箱边界）")
    # L5 装饰音框架：短音类别掩码（duration 秒值 <= 阈值 → 装饰音候选）
    short_thr = float(dcfg.get("ornament_short_duration_sec", 0.15))
    short_mask = torch.tensor(
        [codec.index_to_duration(i) <= short_thr for i in range(int(codec.duration_num_classes))],
        dtype=torch.bool,
    )
    return {
        "duration_short": short_mask,
        "ornament_short_sec": short_thr,
        "pc": build_pitch_class_matrix(int(codec.pitch_vocab_size)),
        "mode": build_mode_distribution_matrix(),
        "duration": build_value_bin_matrix(
            [codec.index_to_duration(i) for i in range(int(codec.duration_num_classes))],
            dcfg["duration_bin_edges"],
        ),
        "shift": build_value_bin_matrix(
            [codec.index_to_shift(i) for i in range(int(codec.shift_num_classes))],
            dcfg["shift_ioi_bin_edges"],
        ),
    }


def compute_compound_loss(
    pred: Dict[str, torch.Tensor],
    tgt: Dict[str, torch.Tensor],
    w_pitch: float,
    w_duration: float,
    w_shift: float,
    w_pc: float = 0.0,
    w_mode: float = 0.0,
    w_dur_dist: float = 0.0,
    w_ioi_dist: float = 0.0,
    w_ornament: float = 0.0,
    matrices: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """E2 三路分类 CE（+ E3 可选分布损失）。

        total = w_pitch * pitch_CE + w_duration * duration_CE + w_shift * shift_CE
              + w_pc       * 音级分布KL      （L1，基于 pitch_head logits）
              + w_mode     * 调式分布KL      （L2，基于 pitch_head logits）
              + w_dur_dist * 时值分布KL      （L3，基于 duration_head logits）
              + w_ioi_dist * IOI分布KL       （L3，基于 shift_head logits）

    shift_CE 使用 ignore_index=IGNORE_INDEX：窗口最后一个事件位（伪标签）与 EOS 位
    不参与损失，也不参与反向传播。分布损失同样剔除这些位置（以及 BOS/EOS 特殊 token）。

    权重为 0 时对应分布损失**完全不计算**（与 E2 口径逐位一致）。
    """
    pitch_ce = F.cross_entropy(
        pred["pred_pitch_logits"].reshape(-1, pred["pred_pitch_logits"].shape[-1]),
        tgt["target_pitch_ids"].reshape(-1),
    )
    duration_ce = F.cross_entropy(
        pred["pred_duration_logits"].reshape(-1, pred["pred_duration_logits"].shape[-1]),
        tgt["target_duration_ids"].reshape(-1),
    )
    shift_ce = F.cross_entropy(
        pred["pred_shift_logits"].reshape(-1, pred["pred_shift_logits"].shape[-1]),
        tgt["target_shift_ids"].reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    total = w_pitch * pitch_ce + w_duration * duration_ce + w_shift * shift_ce

    info: Dict[str, float] = {
        "pitch_ce": float(pitch_ce.detach()),
        "duration_ce": float(duration_ce.detach()),
        "shift_ce": float(shift_ce.detach()),
        "pc_kl": 0.0, "mode_kl": 0.0, "dur_dist_kl": 0.0, "ioi_dist_kl": 0.0,
        "ornament_kl": 0.0,
        # 与 E2 可比口径：只含三路加权 CE（不含分布损失），用于跨组对照与早停判定
        "ce_total": float((w_pitch * pitch_ce + w_duration * duration_ce + w_shift * shift_ce).detach()),
    }

    if matrices is not None and (w_pc or w_mode or w_dur_dist or w_ioi_dist or w_ornament):
        # 音级/调式：剔除 BOS/EOS 等特殊 token 位（pitch id >= 128）
        pitch_valid = tgt["target_pitch_ids"] < matrices["pc"].shape[0]
        if w_pc:
            pc_kl = pitch_class_distribution_loss(
                pred["pred_pitch_logits"], tgt["target_pitch_ids"],
                pc_matrix=matrices["pc"], valid_mask=pitch_valid,
            )
            total = total + w_pc * pc_kl
            info["pc_kl"] = float(pc_kl.detach())
        if w_mode:
            mode_kl_v = mode_distribution_loss(
                pred["pred_pitch_logits"], tgt["target_mode_index"],
                pc_matrix=matrices["pc"], mode_dist_matrix=matrices["mode"],
                valid_mask=pitch_valid,
            )
            total = total + w_mode * mode_kl_v
            info["mode_kl"] = float(mode_kl_v.detach())
        if w_dur_dist:
            dur_kl = duration_distribution_loss(
                pred["pred_duration_logits"], tgt["target_duration_ids"],
                bin_matrix=matrices["duration"],
            )
            total = total + w_dur_dist * dur_kl
            info["dur_dist_kl"] = float(dur_kl.detach())
        if w_ioi_dist:
            ioi_kl = shift_ioi_distribution_loss(
                pred["pred_shift_logits"], tgt["target_shift_ids"],
                bin_matrix=matrices["shift"],
            )
            total = total + w_ioi_dist * ioi_kl
            info["ioi_dist_kl"] = float(ioi_kl.detach())
        if w_ornament:
            # L5 框架：GT 短音比例 vs duration_head 预测的短音概率质量
            num_dur = int(matrices["duration_short"].shape[0])
            dur_valid = tgt["target_duration_ids"] < num_dur
            short_tgt = matrices["duration_short"].to(device=tgt["target_duration_ids"].device)
            short_target = short_tgt[tgt["target_duration_ids"].clamp(min=0, max=num_dur - 1)]
            orn_loss = ornament_density_distribution_loss(
                pred["pred_duration_logits"], short_target,
                short_class_mask=matrices["duration_short"], valid_mask=dur_valid,
            )
            total = total + w_ornament * orn_loss
            info["ornament_kl"] = float(orn_loss.detach())

    info["total"] = float(total.detach())
    return total, info


@torch.no_grad()
def evaluate_compound(
    model: NanyinBoYaTCN,
    val_loader: DataLoader,
    device: torch.device,
    w_pitch: float,
    w_duration: float,
    w_shift: float,
    w_pc: float = 0.0,
    w_mode: float = 0.0,
    w_dur_dist: float = 0.0,
    w_ioi_dist: float = 0.0,
    w_ornament: float = 0.0,
    matrices: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    """验证集评估：加权 total loss + 三路分项 CE + 分布损失 + token 级准确率。"""
    model.eval()
    acc = {"pitch_ce": 0.0, "duration_ce": 0.0, "shift_ce": 0.0, "total": 0.0,
           "pc_kl": 0.0, "mode_kl": 0.0, "dur_dist_kl": 0.0, "ioi_dist_kl": 0.0,
           "ornament_kl": 0.0, "ce_total": 0.0}
    correct = {"pitch": 0, "duration": 0, "shift": 0}
    n_tok = {"pitch": 0, "duration": 0, "shift": 0}
    num_batches = 0

    for batch in val_loader:
        tgt = {
            "target_pitch_ids": batch["target_pitch_ids"].to(device),
            "target_duration_ids": batch["target_duration_ids"].to(device),
            "target_shift_ids": batch["target_shift_ids"].to(device),
            "target_mode_index": batch["target_mode_index"].to(device),
        }
        pred = model(
            batch["input_pitch_ids"].to(device),
            batch["input_duration_ids"].to(device),
            batch["input_shift_ids"].to(device),
        )
        _, info = compute_compound_loss(
            pred, tgt, w_pitch, w_duration, w_shift,
            w_pc=w_pc, w_mode=w_mode, w_dur_dist=w_dur_dist, w_ioi_dist=w_ioi_dist,
            w_ornament=w_ornament, matrices=matrices,
        )
        for k in acc:
            acc[k] += info[k]
        num_batches += 1

        correct["pitch"] += int((pred["pred_pitch_logits"].argmax(-1) == tgt["target_pitch_ids"]).sum())
        correct["duration"] += int((pred["pred_duration_logits"].argmax(-1) == tgt["target_duration_ids"]).sum())
        valid = tgt["target_shift_ids"] != IGNORE_INDEX
        correct["shift"] += int((pred["pred_shift_logits"].argmax(-1)[valid] == tgt["target_shift_ids"][valid]).sum())
        n_tok["pitch"] += int(tgt["target_pitch_ids"].numel())
        n_tok["duration"] += int(tgt["target_duration_ids"].numel())
        n_tok["shift"] += int(valid.sum())

    n = max(num_batches, 1)
    out = {k: v / n for k, v in acc.items()}
    for k in ("pitch", "duration", "shift"):
        out[f"{k}_acc"] = correct[k] / n_tok[k] if n_tok[k] else 0.0
    return out


def _e2_log(log_lines: List[str], msg: str, fh=None) -> None:
    """打印到 stdout 并同步写入 E2 日志文件。"""
    print(msg)
    if fh is not None:
        fh.write(msg + "\n")
        fh.flush()
    log_lines.append(msg)


def _log_vocab_stats(stats: Dict[str, Any], log_lines: List[str], fh=None,
                     top_k: int = 10, compact: bool = False) -> None:
    """打印 duration / time-shift 词表类别频次统计（长尾监控）。"""
    _e2_log(log_lines, f"  [词表统计] 窗口数={stats['num_windows']}, 事件数={stats['num_events']}", fh)
    for name in ("duration", "shift"):
        s = stats[name]
        _e2_log(log_lines,
                f"    - {name:<9} 类别总数={s['num_classes']}, 实际出现={s['used_classes']}, "
                f"零样本类别={s['zero_count_classes']}, 样本token数={s['total_tokens']}", fh)
        _e2_log(log_lines,
                f"      长尾（频次<={s['long_tail_threshold']}）类别={s['long_tail_classes']}, "
                f"占样本比例={s['long_tail_token_share'] * 100:.2f}%", fh)
        if not compact:
            top = ", ".join(f"{c['class_id']}:{c['count']}" for c in s["top_classes"])
            _e2_log(log_lines, f"      频次 Top{top_k}: {top}", fh)


def train_compound(
    num_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    seed: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
    patience: Optional[int] = None,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """E2/E3 复合事件窗口训练主流程（加权三路 CE + 可选分布损失 + 早停 best-val）。

    参数:
        config_path: E3 新增。覆盖层 YAML 路径（如 configs/e3_L2.yaml）；
                     None 时沿用 E2 配置 configs/e2_compound_window.yaml（纯三路 CE）。
    """
    cfg = load_config(config_path if config_path is not None else E2_CONFIG_PATH)
    tcfg = cfg["training"]
    outcfg = cfg["outputs"]

    num_epochs = int(num_epochs if num_epochs is not None else tcfg["num_epochs"])
    batch_size = int(batch_size if batch_size is not None else tcfg["batch_size"])
    learning_rate = float(learning_rate if learning_rate is not None else tcfg["learning_rate"])
    seed = int(seed if seed is not None else cfg["experiment"]["seed"])
    patience = int(patience if patience is not None else tcfg["early_stop_patience"])

    def _fmt(p: Optional[str]) -> Optional[str]:
        """路径模板：把 {seed} 替换为当前随机种子（E3 多籽实验用）。"""
        return p.replace("{seed}", str(seed)) if isinstance(p, str) else p

    checkpoint_dir = _fmt(checkpoint_dir if checkpoint_dir is not None else tcfg["checkpoint_dir"])
    os.makedirs(checkpoint_dir, exist_ok=True)

    w_pitch = float(tcfg["loss_weight_pitch"])
    w_duration = float(tcfg["loss_weight_duration"])
    w_shift = float(tcfg["loss_weight_shift"])
    # ---- E3：四项分布损失权重（0 = 不启用；E2 配置无这些键 → 默认 0，口径不变）----
    w_pc = float(tcfg.get("loss_weight_pitch_class", 0.0))
    w_mode = float(tcfg.get("loss_weight_mode", 0.0))
    w_dur_dist = float(tcfg.get("loss_weight_duration_dist", 0.0))
    w_ioi_dist = float(tcfg.get("loss_weight_ioi_dist", 0.0))
    w_ornament = float(tcfg.get("loss_weight_ornament", 0.0))     # L5 框架（默认 0）
    vocab_every = int(tcfg["vocab_log_every_epochs"])
    top_k = int(tcfg["top_k_classes"])

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_path = _fmt(outcfg["log_file"])
    csv_path = _fmt(outcfg["csv_file"])
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log_lines: List[str] = []
    fh = open(log_path, "w", encoding="utf-8")

    stage = cfg["experiment"].get("name", "E2_compound_window")
    _e2_log(log_lines, "=" * 74, fh)
    _e2_log(log_lines, f"{stage} — 复合事件窗口训练 BoYaTCN（分支 {cfg['experiment']['branch']}，"
                       f"配置 {config_path or E2_CONFIG_PATH}）", fh)
    _e2_log(log_lines, "=" * 74, fh)
    _e2_log(log_lines, f"设备={device}, seed={seed}, epochs={num_epochs}, batch_size={batch_size}, "
                       f"lr={learning_rate}, weight_decay={tcfg['weight_decay']}, "
                       f"early_stop_patience={patience}", fh)
    _e2_log(log_lines, f"损失: total = {w_pitch}*pitch_CE + {w_duration}*duration_CE + {w_shift}*shift_CE"
                       f"（shift 使用 ignore_index={IGNORE_INDEX} 屏蔽窗口尾伪标签）", fh)
    if any((w_pc, w_mode, w_dur_dist, w_ioi_dist, w_ornament)):
        _e2_log(log_lines,
                f"      + {w_pc}*音级分布KL + {w_mode}*调式分布KL + {w_dur_dist}*时值分布KL"
                f" + {w_ioi_dist}*IOI分布KL + {w_ornament}*装饰音密度损失"
                f"（全部基于生成头 logits，见 loss_nanyin 第九节）", fh)
    else:
        _e2_log(log_lines, "      无分布损失（L0 基线口径：仅三路 CE）。", fh)
    _e2_log(log_lines, "早停/最优判定监控量 = 验证集三路加权 CE（与 E2 口径一致，跨组可比）；"
                       "分布损失只参与优化，不参与最优判定。", fh)

    # ---------------- 数据 ----------------
    codec = get_compound_codec(cfg)
    _e2_log(log_lines, f"\n[编解码] time_unit={codec.time_unit_sec}s, 词表 "
                       f"pitch={codec.pitch_vocab_total}/duration={codec.duration_vocab_total}"
                       f"/shift={codec.shift_vocab_total}（含BOS/EOS）, max_notes={codec.max_notes}", fh)

    # ---- E3：分布损失聚合矩阵（常量，不参与梯度）----
    use_dist = any((w_pc, w_mode, w_dur_dist, w_ioi_dist, w_ornament))
    matrices = build_distribution_matrices(codec, cfg) if use_dist else None
    if matrices is not None:
        _e2_log(log_lines,
                f"[分布损失] 聚合矩阵 pc={tuple(matrices['pc'].shape)}, "
                f"mode={tuple(matrices['mode'].shape)}, duration={tuple(matrices['duration'].shape)}, "
                f"shift={tuple(matrices['shift'].shape)}", fh)

    train_ds, val_ds = get_compound_window_datasets(cfg=cfg, verbose=False)

    # ---- E3：如实记录窗口级调式真值分布（mode 损失的目标来源）----
    tr_modes, va_modes = train_ds.mode_index_counts(), val_ds.mode_index_counts()
    _e2_log(log_lines, f"[调式真值分布] 训练窗口 " + ", ".join(
        f"{NANYIN_FOUR_MODES[i]}={tr_modes[i]}" for i in range(4)), fh)
    _e2_log(log_lines, f"{'':>16}验证窗口 " + ", ".join(
        f"{NANYIN_FOUR_MODES[i]}={va_modes[i]}" for i in range(4)), fh)

    _e2_log(log_lines, "\n[窗口统计] 训练集（滑动，步长128） + 验证集（非重叠，步长256）", fh)
    _e2_log(log_lines, f"  {'曲目':<28}{'划分':<7}{'事件数':>8}{'窗口数':>8}{'尾丢弃':>8}{'max_notes截断':>14}", fh)
    rows = train_ds.song_stats_rows() + val_ds.song_stats_rows()
    for r in rows:
        _e2_log(log_lines, f"  {r['song_id']:<28}{r['split']:<7}{r['num_events']:>8}"
                           f"{r['num_windows']:>8}{r['discarded_tail_events']:>8}"
                           f"{r['truncated_notes']:>14}", fh)
    _e2_log(log_lines, f"  训练集窗口数={len(train_ds)}, 验证集窗口数={len(val_ds)}", fh)

    # ---- 硬性要求：如实打印被 max_notes 截断的曲目清单 ----
    truncated = train_ds.truncated_songs() + val_ds.truncated_songs()
    _e2_log(log_lines, f"\n[截断清单] 被 max_notes={codec.max_notes} 截断的曲目（共 {len(truncated)} 首）：", fh)
    if truncated:
        for r in truncated:
            _e2_log(log_lines, f"  - {r['song_id']} [{r['split']}]：丢弃 {r['truncated_notes']} 个音符"
                               f"（保留 {r['num_events']}）", fh)
    else:
        _e2_log(log_lines, "  （无）", fh)

    window_stats_csv = _fmt(outcfg["window_stats_csv"])
    os.makedirs(os.path.dirname(window_stats_csv) or ".", exist_ok=True)
    with open(window_stats_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _e2_log(log_lines, f"  窗口统计清单已写入: {window_stats_csv}", fh)

    # ---- 硬性要求：词表类别频次统计（长尾监控）----
    _e2_log(log_lines, "\n[词表类别频次统计] 训练集监督位置（shift 跳过伪标签位）：", fh)
    vocab_stats = compute_vocab_class_stats(
        train_ds, top_k=top_k, long_tail_threshold=int(tcfg["long_tail_threshold"])
    )
    _log_vocab_stats(vocab_stats, log_lines, fh, top_k=top_k)
    vocab_stats_json = _fmt(outcfg["vocab_stats_json"])
    os.makedirs(os.path.dirname(vocab_stats_json) or ".", exist_ok=True)
    with open(vocab_stats_json, "w", encoding="utf-8") as f:
        json.dump(vocab_stats, f, ensure_ascii=False, indent=2)
    _e2_log(log_lines, f"  词表统计明细已写入: {vocab_stats_json}", fh)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=bool(tcfg["shuffle"]),
        collate_fn=collate_compound_batch, generator=g,
        num_workers=int(tcfg["num_workers"]),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_compound_batch, num_workers=int(tcfg["num_workers"]),
    )

    # ---------------- 模型 ----------------
    model = NanyinBoYaTCN().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=float(tcfg["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(tcfg["lr_step_size"]), gamma=float(tcfg["lr_gamma"])
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _e2_log(log_lines, f"\n[模型] NanyinBoYaTCN 可训练参数 {n_params:,}", fh)

    # ---------------- 训练循环 ----------------
    history: List[Dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = -1
    no_improve = 0
    best_path = os.path.join(checkpoint_dir, _fmt(tcfg["best_ckpt_name"]))
    periodic_every = int(tcfg.get("periodic_ckpt_every", 0))
    start_time = time.time()

    _e2_log(log_lines, "\n[训练开始]", fh)
    _e2_log(log_lines, f"{'epoch':>6}{'train_total':>13}{'pitch':>10}{'dur':>10}{'shift':>10}"
                       f"{'val_total':>12}{'val_p':>9}{'val_d':>9}{'val_s':>9}"
                       f"{'v_pc':>8}{'v_md':>8}{'v_du':>8}{'v_ioi':>8}{'lr':>10}{'备注':>8}", fh)

    for epoch in range(1, num_epochs + 1):
        model.train()
        run = {"pitch_ce": 0.0, "duration_ce": 0.0, "shift_ce": 0.0, "total": 0.0,
               "pc_kl": 0.0, "mode_kl": 0.0, "dur_dist_kl": 0.0, "ioi_dist_kl": 0.0,
               "ornament_kl": 0.0, "ce_total": 0.0}
        n_batches = 0
        for batch in train_loader:
            tgt = {
                "target_pitch_ids": batch["target_pitch_ids"].to(device),
                "target_duration_ids": batch["target_duration_ids"].to(device),
                "target_shift_ids": batch["target_shift_ids"].to(device),
                "target_mode_index": batch["target_mode_index"].to(device),
            }
            pred = model(
                batch["input_pitch_ids"].to(device),
                batch["input_duration_ids"].to(device),
                batch["input_shift_ids"].to(device),
            )
            loss, info = compute_compound_loss(
                pred, tgt, w_pitch, w_duration, w_shift,
                w_pc=w_pc, w_mode=w_mode, w_dur_dist=w_dur_dist, w_ioi_dist=w_ioi_dist,
                w_ornament=w_ornament, matrices=matrices,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(tcfg["grad_clip"]))
            optimizer.step()
            for k in run:
                run[k] += info[k]
            n_batches += 1

        n_batches = max(n_batches, 1)
        train_avg = {k: v / n_batches for k, v in run.items()}
        val = evaluate_compound(
            model, val_loader, device, w_pitch, w_duration, w_shift,
            w_pc=w_pc, w_mode=w_mode, w_dur_dist=w_dur_dist, w_ioi_dist=w_ioi_dist,
            w_ornament=w_ornament, matrices=matrices,
        )
        lr_now = optimizer.param_groups[0]["lr"]

        # 最优/早停判定：三路加权 CE（跨组、与 E2 可比）；val["total"] 含分布损失仅作参考
        note = ""
        is_best = val["ce_total"] < best_val
        if is_best:
            best_val = val["ce_total"]
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val,
                "val_total_with_dist": val["total"],
                "train_loss": train_avg["total"],
                "config": cfg,
            }, best_path)
            note = "★best"
        else:
            no_improve += 1

        _e2_log(log_lines,
                f"{epoch:>6}{train_avg['total']:>13.4f}{train_avg['pitch_ce']:>10.4f}"
                f"{train_avg['duration_ce']:>10.4f}{train_avg['shift_ce']:>10.4f}"
                f"{val['total']:>12.4f}{val['pitch_ce']:>9.4f}{val['duration_ce']:>9.4f}"
                f"{val['shift_ce']:>9.4f}{val['pc_kl']:>8.4f}{val['mode_kl']:>8.4f}"
                f"{val['dur_dist_kl']:>8.4f}{val['ioi_dist_kl']:>8.4f}"
                f"{lr_now:>10.2e}{note:>8}", fh)

        history.append({
            "epoch": epoch,
            "train_total": train_avg["total"],
            "train_pitch_ce": train_avg["pitch_ce"],
            "train_duration_ce": train_avg["duration_ce"],
            "train_shift_ce": train_avg["shift_ce"],
            "train_pc_kl": train_avg["pc_kl"],
            "train_mode_kl": train_avg["mode_kl"],
            "train_dur_dist_kl": train_avg["dur_dist_kl"],
            "train_ioi_dist_kl": train_avg["ioi_dist_kl"],
            "train_ornament_kl": train_avg["ornament_kl"],
            "train_ce_total": train_avg["ce_total"],
            "val_total": val["total"],
            "val_pitch_ce": val["pitch_ce"],
            "val_duration_ce": val["duration_ce"],
            "val_shift_ce": val["shift_ce"],
            "val_pc_kl": val["pc_kl"],
            "val_mode_kl": val["mode_kl"],
            "val_dur_dist_kl": val["dur_dist_kl"],
            "val_ioi_dist_kl": val["ioi_dist_kl"],
            "val_ornament_kl": val["ornament_kl"],
            "val_ce_total": val["ce_total"],
            "val_pitch_acc": val["pitch_acc"],
            "val_duration_acc": val["duration_acc"],
            "val_shift_acc": val["shift_acc"],
            "lr": lr_now,
            "is_best": is_best,
        })

        if periodic_every > 0 and (epoch % periodic_every == 0):
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val["total"],
            }, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"))

        if vocab_every > 0 and (epoch % vocab_every == 0):
            _e2_log(log_lines, f"  [词表长尾摘要 @epoch {epoch}]", fh)
            _log_vocab_stats(vocab_stats, log_lines, fh, top_k=top_k, compact=True)

        scheduler.step()

        if no_improve >= patience:
            _e2_log(log_lines, f"\n[早停] 验证集 total loss 连续 {patience} 轮未下降，"
                               f"于 epoch {epoch} 终止训练。", fh)
            break

    elapsed = time.time() - start_time
    _e2_log(log_lines, "\n[训练结束]", fh)
    _e2_log(log_lines, f"  总轮数={len(history)}, 耗时={elapsed / 60:.1f} 分钟", fh)
    _e2_log(log_lines, f"  最优验证三路加权 CE={best_val:.4f} @ epoch {best_epoch}"
                       f"（best-val checkpoint: {best_path}；该量为跨组/与 E2 可比口径）", fh)
    best_row = next((h for h in history if h["epoch"] == best_epoch), None)
    if best_row:
        _e2_log(log_lines, f"  该轮验证 token 准确率: pitch={best_row['val_pitch_acc'] * 100:.2f}%, "
                           f"duration={best_row['val_duration_acc'] * 100:.2f}%, "
                           f"shift={best_row['val_shift_acc'] * 100:.2f}%", fh)

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    _e2_log(log_lines, f"  逐轮日志 CSV: {csv_path}", fh)
    _e2_log(log_lines, f"  完整训练日志: {log_path}", fh)

    # ---- 训练结束绘制损失曲线（E3 硬性要求：每组一张 PNG）----
    curve_png = _fmt(outcfg.get("loss_curve_png"))
    if curve_png:
        try:
            import plot_loss_curve
            os.makedirs(os.path.dirname(curve_png) or ".", exist_ok=True)
            plot_loss_curve.plot(csv_path, curve_png,
                                 title=f"{stage}（seed={seed}）训练/验证损失曲线")
            _e2_log(log_lines, f"  损失曲线 PNG: {curve_png}", fh)
        except Exception as exc:      # 作图失败不阻断训练结果
            _e2_log(log_lines, f"  ⚠ 损失曲线作图失败: {exc}", fh)
    fh.close()

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "best_ckpt": best_path,
        "num_train_windows": len(train_ds),
        "num_val_windows": len(val_ds),
        "elapsed_sec": elapsed,
        "vocab_stats": vocab_stats,
        "num_params": n_params,
        "csv_path": csv_path,
        "log_path": log_path,
        "curve_png": curve_png,
    }


# ==============================================================================
# 八、脚本入口
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BoYaTCN 训练脚本")
    parser.add_argument("--mode", type=str, default="noef", choices=["noef", "compound"],
                        help="noef=E0/exp 主线整曲训练（默认）；compound=E2 复合事件窗口训练")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None,
                        help="覆盖层 YAML 路径（E3 消融：configs/e3_L0.yaml ... e3_L4.yaml）；"
                             "默认 configs/e2_compound_window.yaml")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):           # type: ignore[attr-defined]
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

    if args.mode == "compound":
        result = train_compound(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
            patience=args.patience,
            config_path=args.config,
        )
        print(f"\n完成：最优验证 total loss={result['best_val_loss']:.4f} "
              f"@ epoch {result['best_epoch']}；checkpoint: {result['best_ckpt']}")
    else:
        train()
