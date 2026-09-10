# -*- coding: utf-8 -*-
"""
loss_nanyin.py - 南音NOEF可微分复合损失工具
基于PyTorch实现，支持反向传播

NOEF五维客观评估体系对应损失：
  1.音高(Pitch)  ：pitch_accuracy_MSE、pitch_stability_L1
  2.节奏(Rhythm) ：rhythm_consistency_MSE、tempo_kl
  3.装饰音(Ornament)：ornament_score_MSE、ornament_density_MSE
  4.调式(Mode)   ：mode_kl（适配南音4种专属调式）
  5.风格(Style)  ：fad_diff、style_cosine_loss

总损失 = 序列分类交叉熵 + λ1*音高损失 + λ2*节奏损失 + λ3*装饰音损失 + λ4*调式损失 + λ5*风格损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==============================================================================
# 一、音高类损失 (Pitch Loss)
# ==============================================================================

def pitch_accuracy_MSE(pred_pitch: torch.Tensor, target_pitch: torch.Tensor) -> torch.Tensor:
    """
    音高准确度MSE损失

    计算模型预测音高序列与真实标注音高序列之间的均方误差，
    衡量生成音高与目标音高的偏差程度，值越小表示音高越准确。

    文献来源：
        MelodyGLM (https://arxiv.org/pdf/2309.10738v1)
        该文提出旋律生成模型，使用MSE约束生成音高与目标音高的距离，
        以保证旋律音高轮廓的准确性。

    参数:
        pred_pitch:   模型预测音高张量, shape=(batch, seq_len) 或 (batch, seq_len, 1)
        target_pitch: 真实标注音高张量, shape与pred_pitch相同
    返回:
        MSE损失标量张量
    """
    if pred_pitch.dim() == 3:
        pred_pitch = pred_pitch.squeeze(-1)
    if target_pitch.dim() == 3:
        target_pitch = target_pitch.squeeze(-1)
    loss = F.mse_loss(pred_pitch, target_pitch)
    print(f"[音高] pitch_accuracy_MSE = {loss.item():.6f}")
    return loss


def pitch_stability_L1(pred_pitch: torch.Tensor, target_pitch: torch.Tensor) -> torch.Tensor:
    """
    音高稳定性L1损失

    计算预测音高相邻帧差分与真实音高相邻帧差分之间的L1距离，
    衡量音高变化的平滑程度是否与真实一致，避免生成音高抖动过大。

    文献来源：
        Controllable Symbolic Music Generation
        (https://www.preprints.org/manuscript/202604.0984)
        该文在可控符号音乐生成中，使用差分约束保证旋律轮廓的平滑性，
        避免生成结果出现不自然的音高跳变。

    参数:
        pred_pitch:   模型预测音高张量, shape=(batch, seq_len) 或 (batch, seq_len, 1)
        target_pitch: 真实标注音高张量, shape与pred_pitch相同
    返回:
        L1损失标量张量
    """
    if pred_pitch.dim() == 3:
        pred_pitch = pred_pitch.squeeze(-1)
    if target_pitch.dim() == 3:
        target_pitch = target_pitch.squeeze(-1)
    # 计算相邻帧差分: shape=(batch, seq_len-1)
    pred_diff = pred_pitch[:, 1:] - pred_pitch[:, :-1]
    target_diff = target_pitch[:, 1:] - target_pitch[:, :-1]
    loss = F.l1_loss(pred_diff, target_diff)
    print(f"[音高] pitch_stability_L1 = {loss.item():.6f}")
    return loss


# ==============================================================================
# 二、节奏类损失 (Rhythm Loss)
# ==============================================================================

def rhythm_consistency_MSE(pred_duration: torch.Tensor, target_duration: torch.Tensor) -> torch.Tensor:
    """
    节奏一致性MSE损失

    计算模型预测时值序列与真实标注时值序列之间的均方误差，
    衡量生成节奏与目标节奏的偏差程度，值越小表示节奏越一致。

    文献来源：
        TCSinger (https://arxiv.org/pdf/2409.15977)
        该文在歌唱语音合成中使用时长MSE约束，确保生成音频的
        节奏时值分布与真实演唱节奏对齐。

    参数:
        pred_duration:   模型预测时值张量, shape=(batch, seq_len) 或 (batch, seq_len, 1)
        target_duration: 真实标注时值张量, shape与pred_duration相同
    返回:
        MSE损失标量张量
    """
    if pred_duration.dim() == 3:
        pred_duration = pred_duration.squeeze(-1)
    if target_duration.dim() == 3:
        target_duration = target_duration.squeeze(-1)
    loss = F.mse_loss(pred_duration, target_duration)
    print(f"[节奏] rhythm_consistency_MSE = {loss.item():.6f}")
    return loss


def tempo_kl(pred_tempo_dist: torch.Tensor, target_tempo_dist: torch.Tensor) -> torch.Tensor:
    """
    速度分布KL散度损失

    计算预测速度分布与真实速度分布之间的KL散度，
    衡量生成速度分布与目标速度分布的差异，值越小表示速度风格越匹配。

    文献来源：
        TCSinger (https://arxiv.org/pdf/2409.15977)
        该文使用KL散度约束节奏风格分布，使生成结果的节奏风格
        与目标风格在概率分布层面保持一致。

    参数:
        pred_tempo_dist:   模型预测速度分布, shape=(batch, num_tempo_bins)
                           需经过softmax归一化（函数内部会处理）
        target_tempo_dist: 真实速度分布, shape与pred_tempo_dist相同
                           需为合法概率分布（非负且和为1）
    返回:
        KL散度标量张量
    """
    # 对预测分布做log_softmax确保合法对数概率，真实分布加小常数避免log(0)
    pred_log = F.log_softmax(pred_tempo_dist, dim=-1)
    target_prob = target_tempo_dist + 1e-8
    target_prob = target_prob / target_prob.sum(dim=-1, keepdim=True)
    loss = F.kl_div(pred_log, target_prob, reduction='batchmean')
    print(f"[节奏] tempo_kl = {loss.item():.6f}")
    return loss


# ==============================================================================
# 三、装饰音类损失 (Ornament Loss)
# ==============================================================================

def ornament_score_MSE(
    pred_ornament: torch.Tensor,
    target_ornament: torch.Tensor,
    pred_pitch: torch.Tensor,
    target_pitch: torch.Tensor,
    pred_duration: torch.Tensor,
    target_duration: torch.Tensor,
) -> torch.Tensor:
    """
    装饰音综合评分MSE损失 (ORS)

    按照NOEF装饰音评分公式，ORS由4个子指标加权组合：
      - 旋律融合度 (权重0.4)：装饰音与主旋律音高的协调程度
      - 节奏适配度 (权重0.25)：装饰音时值与周围音符时值的匹配程度
      - 装饰音密度 (权重0.2)：装饰音出现频率的合理性
      - 均匀度 (权重0.15)：装饰音分布的均匀程度

    文献来源：
        NanyinHGNN (arXiv:2510.26817)
        该文构建南音异构图神经网络，对南音装饰音进行建模与评估，
        提出装饰音综合评分ORS，包含旋律融合、节奏适配、密度、均匀度四个子维度。

    参数:
        pred_ornament:   模型预测装饰音标记, shape=(batch, seq_len), 值域[0,1]
        target_ornament: 真实装饰音标记, shape=(batch, seq_len)
        pred_pitch:      模型预测音高, shape=(batch, seq_len)
        target_pitch:    真实音高, shape=(batch, seq_len)
        pred_duration:   模型预测时值, shape=(batch, seq_len)
        target_duration: 真实时值, shape=(batch, seq_len)
    返回:
        ORS综合MSE损失标量张量
    """
    # 统一维度
    if pred_pitch.dim() == 3:
        pred_pitch = pred_pitch.squeeze(-1)
    if target_pitch.dim() == 3:
        target_pitch = target_pitch.squeeze(-1)
    if pred_duration.dim() == 3:
        pred_duration = pred_duration.squeeze(-1)
    if target_duration.dim() == 3:
        target_duration = target_duration.squeeze(-1)

    # ---- 子指标1: 旋律融合度 (权重0.4) ----
    # 装饰音位置的音高应与周围主旋律音高协调
    # 计算装饰音位置音高均值与全局音高均值的比率
    pred_ornament_mask = pred_ornament    # (batch, seq_len)
    target_ornament_mask = target_ornament

    # 防爆：pred_pitch 为无约束线性输出，训练初期均值可能接近 0，
    # 若用 clamp(1e-8)，融合比 pred_orn_pitch/mean 会爆炸到 1e8+（MusicMamba 首轮 6e13），
    # 导致梯度爆炸、训练崩溃。南音音高为 MIDI 值（典型 20~110），均值下限取 1.0 即可防爆，
    # 且训练后期 pred_pitch 均值远大于 1，此 clamp 不改变正常训练路径的 loss。
    pred_global_pitch_mean = pred_pitch.mean(dim=-1, keepdim=True).clamp(min=1.0)
    target_global_pitch_mean = target_pitch.mean(dim=-1, keepdim=True).clamp(min=1.0)

    pred_orn_pitch = (pred_pitch * pred_ornament_mask).sum(dim=-1, keepdim=True) / \
                     pred_ornament_mask.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    target_orn_pitch = (target_pitch * target_ornament_mask).sum(dim=-1, keepdim=True) / \
                       target_ornament_mask.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    pred_fusion = pred_orn_pitch / pred_global_pitch_mean
    target_fusion = target_orn_pitch / target_global_pitch_mean
    fusion_loss = F.mse_loss(pred_fusion, target_fusion)

    # ---- 子指标2: 节奏适配度 (权重0.25) ----
    # 装饰音时值应与周围音符时值匹配
    pred_orn_dur = (pred_duration * pred_ornament_mask).sum(dim=-1, keepdim=True) / \
                   pred_ornament_mask.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    target_orn_dur = (target_duration * target_ornament_mask).sum(dim=-1, keepdim=True) / \
                     target_ornament_mask.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # 防爆：pred_duration = softplus(线性输出)，头部输出大负值时 softplus 趋于 0，
    # 同样会导致节奏适配比除零爆炸，下限取 1e-3 足够安全。
    pred_global_dur_mean = pred_duration.mean(dim=-1, keepdim=True).clamp(min=1e-3)
    target_global_dur_mean = target_duration.mean(dim=-1, keepdim=True).clamp(min=1e-3)

    pred_rhythm_fit = pred_orn_dur / pred_global_dur_mean
    target_rhythm_fit = target_orn_dur / target_global_dur_mean
    rhythm_fit_loss = F.mse_loss(pred_rhythm_fit, target_rhythm_fit)

    # ---- 子指标3: 装饰音密度 (权重0.2) ----
    # 装饰音出现频率的合理性
    pred_density = pred_ornament_mask.mean(dim=-1)   # (batch,)
    target_density = target_ornament_mask.mean(dim=-1)
    density_loss = F.mse_loss(pred_density, target_density)

    # ---- 子指标4: 均匀度 (权重0.15) ----
    # 装饰音分布的均匀程度，用滑动窗口内装饰音计数的标准差衡量
    batch_size = pred_ornament.shape[0]
    seq_len = pred_ornament.shape[1]

    pred_uniformity = torch.zeros(batch_size, device=pred_ornament.device)
    target_uniformity = torch.zeros(batch_size, device=target_ornament.device)

    window_size = max(1, seq_len // 8)
    for b in range(batch_size):
        pred_counts = []
        target_counts = []
        for start in range(0, seq_len - window_size + 1, window_size):
            pred_counts.append(pred_ornament_mask[b, start:start + window_size].sum())
            target_counts.append(target_ornament_mask[b, start:start + window_size].sum())
        if len(pred_counts) > 1:
            pred_counts_t = torch.stack(pred_counts)
            target_counts_t = torch.stack(target_counts)
            # 均匀度 = 1 / (1 + 标准差)，分布越均匀值越大
            pred_uniformity[b] = 1.0 / (1.0 + pred_counts_t.std())
            target_uniformity[b] = 1.0 / (1.0 + target_counts_t.std())
        else:
            pred_uniformity[b] = 1.0
            target_uniformity[b] = 1.0

    uniformity_loss = F.mse_loss(pred_uniformity, target_uniformity)

    # ---- 加权组合 ----
    loss = (0.4 * fusion_loss +
            0.25 * rhythm_fit_loss +
            0.2 * density_loss +
            0.15 * uniformity_loss)

    print(f"[装饰音] ornament_score_MSE = {loss.item():.6f} "
          f"(融合={fusion_loss.item():.4f}, 节奏适配={rhythm_fit_loss.item():.4f}, "
          f"密度={density_loss.item():.4f}, 均匀度={uniformity_loss.item():.4f})")
    return loss


def ornament_density_MSE(pred_ornament: torch.Tensor, target_ornament: torch.Tensor) -> torch.Tensor:
    """
    装饰音密度MSE损失

    计算预测装饰音密度与真实装饰音密度之间的均方误差，
    衡量装饰音出现频率是否合理，值越小表示密度越接近真实。

    文献来源：
        NanyinHGNN (arXiv:2510.26817)
        该文在南音装饰音评估中，将装饰音密度作为独立评估维度，
        衡量生成音乐中装饰音使用的频率是否与南音传统风格一致。

    参数:
        pred_ornament:   模型预测装饰音标记, shape=(batch, seq_len), 值域[0,1]
        target_ornament: 真实装饰音标记, shape=(batch, seq_len)
    返回:
        MSE损失标量张量
    """
    loss = F.mse_loss(pred_ornament, target_ornament)
    print(f"[装饰音] ornament_density_MSE = {loss.item():.6f}")
    return loss


# ==============================================================================
# 四、调式类损失 (Mode Loss)
# ==============================================================================

# 南音4种专属调式名称
NANYIN_FOUR_MODES = ["五空管", "五空四仪管", "倍思管", "四空管"]


def mode_kl(pred_mode_logits: torch.Tensor, target_mode_dist: torch.Tensor) -> torch.Tensor:
    """
    调式KL散度损失

    计算预测调式分布与真实调式分布之间的KL散度，
    适配南音4种专属调式（五空管、五空四仪管、倍思管、四空管），
    衡量生成音乐调式归属与真实调式分布的匹配程度。

    文献来源：
        NanyinHGNN (arXiv:2510.26817)
        该文构建南音异构图神经网络，对南音调式进行分类建模，
        使用KL散度约束调式分布预测与真实调式标签的一致性。

    参数:
        pred_mode_logits:  模型预测调式logits, shape=(batch, 4)
                           对应南音4种调式的未归一化得分
        target_mode_dist:  真实调式分布, shape=(batch, 4)
                           one-hot或soft标签，需为合法概率分布
    返回:
        KL散度标量张量
    """
    # 对预测logits做log_softmax
    pred_log = F.log_softmax(pred_mode_logits, dim=-1)
    # 真实分布加小常数并归一化
    target_prob = target_mode_dist + 1e-8
    target_prob = target_prob / target_prob.sum(dim=-1, keepdim=True)
    loss = F.kl_div(pred_log, target_prob, reduction='batchmean')
    print(f"[调式] mode_kl = {loss.item():.6f} (南音四调式: {NANYIN_FOUR_MODES})")
    return loss


# ==============================================================================
# 五、风格类损失 (Style Loss)
# ==============================================================================

def fad_diff(
    pred_features: torch.Tensor,
    target_features: torch.Tensor,
) -> torch.Tensor:
    """
    可微FAD近似损失 (Fréchet Audio Distance Differentiable Approximation)

    FAD用于衡量生成音频特征分布与真实音频特征分布之间的距离。
    原始FAD基于Fréchet距离（Wasserstein-2距离），需要计算特征均值和协方差。
    本函数实现可微近似版本，使用特征均值差的平方加上协方差迹的差作为近似，
    全部操作可微分，支持反向传播。

    文献来源：
        KAD (arXiv:2502.15602)
        该文提出KAD（Kernel Audio Distance）作为FAD的改进指标，
        用于评估生成音频与真实音频的风格距离。本函数借鉴其思想，
        使用可微近似计算特征分布距离。

    参数:
        pred_features:   模型生成音频特征, shape=(batch, feat_dim)
                         通常为预训练特征提取器的输出
        target_features: 真实音频特征, shape=(batch, feat_dim)
    返回:
        可微FAD近似损失标量张量
    """
    # 计算均值向量
    pred_mean = pred_features.mean(dim=0)       # (feat_dim,)
    target_mean = target_features.mean(dim=0)   # (feat_dim,)

    # 均值差的平方
    mean_diff_sq = ((pred_mean - target_mean) ** 2).sum()

    # 计算协方差矩阵的可微近似
    # pred协方差
    pred_centered = pred_features - pred_mean.unsqueeze(0)
    target_centered = target_features - target_mean.unsqueeze(0)

    n_pred = pred_features.shape[0]
    n_target = target_features.shape[0]

    pred_cov = (pred_centered.t() @ pred_centered) / max(n_pred - 1, 1)
    target_cov = (target_centered.t() @ target_centered) / max(n_target - 1, 1)

    # Fréchet距离近似 = ||μ1-μ2||^2 + Tr(C1 + C2 - 2*sqrt(C1*C2))
    # 可微近似：使用 ||μ1-μ2||^2 + Tr(C1) + Tr(C2) - 2*Tr(sqrt_approx)
    # 其中sqrt_approx使用矩阵平方根的泰勒近似
    # 简化可微版本：使用迹的差作为协方差项的近似
    cov_trace_diff = torch.abs(pred_cov.trace() - target_cov.trace())

    loss = mean_diff_sq + cov_trace_diff
    print(f"[风格] fad_diff = {loss.item():.6f}")
    return loss


def style_cosine_loss(pred_style: torch.Tensor, target_style: torch.Tensor) -> torch.Tensor:
    """
    风格余弦相似度损失

    计算预测风格向量与目标风格向量之间的余弦距离（1 - 余弦相似度），
    衡量生成音乐整体风格向量与真实风格向量的方向一致性，
    值越小表示风格方向越一致。

    文献来源：
        KAD (arXiv:2502.15602)
        该文在音频风格评估中使用余弦相似度衡量风格嵌入的一致性，
        本函数采用1-cosine作为损失，使优化目标与评估目标一致。

    参数:
        pred_style:   模型预测风格嵌入向量, shape=(batch, style_dim)
        target_style: 真实风格嵌入向量, shape=(batch, style_dim)
    返回:
        余弦距离损失标量张量
    """
    # F.cosine_similarity返回的是相似度，范围[-1,1]，1表示完全一致
    # 损失 = 1 - 相似度，范围[0,2]，0表示完全一致
    cosine_sim = F.cosine_similarity(pred_style, target_style, dim=-1)
    loss = (1.0 - cosine_sim).mean()
    print(f"[风格] style_cosine_loss = {loss.item():.6f}")
    return loss


# ==============================================================================
# 六、总损失封装
# ==============================================================================

def nanyin_total_loss(
    pred_tokens: torch.Tensor,
    target_dict: dict,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    lambda3: float = 1.0,
    lambda4: float = 1.0,
    lambda5: float = 1.0,
) -> torch.Tensor:
    """
    南音NOEF可微分复合总损失

    总损失 = 序列分类交叉熵 + λ1*音高损失 + λ2*节奏损失 + λ3*装饰音损失 + λ4*调式损失 + λ5*风格损失

    其中各维度损失为对应子函数的加总：
      - 音高损失 = pitch_accuracy_MSE + pitch_stability_L1
      - 节奏损失 = rhythm_consistency_MSE + tempo_kl
      - 装饰音损失 = ornament_score_MSE + ornament_density_MSE
      - 调式损失 = mode_kl
      - 风格损失 = fad_diff + style_cosine_loss

    参数:
        pred_tokens:  模型预测token logits, shape=(batch, seq_len, vocab_size)
                      用于计算序列分类交叉熵
        target_dict:  真值标注字典，需包含以下键：
            - "target_tokens":    目标token索引, shape=(batch, seq_len), dtype=long
            - "target_pitch":     真实音高, shape=(batch, seq_len)
            - "pred_pitch":       预测音高, shape=(batch, seq_len)
            - "target_duration":  真实时值, shape=(batch, seq_len)
            - "pred_duration":    预测时值, shape=(batch, seq_len)
            - "target_tempo_dist": 真实速度分布, shape=(batch, num_tempo_bins)
            - "pred_tempo_dist":   预测速度分布, shape=(batch, num_tempo_bins)
            - "target_ornament":  真实装饰音标记, shape=(batch, seq_len)
            - "pred_ornament":    预测装饰音标记, shape=(batch, seq_len)
            - "target_mode_dist": 真实调式分布, shape=(batch, 4)
            - "pred_mode_logits": 预测调式logits, shape=(batch, 4)
            - "target_features":  真实音频特征, shape=(batch, feat_dim)
            - "pred_features":    预测音频特征, shape=(batch, feat_dim)
            - "target_style":     真实风格嵌入, shape=(batch, style_dim)
            - "pred_style":       预测风格嵌入, shape=(batch, style_dim)
        lambda1: 音高损失权重
        lambda2: 节奏损失权重
        lambda3: 装饰音损失权重
        lambda4: 调式损失权重
        lambda5: 风格损失权重
    返回:
        总损失标量张量（可微分，支持反向传播）
    """
    # ---- 1. 序列分类交叉熵 ----
    # pred_tokens: (batch, seq_len, vocab_size) -> 展平后计算交叉熵
    vocab_size = pred_tokens.shape[-1]
    pred_flat = pred_tokens.reshape(-1, vocab_size)          # (batch*seq_len, vocab_size)
    target_flat = target_dict["target_tokens"].reshape(-1)   # (batch*seq_len,)
    ce_loss = F.cross_entropy(pred_flat, target_flat)
    print(f"[总损失] 序列交叉熵 = {ce_loss.item():.6f}")

    # ---- 2. 音高损失 ----
    pitch_loss = (pitch_accuracy_MSE(target_dict["pred_pitch"], target_dict["target_pitch"]) +
                  pitch_stability_L1(target_dict["pred_pitch"], target_dict["target_pitch"]))
    print(f"[总损失] 音高损失(λ1={lambda1}) = {pitch_loss.item():.6f}")

    # ---- 3. 节奏损失 ----
    rhythm_loss = (rhythm_consistency_MSE(target_dict["pred_duration"], target_dict["target_duration"]) +
                   tempo_kl(target_dict["pred_tempo_dist"], target_dict["target_tempo_dist"]))
    print(f"[总损失] 节奏损失(λ2={lambda2}) = {rhythm_loss.item():.6f}")

    # ---- 4. 装饰音损失 ----
    ornament_loss = (
        ornament_score_MSE(
            pred_ornament=target_dict["pred_ornament"],
            target_ornament=target_dict["target_ornament"],
            pred_pitch=target_dict["pred_pitch"],
            target_pitch=target_dict["target_pitch"],
            pred_duration=target_dict["pred_duration"],
            target_duration=target_dict["target_duration"],
        ) +
        ornament_density_MSE(target_dict["pred_ornament"], target_dict["target_ornament"])
    )
    print(f"[总损失] 装饰音损失(λ3={lambda3}) = {ornament_loss.item():.6f}")

    # ---- 5. 调式损失 ----
    mode_loss = mode_kl(target_dict["pred_mode_logits"], target_dict["target_mode_dist"])
    print(f"[总损失] 调式损失(λ4={lambda4}) = {mode_loss.item():.6f}")

    # ---- 6. 风格损失 ----
    style_loss = (fad_diff(target_dict["pred_features"], target_dict["target_features"]) +
                  style_cosine_loss(target_dict["pred_style"], target_dict["target_style"]))
    print(f"[总损失] 风格损失(λ5={lambda5}) = {style_loss.item():.6f}")

    # ---- 7. 加权总损失 ----
    total = (ce_loss +
             lambda1 * pitch_loss +
             lambda2 * rhythm_loss +
             lambda3 * ornament_loss +
             lambda4 * mode_loss +
             lambda5 * style_loss)

    print(f"=" * 60)
    print(f"[总损失] TotalLoss = {total.item():.6f}")
    print(f"  CE={ce_loss.item():.4f} + "
          f"λ1*Pitch={lambda1}*{pitch_loss.item():.4f} + "
          f"λ2*Rhythm={lambda2}*{rhythm_loss.item():.4f} + "
          f"λ3*Ornament={lambda3}*{ornament_loss.item():.4f} + "
          f"λ4*Mode={lambda4}*{mode_loss.item():.4f} + "
          f"λ5*Style={lambda5}*{style_loss.item():.4f}")
    print(f"=" * 60)

    return total
