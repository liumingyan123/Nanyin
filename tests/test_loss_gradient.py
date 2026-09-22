# -*- coding: utf-8 -*-
"""
tests/test_loss_gradient.py — E3 分布损失梯度单元测试
=====================================================

校验目标（E3 硬性要求：梯度测试未通过禁止启动任何消融训练）：

  G0 模型不再包含独立辅助头 features_head / style_head（参数与前向均已移除）
  G1 音级(pitch-class)分布损失  → pitch_head    存在非零梯度
  G2 时值(duration)分布损失     → duration_head 存在非零梯度
  G3 IOI(time-shift)分布损失    → shift_head    存在非零梯度
  G4 调式(mode)分布损失         → pitch_head    存在非零梯度
  G5 装饰音密度损失（L5 框架）  → duration_head 存在非零梯度（框架校验，不纳入正式消融）

每项检查还额外验证"不该有梯度的头梯度为 0"，确保损失确实只作用于目标生成头；
并打印每项损失对应各网络头的梯度范数日志。

运行：
    venv/Scripts/python.exe tests/test_loss_gradient.py

产物：
    results/e3_gradient_test.json   （梯度范数明细，被 .gitignore 排除）
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (_PROJECT_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from nanyin_config import load_config, E2_CONFIG_PATH
from models import NanyinBoYaTCN
from dataset_nanyin import (
    IGNORE_INDEX,
    build_window_sample,
    get_compound_codec,
    get_compound_window_datasets,
    resolve_window_mode_index,
)
from loss_nanyin import (
    build_pitch_class_matrix,
    build_value_bin_matrix,
    build_mode_distribution_matrix,
    pitch_class_distribution_loss,
    mode_distribution_loss,
    duration_distribution_loss,
    shift_ioi_distribution_loss,
    ornament_density_distribution_loss,
)

HEADS = ("pitch_head", "duration_head", "shift_head")
GRAD_EPS = 0.0        # 梯度范数判"非零"的阈值（严格 > 0）
ZERO_EPS = 1e-12      # 判"零梯度"的容差


# ==============================================================================
# 工具
# ==============================================================================

class Checker:
    """收集 PASS / FAIL 判定项。"""

    def __init__(self):
        self.items: List[Dict[str, Any]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.items.append({"name": name, "pass": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        return bool(ok)

    @property
    def all_pass(self) -> bool:
        return all(it["pass"] for it in self.items)

    def summary_line(self) -> str:
        n_pass = sum(1 for it in self.items if it["pass"])
        return f"{n_pass}/{len(self.items)} 项通过"


def make_batch(cfg: dict, num_windows: int = 2) -> Dict[str, torch.Tensor]:
    """取真实验证集窗口拼一个 batch（保证 target_mode_index 等字段来自真实数据链路）。"""
    train_ds, val_ds = get_compound_window_datasets(cfg=cfg, verbose=False)
    samples = [val_ds[i] for i in range(min(num_windows, len(val_ds)))]
    batch: Dict[str, Any] = {}
    for key in ("input_pitch_ids", "input_duration_ids", "input_shift_ids",
                "target_pitch_ids", "target_duration_ids", "target_shift_ids",
                "attention_mask", "target_mode_index"):
        batch[key] = torch.stack([s[key] for s in samples], dim=0)
    batch["_num_windows"] = len(samples)
    return batch


def grad_norms(model: NanyinBoYaTCN) -> Dict[str, float]:
    """读取三个生成头的权重梯度 L2 范数（无梯度记 0.0）。"""
    out = {}
    for name in HEADS:
        param = getattr(model, name).weight
        out[name] = float(param.grad.norm().item()) if param.grad is not None else 0.0
    return out


def zero_grads(model: NanyinBoYaTCN) -> None:
    model.zero_grad(set_to_none=True)


def run_case(
    model: NanyinBoYaTCN,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    case_name: str,
    loss_fn,
    expect_head: str,
    checker: Checker,
    records: List[Dict[str, Any]],
) -> None:
    """对单项损失执行 forward + backward，并校验目标头梯度非零、其余头为 0。"""
    zero_grads(model)
    pred = model(
        batch["input_pitch_ids"].to(device),
        batch["input_duration_ids"].to(device),
        batch["input_shift_ids"].to(device),
    )
    loss = loss_fn(pred, batch)
    loss.backward()
    norms = grad_norms(model)

    print(f"\n[{case_name}] loss = {float(loss.detach()):.6f}")
    for head in HEADS:
        print(f"    grad_norm {head:<14} = {norms[head]:.6e}")

    target_ok = norms[expect_head] > GRAD_EPS
    checker.add(f"{case_name} → {expect_head} 梯度非零", target_ok,
                f"grad_norm={norms[expect_head]:.6e}")

    others = [h for h in HEADS if h != expect_head]
    others_zero = all(norms[h] <= ZERO_EPS for h in others)
    checker.add(f"{case_name} → 其余生成头梯度为 0", others_zero,
                ", ".join(f"{h}={norms[h]:.3e}" for h in others))

    records.append({
        "case": case_name,
        "loss": float(loss.detach()),
        "grad_norms": norms,
        "expect_head": expect_head,
        "target_nonzero": bool(target_ok),
        "others_zero": bool(others_zero),
    })
    zero_grads(model)


# ==============================================================================
# 主流程
# ==============================================================================

def main() -> int:
    cfg = load_config(E2_CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checker = Checker()
    records: List[Dict[str, Any]] = []

    print("=" * 74)
    print("E3 分布损失梯度单元测试（未通过禁止启动消融训练）")
    print("=" * 74)
    print(f"设备={device}, 配置={E2_CONFIG_PATH}")

    codec = get_compound_codec(cfg)
    model = NanyinBoYaTCN().to(device)
    # 注意：必须保持 train 模式 —— BiLSTM 的 cuDNN 反向要求 training mode
    model.train()

    # ---------------- G0：独立辅助头已停用 ----------------
    print("\n[G0] 独立辅助头 features_head / style_head 停用校验")
    has_feat = model.features_head is not None
    has_style = model.style_head is not None
    param_names = [n for n, _ in model.named_parameters()]
    feat_params = [n for n in param_names if "features_head" in n or "style_head" in n]
    checker.add("G0 features_head 已停用（模块为 None）", not has_feat)
    checker.add("G0 style_head 已停用（模块为 None）", not has_style)
    checker.add("G0 参数表中无 features_head/style_head 参数", len(feat_params) == 0,
                f"匹配到 {len(feat_params)} 个")
    pred_probe = model(
        torch.zeros(1, 8, dtype=torch.long, device=device),
        torch.zeros(1, 8, dtype=torch.long, device=device),
        torch.zeros(1, 8, dtype=torch.long, device=device),
    )
    checker.add("G0 forward 不再返回 pred_features / pred_style",
                pred_probe.get("pred_features") is None and pred_probe.get("pred_style") is None)
    checker.add("G0 三个生成头全部存在",
                all(hasattr(model, h) for h in HEADS))

    # ---------------- 准备真实 batch 与聚合矩阵 ----------------
    batch = make_batch(cfg, num_windows=2)
    print(f"\n[数据] 验证集真实窗口 batch：B={batch['_num_windows']}, "
          f"L={batch['target_pitch_ids'].shape[1]}, "
          f"target_mode_index={batch['target_mode_index'].tolist()}")

    dcfg = cfg.get("distribution", {})
    # E2 配置无 distribution 小节时，回退 L0 配置中的同一份分箱真源
    if not dcfg:
        dcfg = load_config(os.path.join(_PROJECT_ROOT, "configs", "e3_L0.yaml"))["distribution"]
    pc_matrix = build_pitch_class_matrix(int(codec.pitch_vocab_size)).to(device)
    mode_matrix = build_mode_distribution_matrix().to(device)
    dur_matrix = build_value_bin_matrix(
        [codec.index_to_duration(i) for i in range(int(codec.duration_num_classes))],
        dcfg["duration_bin_edges"],
    ).to(device)
    shift_matrix = build_value_bin_matrix(
        [codec.index_to_shift(i) for i in range(int(codec.shift_num_classes))],
        dcfg["shift_ioi_bin_edges"],
    ).to(device)
    short_mask = torch.tensor(
        [codec.index_to_duration(i) <= float(dcfg.get("ornament_short_duration_sec", 0.15))
         for i in range(int(codec.duration_num_classes))],
        dtype=torch.bool, device=device,
    )
    print(f"[矩阵] pc={tuple(pc_matrix.shape)}, mode={tuple(mode_matrix.shape)}, "
          f"duration={tuple(dur_matrix.shape)}, shift={tuple(shift_matrix.shape)}, "
          f"短音类别数={int(short_mask.sum())}")

    num_pitch_real = int(codec.pitch_vocab_size)
    num_dur_real = int(codec.duration_num_classes)
    num_shift_real = int(codec.shift_num_classes)

    def _pitch_valid(pred, b):
        return b["target_pitch_ids"].to(device) < num_pitch_real

    # ---------------- G1：音级分布损失 → pitch_head ----------------
    def loss_pc(pred, b):
        return pitch_class_distribution_loss(
            pred["pred_pitch_logits"], b["target_pitch_ids"].to(device),
            pc_matrix=pc_matrix, valid_mask=_pitch_valid(pred, b),
        )

    run_case(model, batch, device, "G1 音级(pitch-class)分布损失", loss_pc,
             "pitch_head", checker, records)

    # ---------------- G2：时值分布损失 → duration_head ----------------
    def loss_dur(pred, b):
        return duration_distribution_loss(
            pred["pred_duration_logits"], b["target_duration_ids"].to(device),
            bin_matrix=dur_matrix,
        )

    run_case(model, batch, device, "G2 时值(duration)分布损失", loss_dur,
             "duration_head", checker, records)

    # ---------------- G3：IOI 分布损失 → shift_head ----------------
    def loss_ioi(pred, b):
        tgt = b["target_shift_ids"].to(device)
        return shift_ioi_distribution_loss(
            pred["pred_shift_logits"], tgt, bin_matrix=shift_matrix,
            valid_mask=tgt != IGNORE_INDEX,
        )

    run_case(model, batch, device, "G3 IOI(time-shift)分布损失", loss_ioi,
             "shift_head", checker, records)

    # ---------------- G4：调式损失 → pitch_head ----------------
    def loss_mode(pred, b):
        return mode_distribution_loss(
            pred["pred_pitch_logits"], b["target_mode_index"].to(device),
            pc_matrix=pc_matrix, mode_dist_matrix=mode_matrix,
            valid_mask=_pitch_valid(pred, b),
        )

    run_case(model, batch, device, "G4 调式(mode)符合度损失", loss_mode,
             "pitch_head", checker, records)

    # ---------------- G5：装饰音密度损失（L5 框架）→ duration_head ----------------
    def loss_orn(pred, b):
        tgt_dur = b["target_duration_ids"].to(device)
        valid = tgt_dur < num_dur_real
        short_target = short_mask[tgt_dur.clamp(min=0, max=num_dur_real - 1)]
        return ornament_density_distribution_loss(
            pred["pred_duration_logits"], short_target,
            short_class_mask=short_mask, valid_mask=valid,
        )

    run_case(model, batch, device, "G5 装饰音密度损失（L5 框架，不纳入正式消融）", loss_orn,
             "duration_head", checker, records)

    # ---------------- 附加：掩码有效性 ----------------
    print("\n[G6] 掩码与真值链路附加校验")
    tgt_shift = batch["target_shift_ids"]
    masked = int((tgt_shift == IGNORE_INDEX).sum())
    checker.add("G6 shift 目标含 IGNORE_INDEX 掩码位（窗口尾伪标签 + EOS）", masked >= 2,
                f"掩码位数={masked}")
    # 掩码位不参与 IOI 直方图：把掩码位改成合法类别，损失必须变化
    def loss_ioi_nomask(pred, b):
        tgt = b["target_shift_ids"].to(device).clone()
        tgt[tgt == IGNORE_INDEX] = 1        # 强行把伪标签位当作真实类别
        return shift_ioi_distribution_loss(
            pred["pred_shift_logits"], tgt, bin_matrix=shift_matrix,
            valid_mask=torch.ones_like(tgt, dtype=torch.bool),
        )

    zero_grads(model)
    pred_nm = model(
        batch["input_pitch_ids"].to(device),
        batch["input_duration_ids"].to(device),
        batch["input_shift_ids"].to(device),
    )
    l_mask = float(loss_ioi(pred_nm, batch).detach())
    l_nomask = float(loss_ioi_nomask(pred_nm, batch).detach())
    checker.add("G6 去掉伪标签掩码后 IOI 损失发生变化（证明掩码确实生效）",
                abs(l_mask - l_nomask) > 1e-8,
                f"带掩码={l_mask:.6f}, 不带掩码={l_nomask:.6f}")
    # 调式真值解析可用
    mode_idx = resolve_window_mode_index([60, 62, 64, 67, 69])
    checker.add("G6 调式真值解析可用（五空管音级 → 0）", mode_idx == 0, f"mode_index={mode_idx}")

    # ---------------- 汇总 ----------------
    out_path = os.path.join(_PROJECT_ROOT, "results", "e3_gradient_test.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "all_pass": checker.all_pass,
            "summary": checker.summary_line(),
            "device": str(device),
            "cases": records,
            "items": checker.items,
        }, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 74)
    print(f"梯度测试汇总: {checker.summary_line()}；明细已写入 {out_path}")
    print("=" * 74)
    if not checker.all_pass:
        print("[FAIL] 存在梯度为 0（或掩码失效）的项目 -> 按实验规程禁止启动消融训练。")
        return 1
    print("[OK] 全部损失项梯度有效，可以启动 L0-L4 消融训练。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
