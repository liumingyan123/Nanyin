# -*- coding: utf-8 -*-
"""
plot_loss_curve.py - 训练/验证损失曲线作图
=========================================

读取训练脚本产出的逐轮损失 CSV（如 ``results/e2_boyatcn_train_log.csv``），
输出损失曲线图（如 ``reports/E2_loss_curve.png``）。

图面：
  左图：train_total / val_total 总损失曲线 + best-val 轮次竖线标注；
  右图：三路分项 CE（pitch / duration / shift），train 实线、val 虚线。

用法：
    venv/Scripts/python.exe plot_loss_curve.py
    venv/Scripts/python.exe plot_loss_curve.py --csv results/xxx.csv --out reports/xxx.png
"""

import argparse
import csv
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------------------------------------------------------------------
# 中文字体（与 visualize_embedding.py 一致：Windows 微软雅黑，找不到则退回默认）
# ---------------------------------------------------------------------------
_CN_FONT: Optional[str] = None
for _fp in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc",
            r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(_fp):
        try:
            font_manager.fontManager.addfont(_fp)
            _CN_FONT = font_manager.FontProperties(fname=_fp).get_name()
            break
        except Exception:
            pass
if _CN_FONT:
    plt.rcParams["font.family"] = _CN_FONT
    plt.rcParams["axes.unicode_minus"] = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(_SCRIPT_DIR, "results", "e2_boyatcn_train_log.csv")
DEFAULT_OUT = os.path.join(_SCRIPT_DIR, "reports", "E2_loss_curve.png")

# 三路分项的配色（train / val 同色，靠线型区分）
_COMPONENTS = (
    ("pitch", "音高 pitch", "#1f77b4"),
    ("duration", "时值 duration", "#d62728"),
    ("shift", "时间偏移 time-shift", "#2ca02c"),
)


def read_history(csv_path: str) -> List[Dict[str, float]]:
    """读取逐轮损失 CSV（数值列转 float，is_best 转 bool）。"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[plot_loss_curve] 损失 CSV 不存在: {csv_path}")
    rows: List[Dict[str, float]] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        for raw in csv.DictReader(f):
            rows.append({
                "epoch": int(float(raw["epoch"])),
                "train_total": float(raw["train_total"]),
                "train_pitch_ce": float(raw["train_pitch_ce"]),
                "train_duration_ce": float(raw["train_duration_ce"]),
                "train_shift_ce": float(raw["train_shift_ce"]),
                "val_total": float(raw["val_total"]),
                "val_pitch_ce": float(raw["val_pitch_ce"]),
                "val_duration_ce": float(raw["val_duration_ce"]),
                "val_shift_ce": float(raw["val_shift_ce"]),
                "is_best": str(raw.get("is_best", "")).strip().lower() == "true",
            })
    if not rows:
        raise ValueError(f"[plot_loss_curve] 损失 CSV 无数据行: {csv_path}")
    return rows


def plot(csv_path: str, out_path: str, title: str = "E2 复合事件窗口训练损失曲线") -> str:
    rows = read_history(csv_path)
    epochs = [r["epoch"] for r in rows]
    best_rows = [r for r in rows if r["is_best"]]
    best_epoch = best_rows[-1]["epoch"] if best_rows else None
    best_val = best_rows[-1]["val_total"] if best_rows else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ---------------- 左图：总损失 train / val ----------------
    ax1.plot(epochs, [r["train_total"] for r in rows],
             color="#1f77b4", lw=2.0, marker="o", ms=3.5, label="train_total（训练集）")
    ax1.plot(epochs, [r["val_total"] for r in rows],
             color="#d62728", lw=2.0, marker="s", ms=3.5, label="val_total（验证集）")
    if best_epoch is not None:
        ax1.axvline(best_epoch, color="#2ca02c", ls="--", lw=1.5,
                    label=f"best-val @ epoch {best_epoch}（val={best_val:.4f}）")
        ax1.scatter([best_epoch], [best_val], color="#2ca02c", s=70, zorder=5)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("total loss（1.0·pitch_CE + 0.7·duration_CE + 0.7·shift_CE）")
    ax1.set_title("总损失：训练集持续下降，验证集第 %s 轮后回升（过拟合）"
                  % (best_epoch if best_epoch is not None else "?"))
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_xlim(left=0)

    # ---------------- 右图：三路分项 CE ----------------
    for key, label, color in _COMPONENTS:
        ax2.plot(epochs, [r[f"train_{key}_ce"] for r in rows],
                 color=color, lw=1.8, ls="-", label=f"{label} · train")
        ax2.plot(epochs, [r[f"val_{key}_ce"] for r in rows],
                 color=color, lw=1.8, ls="--", alpha=0.85, label=f"{label} · val")
    if best_epoch is not None:
        ax2.axvline(best_epoch, color="#2ca02c", ls="--", lw=1.2, alpha=0.8)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("CE（未加权）")
    ax2.set_title("三路分项交叉熵（实线=train，虚线=val）")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8, ncol=2)
    ax2.set_xlim(left=0)

    fig.suptitle(f"{title}｜共 {len(rows)} 轮（早停 patience=20）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="训练/验证损失曲线作图")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="逐轮损失 CSV 路径")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT, help="输出 PNG 路径")
    parser.add_argument("--title", type=str, default="E2 复合事件窗口训练损失曲线")
    args = parser.parse_args()

    out = plot(args.csv, args.out, args.title)
    print(f"[plot_loss_curve] 已输出: {out}")


if __name__ == "__main__":
    main()
