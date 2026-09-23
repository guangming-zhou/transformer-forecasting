#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把训练日志（`outputs/logs*/seed*.json`）里的 loss 曲线画出来对比。

用途：训练预算审计 —— 把「预算翻倍 + warmup」的那次运行和主结果叠在一起，
直观看出 train_loss 一直降、val_loss 却早已回头。

只读 JSON，不训练。日志文件本身是 gitignore 的运行产物，所以入库的是这张图。

用法
    python eval/plot_training_curves.py \
        --log "A (30 epoch, 早停) =outputs/logs/seed42.json" \
        --log "A60 (60 epoch + warmup) =outputs/_runs/budget/A60_warm5/logs/seed42.json" \
        --out docs/figs/budget_audit.png --title "训练预算审计（seed=42）"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="训练 loss 曲线对比")
    p.add_argument("--log", action="append", default=[], metavar="名称=日志路径",
                   help="可重复；名称会出现在图例里")
    p.add_argument("--out", default=os.path.join(PROJECT_ROOT, "docs", "figs", "budget_audit.png"))
    p.add_argument("--title", default="训练曲线对比")
    return p.parse_args(argv)


def split_pair(spec: str) -> Tuple[str, str]:
    name, sep, path = spec.partition("=")
    if not sep:
        raise SystemExit(f"--log 需为 名称=路径 形式，收到 {spec!r}")
    return name.strip(), path.strip()


def load_log(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if not args.log:
        raise SystemExit("至少给一个 --log 名称=路径")

    series: List[Tuple[str, Dict]] = []
    for spec in args.log:
        name, path = split_pair(spec)
        if not os.path.isfile(path):
            raise SystemExit(f"找不到日志: {path}")
        series.append((name, load_log(path)))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for i, (name, log) in enumerate(series):
        color = COLORS[i % len(COLORS)]
        hist = log["history"]
        epochs = [h["epoch"] for h in hist]
        axes[0].plot(epochs, [h["train_loss"] for h in hist], lw=1.7, color=color,
                     label=f"{name}  train")
        axes[1].plot(epochs, [h["val_loss"] for h in hist], lw=1.7, color=color,
                     label=f"{name}  val")
        # 标出 val 最优点（两条曲线的 best 挨得很近，注释按序号错开避免压字）
        best = log["best_epoch"]
        best_val = log["best_val"]
        axes[1].scatter([best], [best_val], s=70, marker="*", color=color, zorder=5)
        axes[1].annotate(f"best epoch {best}  val={best_val:.4f}", (best, best_val),
                         textcoords="offset points", xytext=(14, -6 - 26 * i),
                         fontsize=8, color=color,
                         arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8})

    axes[0].set_title("train loss（每轮都还在降）", fontsize=10)
    axes[1].set_title("val loss（很早就回头，之后再没回来）", fontsize=10)
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE（归一化空间）")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle(args.title, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"  已写出: {os.path.abspath(args.out)}")

    for name, log in series:
        print(f"  {name:<34s} epochs={len(log['history']):>3}  best_epoch={log['best_epoch']:>3}  "
              f"best_val={log['best_val']:.6f}  "
              f"train_loss(末)={log['history'][-1]['train_loss']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
