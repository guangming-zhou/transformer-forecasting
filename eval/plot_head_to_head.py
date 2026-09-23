#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 `eval/head_to_head.py` 写出的 JSON 画成两张图（README 首屏与 Main Results 用）。

左图：各预测器 overall MAE 的横向条形图（多 seed 带 ±std 误差棒），
      persistence 画成竖直参考线 —— 一眼看出谁在线的左边。
右图：分 horizon MAE 曲线（h1 / h6 / h12 / h24），看差距随步数怎么变化。

只读 JSON，不训练、不评估，秒级完成。

用法
    python eval/plot_head_to_head.py \
        --json outputs/head_to_head_24.json --out docs/figs/head_to_head.png
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
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

# 每个预测器的配色：(条形色, 曲线色)。按名字前缀匹配，未命中走默认灰。
COLOR_RULES: List[Tuple[str, str, str]] = [
    ("persistence", "#333333", "#333333"),
    ("ARIMA", "#7f7f7f", "#7f7f7f"),
    ("DLinear", "#2ca02c", "#2ca02c"),
    ("LSTM", "#1f77b4", "#1f77b4"),
    ("Transformer+", "#9467bd", "#9467bd"),
    ("Transformer+残差", "#9467bd", "#9467bd"),
    ("Transformer", "#d62728", "#d62728"),
]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="head-to-head JSON -> 对比图")
    p.add_argument("--json", default=os.path.join(PROJECT_ROOT, "outputs", "head_to_head_24.json"))
    p.add_argument("--out", default=os.path.join(PROJECT_ROOT, "docs", "figs", "head_to_head.png"))
    p.add_argument("--title", default=None)
    return p.parse_args(argv)


def colors_for(name: str) -> Tuple[str, str]:
    for prefix, bar, line in COLOR_RULES:
        if name.startswith(prefix):
            return bar, line
    return "#8c8c8c", "#8c8c8c"


def summarise(predictor: Dict) -> Tuple[float, float, Dict[str, float]]:
    """返回 (MAE mean, MAE std, {horizon: MAE})。确定性预测器的 std 记为 0。"""
    runs = predictor["metrics"]
    maes = [r["overall"]["MAE"] for r in runs]
    mean = st.mean(maes)
    std = st.stdev(maes) if len(maes) > 1 else 0.0
    per_h = {}
    for key in runs[0]:
        if len(key) > 1 and key[0] == "h" and key[1:].isdigit():
            per_h[key] = st.mean([r[key]["MAE"] for r in runs])
    return mean, std, per_h


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if not os.path.isfile(args.json):
        raise SystemExit(f"找不到 {args.json}；先跑 eval/head_to_head.py --out <同一路径>")

    with open(args.json, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    entries = []
    for p in payload["predictors"]:
        mean, std, per_h = summarise(p)
        n = len(p["metrics"])
        label = p["name"] if n <= 1 else f"{p['name']} ({n} seed)"
        entries.append({"name": p["name"], "label": label, "mean": mean, "std": std,
                        "per_h": per_h, "kind": p["kind"]})
    # 按 MAE 从好到差排序，条形图自上而下
    entries.sort(key=lambda e: e["mean"])

    pers = next((e for e in entries if e["name"] == "persistence"), None)
    horizons = sorted({k for e in entries for k in e["per_h"]}, key=lambda k: int(k[1:]))

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

    # ---------------- 左：overall MAE 条形图 ----------------
    ax = axes[0]
    ys = range(len(entries))
    ax.barh(list(ys), [e["mean"] for e in entries],
            xerr=[e["std"] for e in entries], height=0.62,
            color=[colors_for(e["name"])[0] for e in entries],
            error_kw={"ecolor": "#555555", "capsize": 3, "lw": 1.0}, alpha=0.9)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([e["label"] for e in entries], fontsize=9)
    for y, e in zip(ys, entries):
        ax.text(e["mean"] + max(e["std"], 0.02) + 0.02, y, f"{e['mean']:.4f}",
                va="center", fontsize=8.5)
    if pers is not None:
        ax.axvline(pers["mean"], color="#333333", ls="--", lw=1.3)
        # 标注放在最下方，避免和标题/最高那根柱子打架
        ax.text(pers["mean"] + 0.03, -1.0, f"persistence {pers['mean']:.4f}",
                color="#333333", fontsize=8.5, va="center")
    ax.set_xlabel("test MAE (°C)   ← 越靠左越好")
    ax.set_title("各预测器 overall MAE（误差棒 = ±1 std，多 seed）", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(-1.45, len(entries) - 0.4)

    # ---------------- 右：分 horizon MAE 曲线 ----------------
    ax2 = axes[1]
    xs = [int(h[1:]) for h in horizons]
    for e in entries:
        ax2.plot(xs, [e["per_h"][h] for h in horizons], marker="o", ms=4.5, lw=1.6,
                 color=colors_for(e["name"])[1], label=e["label"])
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"h{h}" for h in xs])
    ax2.set_xlabel("预测步（小时）")
    ax2.set_ylabel("MAE (°C)")
    ax2.set_title("误差随 horizon 的增长（MAE）", fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="upper left")

    fig.suptitle(args.title or payload.get("tag", "head-to-head"), fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"  已写出: {os.path.abspath(args.out)}")

    print("\n  图里用到的数字：")
    for e in entries:
        print(f"    {e['label']:<34s} MAE={e['mean']:.4f} ± {e['std']:.4f}  "
              + "  ".join(f"{h}={e['per_h'][h]:.4f}" for h in horizons))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
