#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 README 引用的两张补充图。

1. `outputs/figs/ablation.png` —— 四维消融的 MAE 柱状图（带 ±std 误差棒），
   数据来自 `outputs/ablation.json`（由 `eval/ablation.py` 产出）。
2. `outputs/figs/drift.png` —— 分布漂移示意：OT 全序列 + train/val/test 划分，
   以及三个 split 的 OT 分布与 train-only 归一化下的均值偏移。

用法
    python eval/make_figs.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402

from utils.dataset import (  # noqa: E402
    FEATURES, TARGET_IDX, clean_dataframe, load_data_config, load_raw, split_dataframe,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(PROJECT_ROOT, "outputs", "figs")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DIMS = [("use_pe", "use_pe"), ("nhead", "nhead"),
        ("num_layers", "num_layers"), ("input_len", "input_len")]


def plot_ablation() -> str:
    path = os.path.join(PROJECT_ROOT, "outputs", "ablation.json")
    if not os.path.isfile(path):
        raise SystemExit(f"缺少 {path}，请先运行 eval/ablation.py")
    data = json.load(open(path, encoding="utf-8"))
    arms = data["arms"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, (dim, title) in zip(axes, DIMS):
        group = [a for a in arms if a["dimension"] == dim]
        labels = [a["label"] + ("\n(ref)" if a["is_ref"] else "") for a in group]
        means = [a["MAE"][0] for a in group]
        stds = [a["MAE"][1] for a in group]
        colors = ["tab:gray" if a["is_ref"] else "tab:blue" for a in group]
        x = np.arange(len(group))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
        for xi, (m, s) in zip(x, zip(means, stds)):
            ax.text(xi, m + max(stds) * 0.06 + 0.02, f"{m:.3f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3, axis="y")
        if ax is axes[0]:
            ax.set_ylabel("test MAE (°C)")
        ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.25)

    # 参考线：A 组基线
    ref_mae = next(a["MAE"][0] for a in arms
                   if a["dimension"] == "use_pe" and a["is_ref"])
    for ax in axes:
        ax.axhline(ref_mae, color="tab:red", ls="--", lw=1, alpha=0.7)
        ax.text(0.02, 0.96, f"A 基线 {ref_mae:.3f}", transform=ax.transAxes,
                color="tab:red", fontsize=7.5, va="top", ha="left")

    fig.suptitle("结构消融：test MAE（mean ± std over 5 seeds；全部对比均落在种子噪声内）",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIGS, "ablation.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_drift() -> str:
    cfg = load_data_config()
    clean, _ = clean_dataframe(load_raw(cfg), cfg)
    ot = clean["OT"].to_numpy(dtype="float64")
    splits = split_dataframe(clean, cfg)
    tr = splits["train"]["OT"].to_numpy()
    va = splits["val"]["OT"].to_numpy()
    te = splits["test"]["OT"].to_numpy()
    mu, sd = float(tr.mean()), float(tr.std(ddof=0))

    fig, axes = plt.subplots(2, 1, figsize=(13, 8),
                             gridspec_kw={"height_ratios": [1.25, 1]})

    # ---- 上：全序列 + 划分 ----
    ax = axes[0]
    x = np.arange(len(ot))
    ax.plot(x, ot, lw=0.6, color="tab:blue")
    bounds = [(0, len(tr), "tab:green", "train 60%"),
              (len(tr), len(tr) + len(va), "tab:orange", "val 20%"),
              (len(tr) + len(va), len(ot), "tab:red", "test 20%")]
    for a, b, c, lab in bounds:
        ax.axvspan(a, b, color=c, alpha=0.10)
        ax.plot([a, b], [tr.mean() if lab.startswith("train") else
                         (va.mean() if lab.startswith("val") else te.mean())] * 2,
                color=c, ls="--", lw=1.6)
    ax.axhline(mu, color="k", ls=":", lw=1.0)
    ax.text(len(ot) * 0.005, mu, f" train 均值 {mu:.2f}°C", fontsize=8, va="bottom")
    ax.set_ylabel("OT (°C)")
    ax.set_title("OT 全序列与时间顺序 6:2:2 划分（虚线 = 各段均值）", fontsize=11)
    ax.grid(alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.25) for _, _, c, _ in bounds]
    ax.legend(handles, [lab for _, _, _, lab in bounds], fontsize=8, loc="upper right")

    # ---- 下：三个 split 的分布 ----
    ax2 = axes[1]
    bins = np.linspace(float(ot.min()), float(ot.max()), 60)
    for arr, c, lab in ((tr, "tab:green", "train"), (va, "tab:orange", "val"),
                        (te, "tab:red", "test")):
        ax2.hist(arr, bins=bins, density=True, histtype="step", lw=2, color=c,
                 label=f"{lab}  mean={arr.mean():.2f}  std={arr.std(ddof=0):.2f}")
        ax2.axvline(float(arr.mean()), color=c, ls="--", lw=1.2)
    off_v = (va.mean() - mu) / sd
    off_t = (te.mean() - mu) / sd
    ax2.set_xlabel("OT (°C)")
    ax2.set_ylabel("密度")
    ax2.set_title(f"三个 split 的 OT 分布：相对 train-only 归一化，"
                  f"val 偏移 {off_v:+.2f}σ，test 偏移 {off_t:+.2f}σ", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIGS, "drift.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    p1 = plot_ablation()
    p2 = plot_drift()
    print(f"  ablation.png : {p1}")
    print(f"  drift.png    : {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
