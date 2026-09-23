#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Attention 可视化：加载 ckpt，在 test 集上取若干样本，画每层自注意力热力图。

* 用 `TransformerForecaster.forward_with_attn`（对 `forward()` 无影响）拿到
  每层 `(B, nhead, L, L)` 的注意力权重；**对 head 取平均**后画 24×24 热力图。
* 图布局：上两行 = 每层 × 每个样本的热力图；底行 = 每个 key 位置收到的平均注意力
  （看模型是聚焦最近几个时刻还是分散）。
* 脚本内断言 `forward_with_attn` 与 `forward` 数值一致（否则手工路径与
  `nn.TransformerEncoder` 不等价，可视化就无意义）。

用法
    python eval/visualize_attn.py --ckpt outputs/ckpt/seed42.pt --n-samples 3

产物
    outputs/figs/attn.png
    outputs/figs/attn_stats.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.gridspec import GridSpec            # noqa: E402

from models.transformer import build_model          # noqa: E402
from utils.dataset import load_data_config, prepare_data  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAR = "=" * 78

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transformer 自注意力可视化")
    p.add_argument("--ckpt", default=os.path.join(PROJECT_ROOT, "outputs", "ckpt", "seed42.pt"))
    p.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "base.yaml"))
    p.add_argument("--out", default=os.path.join(PROJECT_ROOT, "outputs", "figs"))
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model = build_model(ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cfg = load_data_config(args.config)
    bundle = prepare_data(cfg, save=False, verbose=False)
    x_test = bundle["datasets"]["test"].x.numpy()
    n_win, seq_len, _ = x_test.shape

    picks = np.unique(np.linspace(0, n_win - 1, args.n_samples).astype(int))

    # ---- 数值一致性断言：手工路径必须与 forward() 等价 ----
    with torch.no_grad():
        probe = torch.as_tensor(x_test[picks], dtype=torch.float32, device=device)
        out_plain = model(probe)
        out_attn, attn_list = model.forward_with_attn(probe)
    max_diff = float((out_plain - out_attn).abs().max())
    print(BAR)
    print("注意力可视化")
    print(BAR)
    print(f"  ckpt      = {os.path.abspath(args.ckpt)}")
    print(f"  seed={ckpt.get('seed')}  best_epoch={ckpt.get('epoch')}  "
          f"layers={model.num_layers} nhead={model.nhead} input_len={model.input_len}")
    print(f"  样本      = test 窗口 {picks.tolist()}（共 {n_win} 个）")
    print(f"  forward_with_attn vs forward 最大绝对偏差 = {max_diff:.3e}  → "
          f"{'一致' if max_diff < 1e-5 else '不一致（可视化不可信）'}")
    assert max_diff < 1e-5, "forward_with_attn 与 forward 不等价"

    # ---- 整理注意力：每层 (S, L, L)（对 head 取平均）----
    per_layer = [a.mean(dim=1).cpu().numpy() for a in attn_list]   # [(S, L, L), ...]
    n_layers = len(per_layer)

    # 存**仓库相对路径**，避免把本机绝对路径写进要分发的产物
    ckpt_rel = os.path.relpath(os.path.abspath(args.ckpt), PROJECT_ROOT).replace("\\", "/")
    stats: Dict = {"ckpt": ckpt_rel, "seed": ckpt.get("seed"),
                   "samples": picks.tolist(), "seq_len": int(seq_len),
                   "nhead": model.nhead, "layers": n_layers,
                   "forward_with_attn_max_abs_diff": max_diff,
                   "per_layer": []}

    print()
    print(f"  {'layer':>6s}{'自注意力熵':>14s}{'最近4步占比':>14s}{'最远4步占比':>14s}"
          f"{'对角(自)占比':>14s}")
    for li, attn in enumerate(per_layer):
        mean_attn = attn.mean(axis=0)                    # (L, L) 对样本平均
        eps = 1e-12
        entropy = float(-(mean_attn * np.log(mean_attn + eps)).sum(axis=1).mean())
        recent4 = float(mean_attn[:, -4:].sum(axis=1).mean())
        oldest4 = float(mean_attn[:, :4].sum(axis=1).mean())
        diag = float(np.diag(mean_attn).mean())
        print(f"  {li:>6d}{entropy:>14.4f}{recent4:>13.2%}{oldest4:>14.2%}{diag:>14.2%}")
        stats["per_layer"].append({
            "layer": li, "entropy": entropy, "recent4_share": recent4,
            "oldest4_share": oldest4, "diagonal_share": diag,
        })
    stats["uniform_entropy"] = float(np.log(seq_len))
    stats["uniform_recent4_share"] = 4.0 / seq_len
    print(f"  {'均匀分布':>6s}{np.log(seq_len):>14.4f}{4.0 / seq_len:>13.2%}"
          f"{4.0 / seq_len:>14.2%}{1.0 / seq_len:>14.2%}")

    # ---- 画图 ----
    fig = plt.figure(figsize=(15.5, 9))
    gs = GridSpec(n_layers + 1, len(picks), figure=fig, height_ratios=[1] * n_layers + [0.85],
                  hspace=0.42, wspace=0.28)

    vmax = max(float(a.max()) for a in per_layer)
    for li, attn in enumerate(per_layer):
        for si, wi in enumerate(picks):
            ax = fig.add_subplot(gs[li, si])
            im = ax.imshow(attn[si], cmap="viridis", aspect="auto", origin="lower",
                           vmin=0.0, vmax=vmax)
            ax.set_title(f"layer {li + 1} · window #{wi}", fontsize=9)
            ax.set_xlabel("key 位置（历史时刻）", fontsize=8)
            ax.set_ylabel("query 位置", fontsize=8)
            ax.tick_params(labelsize=7)
            if si == len(picks) - 1:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    axp = fig.add_subplot(gs[n_layers, :])
    positions = np.arange(seq_len)
    for li, attn in enumerate(per_layer):
        profile = attn.mean(axis=(0, 1))                 # (L,) 对样本与 query 平均
        axp.plot(positions, profile, marker="o", ms=3.5, label=f"layer {li + 1}")
    axp.axhline(1.0 / seq_len, color="gray", ls="--", lw=1,
                label=f"均匀基线 ({1.0 / seq_len:.3f})")
    axp.set_xlabel("key 位置（0 = 最早，23 = 最近）")
    axp.set_ylabel("平均注意力权重")
    axp.set_title("每个历史位置收到的平均注意力（对样本与 query 平均）", fontsize=10)
    axp.legend(fontsize=8)
    axp.grid(alpha=0.3)

    fig.suptitle(f"Transformer 自注意力（head 平均，seed={ckpt.get('seed')}，"
                 f"{len(picks)} 个 test 样本 × {n_layers} 层）", fontsize=12)
    os.makedirs(args.out, exist_ok=True)
    png = os.path.join(args.out, "attn.png")
    fig.savefig(png, dpi=140, bbox_inches="tight")
    plt.close(fig)

    json_path = os.path.join(args.out, "attn_stats.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    print()
    print(f"  attn.png       : {os.path.abspath(png)}")
    print(f"  attn_stats.json: {os.path.abspath(json_path)}")
    print(BAR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
