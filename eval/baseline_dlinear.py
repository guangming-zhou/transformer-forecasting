#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Reference: cure-lab/LTSF-Linear/models/DLinear.py (Apache-2.0).
# Copyright 2022 DLinear Authors. All rights reserved.
# Local changes: validation, target-channel output, training and evaluation integration.
# See THIRD_PARTY_NOTICES.md and LICENSES/LTSF-Linear-Apache-2.0.txt.
"""DLinear 基线（Zeng et al., AAAI 2023, "Are Transformers Effective for Time Series Forecasting?"）。

与被比较的 Transformer 严格对齐，只换模型族：
* 同一份数据管道与切分（默认 time_sequential）、同样的 seq_len / pred_len
* 同一套 train-only 拟合的 scaler（读配置里的 `output.scaler_path`）
* 同样的优化配置：AdamW(lr=1e-3, wd=1e-4)、CosineAnnealingLR(T_max=epochs)、
  MSELoss、grad_clip=1.0、batch=64、epochs=30、early stop patience=8
* 同样的验证选点：按 val_loss 保存 best，再在 test 上评估
* 同样的多 step 直接输出（不是自回归），损失只算目标列

相对原论文的两处**有意的偏离**，务必一起读：
1. 原论文做多变量 -> 多变量（对 7 列都算 loss）；本项目只取目标列输出。
   由于该实现逐通道独立计算且没有跨通道混合，目标预测实际上只依赖目标列历史，
   因此结果明确记为 **DLinear-S**，不声称使用了其余输入变量。
2. 原论文 kernel_size=25；这里保持 25，seq_len=24 时靠前后各 pad 12 步凑够长度，
   与官方实现的 `series_decomp` 行为一致。

用法
    python eval/baseline_dlinear.py --config configs/base.yaml --seed 42 --out outputs/figs
    python eval/baseline_dlinear.py --config configs/lstf96.yaml --seed 42 --out outputs/_runs/dlinear96

产物
    <out>/preds_dlinear_seed{seed}.npz
    <out>/metrics_dlinear_seed{seed}.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import (  # noqa: E402
    FEATURES, TARGET_IDX, load_data_config, load_scaler, prepare_data,
)
from utils.metrics import compute_metrics, denorm_ot, format_metrics  # noqa: E402
from utils.seed import make_generator, set_seed  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAR = "=" * 78


class MovingAvg(nn.Module):
    """沿时间维的滑动平均，两端用边界值复制补齐（与官方实现一致）。"""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size 必须 >= 1")
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) -> (B, L, C)"""
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    """序列分解：trend = 滑动平均，seasonal = 原序列 - trend。"""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor):
        trend = self.moving_avg(x)
        return x - trend, trend


class DLinearForecaster(nn.Module):
    """DLinear：对 (trend, seasonal) 各用一个线性层从 seq_len 映射到 pred_len。

    通道独立 + 权重共享：把 (B, C, L) 展平成 (B*C, L) 过同一个 Linear，
    输出再取目标通道。参数量与通道数无关。
    """

    def __init__(self, seq_len: int = 24, pred_len: int = 24, kernel_size: int = 25,
                 target_idx: int = TARGET_IDX, individual: bool = False) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.kernel_size = int(kernel_size)
        self.target_idx = int(target_idx)
        self.individual = bool(individual)
        self.decomp = SeriesDecomp(self.kernel_size)
        self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
        self.config = {"seq_len": self.seq_len, "pred_len": self.pred_len,
                       "kernel_size": self.kernel_size, "target_idx": self.target_idx,
                       "individual": self.individual}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"输入需为 (B, L, C)，收到 {tuple(x.shape)}")
        if x.shape[1] != self.seq_len:
            raise ValueError(f"输入长度需为 {self.seq_len}，收到 {x.shape[1]}")
        seasonal, trend = self.decomp(x)
        b, length, channels = seasonal.shape
        seasonal = seasonal.permute(0, 2, 1).reshape(b * channels, length)
        trend = trend.permute(0, 2, 1).reshape(b * channels, length)
        out = self.linear_seasonal(seasonal) + self.linear_trend(trend)
        out = out.reshape(b, channels, self.pred_len)
        return out[:, self.target_idx, :]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DLinear 基线（与 Transformer 同协议）")
    p.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "base.yaml"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=os.path.join(PROJECT_ROOT, "outputs", "figs"))
    p.add_argument("--kernel-size", type=int, default=25)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    return p.parse_args(argv)


def resolve_device(spec: str) -> torch.device:
    spec = str(spec or "auto").strip().lower()
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def run_epoch(model, loader, criterion, device, optimizer=None, grad_clip=0.0) -> float:
    training = optimizer is not None
    model.train(training)
    total, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            loss = criterion(model(x), y)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total += float(loss.detach()) * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_data_config(args.config)
    device = resolve_device(args.device)
    set_seed(args.seed)

    bundle = prepare_data(cfg, save=False, verbose=False)
    datasets = bundle["datasets"]
    scaler = load_scaler(cfg.scaler_path)
    for key in ("mean", "std"):
        assert np.allclose(scaler[key], bundle["scaler"][key]), f"scaler[{key}] 与磁盘不一致"

    pin = device.type == "cuda"
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                            num_workers=0, pin_memory=pin,
                            generator=make_generator(args.seed + 1)),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False,
                          num_workers=0, pin_memory=pin),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False,
                           num_workers=0, pin_memory=pin),
    }

    model = DLinearForecaster(seq_len=cfg.seq_len, pred_len=cfg.pred_len,
                              kernel_size=args.kernel_size,
                              target_idx=cfg.target_idx).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    print(BAR)
    print(f"DLinear 基线（seed={args.seed}，{cfg.seq_len} -> {cfg.pred_len}）")
    print(BAR)
    print(f"  device   = {device}   参数量 = {n_params:,}（通道独立共享权重）")
    print(f"  kernel_size={args.kernel_size} lr={args.lr} wd={args.weight_decay} "
          f"epochs={args.epochs} patience={args.patience} batch={args.batch_size}")
    print(f"  train/val/test = {len(datasets['train'])}/{len(datasets['val'])}/"
          f"{len(datasets['test'])} 个滑窗")
    print(f"{'epoch':>5s}{'lr':>11s}{'train_loss':>12s}{'val_loss':>11s}{'time':>8s}")

    best_val, best_epoch, best_state, bad = float("inf"), -1, None, 0
    history: List[Dict] = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        te = time.time()
        tr = run_epoch(model, loaders["train"], criterion, device, optimizer, args.grad_clip)
        va = run_epoch(model, loaders["val"], criterion, device)
        lr_now = optimizer.param_groups[0]["lr"]
        improved = va < best_val
        if improved:
            best_val, best_epoch, bad = va, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        scheduler.step()
        dt = time.time() - te
        history.append({"epoch": epoch, "lr": lr_now, "train_loss": tr,
                        "val_loss": va, "elapsed": dt})
        print(f"{epoch:>5d}{lr_now:>11.4e}{tr:>12.6f}{va:>11.6f}{dt:>7.2f}s"
              f"{'  *best' if improved else ''}")
        if bad >= args.patience:
            print(f"\n  early stopping: 连续 {args.patience} 轮无改善，停在第 {epoch} 轮")
            break
    train_time = time.time() - t0

    model.load_state_dict(best_state)
    model.eval()
    preds_n, gts_n = [], []
    with torch.no_grad():
        for x, y in loaders["test"]:
            preds_n.append(model(x.to(device)).cpu().numpy())
            gts_n.append(y.numpy())
    preds = denorm_ot(np.concatenate(preds_n), scaler)
    gts = denorm_ot(np.concatenate(gts_n), scaler)
    metrics = compute_metrics(preds, gts)

    print()
    print(BAR)
    print(f"DLinear test 指标（seed={args.seed}，best_epoch={best_epoch}，"
          f"val_loss={best_val:.6f}）")
    print(BAR)
    print(format_metrics(metrics))

    os.makedirs(args.out, exist_ok=True)
    npz_path = os.path.join(args.out, f"preds_dlinear_seed{args.seed}.npz")
    np.savez(npz_path, preds=preds, gts=gts, horizons=np.arange(1, cfg.pred_len + 1),
             meta=np.array(json.dumps({
                "model": "DLinear-S", "seed": args.seed, "best_epoch": best_epoch,
                 "best_val": best_val, "n_params": n_params,
                 "model_config": model.config, "metrics": metrics,
             }, ensure_ascii=False)))
    json_path = os.path.join(args.out, f"metrics_dlinear_seed{args.seed}.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"model": "DLinear-S", "seed": args.seed,
                   "n_params": n_params, "kernel_size": args.kernel_size,
                   "seq_len": cfg.seq_len, "pred_len": cfg.pred_len,
                   "split_mode": cfg.split_mode,
                   "lr": args.lr, "weight_decay": args.weight_decay,
                   "epochs_run": len(history), "best_epoch": best_epoch,
                   "best_val": best_val, "train_time_sec": round(train_time, 2),
                   "history": history, "metrics": metrics},
                  fh, indent=2, ensure_ascii=False)

    print()
    print(f"  preds   : {os.path.abspath(npz_path)}")
    print(f"  metrics : {os.path.abspath(json_path)}")
    print(BAR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
