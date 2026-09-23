#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LSTM 基线：输入 (B, 24, 7) -> 输出 (B, 24)。

与 A 组（Transformer）严格对齐，只换模型族：
* 同一份数据管道与切分（time_sequential，input_len=24，pred_len=24）
* 同一套 train-only 拟合的 scaler（读 `outputs/scaler.npz`）
* 同样的优化配置：AdamW(lr=1e-3, wd=1e-4)、CosineAnnealingLR(T_max=epochs)、
  MSELoss、grad_clip=1.0、batch=64、epochs=30、early stop patience=8
* 同样的验证选点：按 val_loss 保存 best，再在 test 上评估

多 seed 通过重复调用（严格串行），例如 seed ∈ {42, 0, 1, 2, 9999}。

用法
    python eval/baseline_lstm.py --config configs/base.yaml --seed 42 --out outputs/figs

产物
    <out>/preds_lstm_seed{seed}.npz
    <out>/metrics_lstm_seed{seed}.json
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
    load_data_config, load_scaler, make_dataloaders, prepare_data,
)
from utils.metrics import compute_metrics, denorm_ot, format_metrics  # noqa: E402
from utils.seed import make_generator, set_seed  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAR = "=" * 78


class LSTMForecaster(nn.Module):
    """2 层 LSTM + 线性头：取最后一步隐状态映射到 pred_len。"""

    def __init__(self, input_dim: int = 7, hidden: int = 64, num_layers: int = 2,
                 dropout: float = 0.1, pred_len: int = 24) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, pred_len)
        self.config = {"input_dim": input_dim, "hidden": hidden, "num_layers": num_layers,
                       "dropout": dropout, "pred_len": pred_len}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LSTM 基线（与 A 组同协议）")
    p.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "base.yaml"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=os.path.join(PROJECT_ROOT, "outputs", "figs"))
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
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
    scaler = load_scaler(cfg.scaler_path)                 # 与 evaluate.py 同一份
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

    model = LSTMForecaster(input_dim=len(cfg.features),
                           hidden=args.hidden, num_layers=args.num_layers,
                           dropout=args.dropout, pred_len=cfg.pred_len).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    print(BAR)
    print(f"LSTM 基线（seed={args.seed}）")
    print(BAR)
    print(f"  device   = {device}   参数量 = {n_params:,}")
    print(f"  hidden={args.hidden} layers={args.num_layers} dropout={args.dropout} "
          f"pred_len={cfg.pred_len}")
    print(f"  lr={args.lr} wd={args.weight_decay} epochs={args.epochs} "
          f"patience={args.patience} batch={args.batch_size} grad_clip={args.grad_clip}")
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
    print(f"LSTM test 指标（seed={args.seed}，best_epoch={best_epoch}，val_loss={best_val:.6f}）")
    print(BAR)
    print(format_metrics(metrics))

    os.makedirs(args.out, exist_ok=True)
    npz_path = os.path.join(args.out, f"preds_lstm_seed{args.seed}.npz")
    np.savez(npz_path, preds=preds, gts=gts, horizons=np.arange(1, cfg.pred_len + 1),
             meta=np.array(json.dumps({
                 "model": "LSTM", "seed": args.seed, "best_epoch": best_epoch,
                 "best_val": best_val, "n_params": n_params,
                 "model_config": model.config, "metrics": metrics,
             }, ensure_ascii=False)))
    json_path = os.path.join(args.out, f"metrics_lstm_seed{args.seed}.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"model": "LSTM", "seed": args.seed, "n_seeds": 1,
                   "seq_len": cfg.seq_len, "pred_len": cfg.pred_len,
                   "split_mode": cfg.split_mode,
                   "n_params": n_params, "hidden": args.hidden,
                   "num_layers": args.num_layers, "dropout": args.dropout,
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
