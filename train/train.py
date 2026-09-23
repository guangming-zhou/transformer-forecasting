#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练脚本：ETTh1 多步预测（只使用 train + val，训练阶段绝不接触 test）。

用法
    python train/train.py --config configs/base.yaml --seed 42
    python train/train.py --config configs/base.yaml --seed 42 \\
        --override train.epochs=5 train.lr=5e-4 model.d_model=32

--override 支持点路径，可以一次给多个，也可以重复出现：
    --override train.lr=5e-4 model.d_model=32
    --override train.lr=5e-4 --override model.d_model=32

产物
    outputs/ckpt/seed{seed}.pt     best 模型（按 val_loss）
    outputs/logs/seed{seed}.json   {seed, config, best_val, best_epoch, history, elapsed}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

if __package__ in (None, ""):                      # 允许直接 python train/train.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer import build_model, count_parameters, model_kwargs_from_dict  # noqa: E402
from utils.dataset import data_config_from_config, prepare_data  # noqa: E402
from utils.seed import make_generator, set_seed  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAR = "=" * 78


# --------------------------------------------------------------------------- #
# CLI / 配置
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transformer 训练（ETTh1 24 -> 24），仅使用 train + val")
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "base.yaml"),
                        help="yaml 配置文件路径")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子（覆盖 train.seed）")
    parser.add_argument("--override", action="extend", nargs="*", default=[],
                        metavar="KEY=VALUE",
                        help="点路径覆盖配置，如 train.lr=5e-4 model.d_model=32")
    return parser.parse_args(argv)


def _parse_scalar(text: str):
    """用 yaml.safe_load 解析 CLI 值，并对科学计数法做一次数值兜底。

    yaml 能识别 true / false / null / 32 / 0.001 / "abc"；但 PyYAML 的 float 解析器
    要求指数前有点号（`1.0e-4` 是 float，`1e-3` / `5e-4` 会被当成字符串），
    因此这里把「纯数值字符串」再兜底转成 float，保证 --override train.lr=5e-4 可用。
    """
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError:
        return text

    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def apply_overrides(config: Dict, items: List[str]) -> Dict:
    """按点路径把 KEY=VALUE 写进嵌套配置字典（缺失的中间层自动创建）。"""
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--override 需要 KEY=VALUE 形式，收到: {item!r}")
        key, value = item.split("=", 1)
        path = [p for p in key.strip().split(".") if p]
        if not path:
            raise SystemExit(f"--override 的键为空: {item!r}")

        node = config
        for part in path[:-1]:
            nxt = node.get(part)
            if nxt is None:
                nxt = node[part] = {}
            if not isinstance(nxt, dict):
                raise SystemExit(f"--override {key}: 路径 {part!r} 不是配置段")
            node = nxt
        node[path[-1]] = _parse_scalar(value)
    return config


def load_config(path: str, override_items: List[str]) -> Dict:
    if not os.path.isfile(path):
        raise SystemExit(f"找不到配置文件: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    if not isinstance(config, dict):
        raise SystemExit(f"配置文件内容不是键值对: {path}")
    return apply_overrides(config, override_items)


# --------------------------------------------------------------------------- #
# 设备
# --------------------------------------------------------------------------- #
def resolve_device(spec: str) -> torch.device:
    spec = str(spec or "auto").strip().lower()
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"配置要求 device={spec}，但当前环境 CUDA 不可用")
    return torch.device(spec)


# --------------------------------------------------------------------------- #
# 交付物指纹
# --------------------------------------------------------------------------- #
def _sha256_file(path: str) -> str:
    """ckpt 文件的 sha256。文件一变这个值就变，用于发现「交付物被覆盖过」。"""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def weights_fingerprint(state_dict: Dict) -> str:
    """模型参数的内容级指纹。

    注意：torch.save 的产物**不是字节稳定的**（zip 头里带时间戳，同样的权重
    连存三次会得到三个不同的文件 sha256），所以文件 hash 只能证明「文件被重写过」。
    要对齐两次运行是否产出了**同样的权重**，必须用这个不依赖序列化格式的指纹。
    """
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key]
        if torch.is_tensor(tensor):
            digest.update(key.encode("utf-8"))
            digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# 单轮
# --------------------------------------------------------------------------- #
def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
              device: torch.device, optimizer: Optional[torch.optim.Optimizer] = None,
              grad_clip: float = 0.0) -> float:
    """跑一轮，返回按样本数加权的平均 loss。optimizer=None 时为验证模式。"""
    training = optimizer is not None
    model.train(training)

    total_loss, total_n = 0.0, 0
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

        n = x.size(0)
        total_loss += float(loss.detach()) * n
        total_n += n

    return total_loss / max(total_n, 1)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config, args.override)
    if args.seed is not None:
        config.setdefault("train", {})["seed"] = int(args.seed)

    train_cfg: Dict = dict(config.get("train") or {})
    output_cfg: Dict = dict(config.get("output") or {})
    model_section: Dict = dict(config.get("model") or {})

    # ---- 种子与设备 ----
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)
    device = resolve_device(str(train_cfg.get("device", "auto")))

    # ---- 超参 ----
    batch_size = int(train_cfg.get("batch_size", 64))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    epochs = int(train_cfg.get("epochs", 30))
    grad_clip = float(train_cfg.get("grad_clip", 1.0) or 0.0)
    patience = int(train_cfg.get("patience", 0) or 0)
    scheduler_name = str(train_cfg.get("scheduler", "cosine")).strip().lower()
    warmup_epochs = int(train_cfg.get("warmup_epochs", 0) or 0)
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.0) or 0.0)
    ckpt_dir = str(output_cfg.get("ckpt_dir", "outputs/ckpt"))
    log_dir = str(output_cfg.get("log_dir", "outputs/logs"))
    ckpt_path = os.path.join(ckpt_dir, f"seed{seed}.pt")
    log_path = os.path.join(log_dir, f"seed{seed}.json")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(BAR)
    print("训练配置")
    print(BAR)
    print(f"  config  = {os.path.abspath(args.config)}")
    print(f"  seed    = {seed}")
    print(f"  device  = {device}  (cuda_available={torch.cuda.is_available()})")
    print(f"  overrides = {args.override or '(无)'}")
    print(f"  batch_size={batch_size}  lr={lr}  weight_decay={weight_decay}  "
          f"epochs={epochs}  grad_clip={grad_clip}")
    print(f"  scheduler={scheduler_name}  warmup_epochs={warmup_epochs}  "
          f"min_lr_ratio={min_lr_ratio}  patience={patience}")

    # ---- 数据（只构建 train / val，test 不参与训练）----
    data_cfg = data_config_from_config(config)
    bundle = prepare_data(data_cfg, save=True, verbose=False)
    datasets = bundle["datasets"]

    pin_memory = device.type == "cuda"
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=pin_memory,
                            generator=make_generator(seed + 1), drop_last=False),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=pin_memory, drop_last=False),
    }
    print(f"  train   = {len(datasets['train'])} 个滑窗 / {len(loaders['train'])} 个 batch")
    print(f"  val     = {len(datasets['val'])} 个滑窗 / {len(loaders['val'])} 个 batch")
    print(f"  num_workers=0  pin_memory={pin_memory}  (训练阶段不读 test)")

    # ---- 模型 ----
    # 输入维度与目标列下标**从数据配置推导**，不写死 7 / 6：
    # 换一份 CSV（data.feature_cols / data.target_col）后模型会自动对上，
    # 不需要手改 model 段。显式写在 model 段里的值仍然优先。
    model_section.setdefault("input_dim", len(data_cfg.features))
    model_section.setdefault("target_idx", int(data_cfg.target_idx))
    model = build_model(model_section).to(device)
    if model.input_len != data_cfg.seq_len or model.pred_len != data_cfg.pred_len:
        raise SystemExit(
            f"model.input_len/pred_len = {model.input_len}/{model.pred_len} 必须与 "
            f"data.seq_len/pred_len = {data_cfg.seq_len}/{data_cfg.pred_len} 一致；"
            f"请同时覆盖两处，例如 --override data.pred_len=48 model.pred_len=48")
    total_p, trainable_p = count_parameters(model)
    print(f"  model   = d_model={model.d_model} nhead={model.nhead} "
          f"layers={model.num_layers} dim_ff={model.dim_ff} dropout={model.dropout_p} "
          f"use_pe={model.use_pe}")
    print(f"  input   = {model.input_dim} 列 {list(data_cfg.features)}  "
          f"目标列 #{model.target_idx} = {data_cfg.target}  residual={model.residual}")
    print(f"  参数量  = {total_p:,} (可训练 {trainable_p:,})")

    # ---- 优化器 / 调度器 / 损失 ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler_name in ("cosine", "cosineannealinglr"):
        if warmup_epochs > 0:
            # 线性 warmup + cosine 衰减。warmup_epochs=0 时**刻意**走下面的
            # CosineAnnealingLR 分支，保证 lr 序列与历史运行逐位一致。
            def lr_lambda(epoch: int) -> float:
                if epoch < warmup_epochs:
                    return float(epoch + 1) / float(warmup_epochs)
                prog = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
                prog = min(max(prog, 0.0), 1.0)
                return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))

            scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = \
                torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name in ("none", "null", ""):
        scheduler = None
    elif scheduler_name in ("cosine_warmup", "cosinewarmup"):
        def lr_lambda_warm(epoch: int) -> float:      # 旧 torch 上 warmup_epochs 也能生效
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            prog = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            prog = min(max(prog, 0.0), 1.0)
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_warm)
    else:
        raise SystemExit(f"未知 scheduler: {scheduler_name}（支持 cosine / cosine_warmup / none）")
    criterion = nn.MSELoss()
    print(BAR)

    # ---- 训练循环 ----
    history: List[Dict] = []
    best_val, best_epoch, bad_epochs = float("inf"), -1, 0
    t_start = time.time()

    print(f"{'epoch':>5s} {'lr':>11s} {'train_loss':>12s} {'val_loss':>12s} {'time':>9s}")
    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        train_loss = run_epoch(model, loaders["train"], criterion, device,
                               optimizer=optimizer, grad_clip=grad_clip)
        val_loss = run_epoch(model, loaders["val"], criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        improved = val_loss < best_val
        if improved:
            best_val, best_epoch, bad_epochs = val_loss, epoch, 0
            torch.save({
                "seed": seed,
                "epoch": epoch,
                "val_loss": val_loss,
                "train_loss": train_loss,
                "model_state_dict": model.state_dict(),
                "model_config": model.config(),
                "config": config,
            }, ckpt_path)
        else:
            bad_epochs += 1

        if scheduler is not None:
            scheduler.step()

        elapsed_epoch = time.time() - t_epoch
        history.append({"epoch": epoch, "lr": current_lr, "train_loss": train_loss,
                        "val_loss": val_loss, "elapsed": elapsed_epoch})
        print(f"{epoch:>5d} {current_lr:>11.4e} {train_loss:>12.6f} {val_loss:>12.6f} "
              f"{elapsed_epoch:>8.2f}s {('*best' if improved else '')}")

        if patience > 0 and bad_epochs >= patience:
            print(f"\nearly stopping: 连续 {patience} 个 epoch val_loss 无改善，"
                  f"在第 {epoch} 轮停止")
            break

    elapsed = time.time() - t_start

    # ---- 交付物指纹：文件 sha256（看得出被覆盖）+ 权重内容指纹（可跨运行比对）----
    ckpt_sha256, weights_sha256 = "", ""
    if os.path.isfile(ckpt_path):
        ckpt_sha256 = _sha256_file(ckpt_path)
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        weights_sha256 = weights_fingerprint(saved["model_state_dict"])

    # ---- 日志 ----
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump({"seed": seed, "config": config, "best_val": best_val,
                   "best_epoch": best_epoch, "history": history, "elapsed": elapsed,
                   "ckpt_sha256": ckpt_sha256, "ckpt_sha256_8": ckpt_sha256[:8],
                   "weights_sha256": weights_sha256,
                   "weights_sha256_8": weights_sha256[:8]},
                  fh, indent=2, ensure_ascii=False)

    print(BAR)
    print(f"best val_loss = {best_val:.6f}  @ epoch {best_epoch}")
    print(f"ckpt : {os.path.abspath(ckpt_path)}  "
          f"({os.path.getsize(ckpt_path) / 1024:.1f} KB)")
    print(f"ckpt    sha256[:8] = {ckpt_sha256[:8]}   (文件指纹：变了就说明被覆盖过)")
    print(f"weights sha256[:8] = {weights_sha256[:8]}   (内容指纹：可跨运行比对同一权重)")
    print(f"log  : {os.path.abspath(log_path)}")
    print(f"总用时: {elapsed:.2f}s")
    print(BAR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
