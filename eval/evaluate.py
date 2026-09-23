#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""评估脚本：读一个 ckpt，在 **test 集**上推理，输出反归一化（摄氏度）的 MAE / RMSE。

只读 test，不训练。输入归一化与反归一化**统一使用 `outputs/scaler.npz`**（train 段拟合），
本脚本不重算、不覆盖 scaler。

用法
    python eval/evaluate.py --ckpt outputs/ckpt/seed42.pt \
        --config configs/base.yaml --out outputs/figs

产物
    <out>/preds.npz       keys: preds, gts, last_input_ot, input_ot, target_times,
                                horizons, meta(JSON 字符串)
    <out>/pred_curve.png  预测曲线（test 示例窗口）
    <out>/err_hist.png    误差分布 + 分步误差
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np
import torch

if __package__ in (None, ""):                      # 允许直接 python eval/evaluate.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib                                 # noqa: E402
matplotlib.use("Agg")                             # 无显示环境
import matplotlib.pyplot as plt                   # noqa: E402

from eval.common import load_test_set, resolve_device  # noqa: E402
from models.transformer import build_model, count_parameters  # noqa: E402
from utils.dataset import load_data_config, load_scaler, prepare_data  # noqa: E402
from utils.metrics import (  # noqa: E402
    compute_metrics, denorm_ot, format_metrics, horizons_for, target_index,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAR = "=" * 78

# 让中文标注正常渲染（Windows 自带字体）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transformer 预测评估（仅 test 集，不训练）")
    p.add_argument("--ckpt", required=True, help="checkpoint 路径")
    p.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "base.yaml"))
    p.add_argument("--out", default=os.path.join(PROJECT_ROOT, "outputs", "figs"),
                   help="输出目录（preds.npz 与两张图）")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="auto", help="auto / cpu / cuda")
    return p.parse_args(argv)


def check_scaler(cfg, disk_scaler: Dict, bundle: Dict) -> None:
    """核对磁盘 scaler 与按同一配置重新拟合的结果是否一致（只告警，不用重算值评估）。"""
    ref = bundle["scaler"]
    for key in ("mean", "std", "low", "high"):
        if not np.allclose(disk_scaler[key], ref[key], rtol=1e-9, atol=1e-9):
            print(f"  [WARN] 磁盘 scaler 的 {key} 与按 --config 重新拟合的结果不一致，"
                  f"请确认 outputs/scaler.npz 没有被其它配置覆盖过")
            return
    print("  [OK] 磁盘 scaler 与 --config 重新拟合结果一致（评估仍只使用磁盘 scaler）")


def plot_pred_curve(preds, gts, input_ot, target_times, cfg, path) -> None:
    n = preds.shape[0]
    picks = np.unique(np.linspace(0, n - 1, 4).astype(int))
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    t_hist = np.arange(-cfg.seq_len + 1, 1)
    t_fut = np.arange(1, cfg.pred_len + 1)
    for ax, i in zip(axes.ravel(), picks):
        ax.plot(t_hist, input_ot[i], color="tab:gray", lw=1.4, label="输入历史 OT")
        ax.plot(t_fut, gts[i], color="tab:blue", lw=1.6, marker="o", ms=2.5, label="真值")
        ax.plot(t_fut, preds[i], color="tab:red", lw=1.6, ls="--", marker="s", ms=2.5,
                label="预测")
        ax.axvline(0, color="k", lw=0.8, alpha=0.5)
        mae_i = float(np.mean(np.abs(preds[i] - gts[i])))
        ax.set_title(f"window #{i}   MAE={mae_i:.2f}°C   "
                     f"{str(target_times[i, 0])[:16]} 起", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlabel("相对预测起点的小时数")
        ax.set_ylabel("OT (°C)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"ETTh1 OT {cfg.pred_len} 步预测（test 集示例窗口）", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_error(preds, gts, path) -> None:
    err = preds - gts
    n_steps = preds.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    axes[0].hist(err.ravel(), bins=70, color="tab:red", alpha=0.75)
    axes[0].axvline(0.0, color="k", lw=1.2)
    axes[0].axvline(err.mean(), color="tab:blue", lw=1.2, ls="--",
                    label=f"均值 {err.mean():+.2f}")
    axes[0].set_title(f"误差分布 (pred − gt)   std={err.std():.2f}°C")
    axes[0].set_xlabel("误差 (°C)")
    axes[0].set_ylabel("计数")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    steps = np.arange(1, n_steps + 1)
    mae = np.array([np.mean(np.abs(err[:, h])) for h in range(n_steps)])
    rmse = np.array([np.sqrt(np.mean(err[:, h] ** 2)) for h in range(n_steps)])
    axes[1].plot(steps, mae, marker="o", ms=3.5, color="tab:red", label="MAE")
    axes[1].plot(steps, rmse, marker="s", ms=3.5, color="tab:blue", label="RMSE")
    for h in horizons_for(n_steps):
        if h <= n_steps:
            axes[1].axvline(h, color="gray", ls=":", lw=0.8)
    axes[1].set_title("分步误差（虚线为 "
                      + "/".join(f"h{h}" for h in horizons_for(n_steps)) + "）")
    axes[1].set_xlabel("预测步（小时）")
    axes[1].set_ylabel("°C")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    if not os.path.isfile(args.ckpt):
        raise SystemExit(f"找不到 ckpt: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = load_data_config(args.config)
    model = build_model(ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])      # 必须加载权重，否则评估的是随机初始化模型
    model.eval()

    if model.input_len != cfg.seq_len or model.pred_len != cfg.pred_len:
        raise SystemExit(f"ckpt 的 input_len/pred_len = {model.input_len}/{model.pred_len} "
                         f"与 config 的 seq_len/pred_len = {cfg.seq_len}/{cfg.pred_len} 不一致")
    ckpt_data = (ckpt.get("config") or {}).get("data", {})
    if ckpt_data.get("split_mode") not in (None, cfg.split_mode):
        print(f"  i ckpt 训练时 split_mode={ckpt_data.get('split_mode')}，"
              f"当前 config={cfg.split_mode}；test 段在两种模式下相同，不影响评估")

    total_p, _ = count_parameters(model)
    scaler = load_scaler(cfg.scaler_path)          # 反归一化只认这个文件

    print(BAR)
    print("评估配置")
    print(BAR)
    print(f"  ckpt     = {os.path.abspath(args.ckpt)}")
    val_loss = ckpt.get("val_loss")
    val_str = f"{val_loss:.6f}" if isinstance(val_loss, float) else str(val_loss)
    print(f"  seed     = {ckpt.get('seed')}    best_epoch = {ckpt.get('epoch')}   "
          f"val_loss = {val_str}")
    print(f"  model    = d_model={model.d_model} nhead={model.nhead} layers={model.num_layers} "
          f"dim_ff={model.dim_ff} dropout={model.dropout_p} use_pe={model.use_pe}  "
          f"参数量={total_p:,}")
    print(f"  device   = {device}")
    ot_idx = target_index(scaler)
    print(f"  scaler   = {os.path.abspath(cfg.scaler_path)}  "
          f"目标列 #{ot_idx} mean={scaler['mean'][ot_idx]:.4f} std={scaler['std'][ot_idx]:.4f}")

    bundle = prepare_data(cfg, save=False, verbose=False)   # 仅用于 scaler 一致性核对
    check_scaler(cfg, scaler, bundle)

    ts = load_test_set(cfg, scaler)
    x = ts["x"]
    input_ot, last_input_ot, target_times = ts["input_ot"], ts["last_input_ot"], ts["target_times"]
    print(f"  test     = {x.shape[0]} 个滑窗   x={x.shape}   "
          f"{target_times[0, 0]} -> {target_times[-1, -1]}")

    with torch.no_grad():
        preds_norm = torch.cat([
            model(torch.as_tensor(x[i:i + args.batch_size], dtype=torch.float32, device=device))
            for i in range(0, len(x), args.batch_size)
        ]).cpu().numpy()

    # 反归一化 -> 摄氏度
    preds = denorm_ot(preds_norm, scaler)
    gts = ts["gts"]
    metrics = compute_metrics(preds, gts)

    print()
    print(BAR)
    print(f"test 指标（反归一化到摄氏度，{len(x)} 个滑窗 × {cfg.pred_len} 步）")
    print(BAR)
    print(format_metrics(metrics))

    os.makedirs(args.out, exist_ok=True)
    npz_path = os.path.join(args.out, "preds.npz")
    meta = {
        "ckpt": os.path.abspath(args.ckpt),
        "seed": ckpt.get("seed"),
        "best_epoch": ckpt.get("epoch"),
        "val_loss": ckpt.get("val_loss"),
        "model_config": model.config(),
        "split": "test",
        "n_windows": int(len(x)),
        "seq_len": cfg.seq_len,
        "pred_len": cfg.pred_len,
        "scaler_path": os.path.abspath(cfg.scaler_path),
        "ot_mean": float(scaler["mean"][target_index(scaler)]),
        "ot_std": float(scaler["std"][target_index(scaler)]),
        "target_time_start": str(target_times[0, 0]),
        "target_time_end": str(target_times[-1, -1]),
        "metrics": metrics,
    }
    np.savez(
        npz_path,
        preds=preds,                                  # (N, pred_len) 摄氏度
        gts=gts,                                      # (N, pred_len) 摄氏度
        last_input_ot=last_input_ot,                  # (N,)    输入段最后一步的 OT（摄氏度）
        input_ot=input_ot,                            # (N, seq_len) 输入段 OT 历史（摄氏度）
        target_times=target_times.astype("datetime64[ns]"),   # (N, pred_len)
        horizons=np.arange(1, cfg.pred_len + 1),      # (24,)
        meta=np.array(json.dumps(meta, ensure_ascii=False)),
    )

    curve_path = os.path.join(args.out, "pred_curve.png")
    hist_path = os.path.join(args.out, "err_hist.png")
    plot_pred_curve(preds, gts, input_ot, target_times, cfg, curve_path)
    plot_error(preds, gts, hist_path)

    print()
    print(f"  preds.npz     : {os.path.abspath(npz_path)}  "
          f"({os.path.getsize(npz_path) / 1024:.1f} KB)")
    print(f"  pred_curve.png: {os.path.abspath(curve_path)}")
    print(f"  err_hist.png  : {os.path.abspath(hist_path)}")
    print(BAR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
