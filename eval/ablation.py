#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""正式消融汇总（多 seed 口径）：use_pe / nhead / num_layers / input_len。

口径
* 实验台统一为主切分 `time_sequential`，超参照 A 组，只改被消融的那一项。
* 每档 5 seed（42, 0, 1, 2, 9999）；基准臂直接复用 A 组的 5 个 seed（`outputs/ckpt_A_ms`）。
* 每个手臂的**数据配置从 ckpt 内存的 config 读回**，保证评估口径与训练时严格一致
  （尤其 input_len=48/96 会改变 seq_len，进而改变 test 滑窗集合）。
* 指标全部在摄氏度上算；`norm_MSE` 为归一化空间的 MSE（= MSE_°C / std_OT²），
  与文献里常报的 MSE 同口径。
* 判定按 README「多 seed 规范」第二档：只写是否超过 2σ。

用法
    python eval/ablation.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer import build_model  # noqa: E402
from utils.dataset import (  # noqa: E402
    data_config_from_config, load_data_config, load_scaler, prepare_data,
)
from utils.metrics import compute_metrics, denorm_ot, target_index  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = (42, 0, 1, 2, 9999)
BAR = "=" * 104

# (维度, 配置标签, ckpt 目录, 是否为该维度的参考臂)
ARMS: List[Tuple[str, str, str, bool]] = [
    ("use_pe",     "True",  "outputs/ckpt_A_ms",        True),
    ("use_pe",     "False", "outputs/ckpt_abl_pe_false", False),
    ("nhead",      "4",     "outputs/ckpt_A_ms",        True),
    ("nhead",      "8",     "outputs/ckpt_abl_nhead8",  False),
    ("num_layers", "2",     "outputs/ckpt_A_ms",        True),
    ("num_layers", "1",     "outputs/ckpt_abl_layers1", False),
    ("input_len",  "24",    "outputs/ckpt_A_ms",        True),
    ("input_len",  "48",    "outputs/ckpt_abl_len48",   False),
    ("input_len",  "96",    "outputs/ckpt_abl_len96",   False),
]

_bundle_cache: Dict[tuple, Dict] = {}


def get_bundle(data_cfg):
    key = (data_cfg.seq_len, data_cfg.pred_len, data_cfg.split_mode, data_cfg.window_stride)
    if key not in _bundle_cache:
        _bundle_cache[key] = prepare_data(data_cfg, save=False, verbose=False)
    return _bundle_cache[key]


def eval_arm(ckpt_dir: str, scaler: Dict) -> Tuple[List[Dict], Dict]:
    """对一个手臂的 5 个 seed 求 test 指标。返回 (每 seed 指标, 附加信息)。"""
    runs, info = [], {}
    for s in SEEDS:
        ck_path = os.path.join(ckpt_dir, f"seed{s}.pt")
        if not os.path.isfile(ck_path):
            continue
        ck = torch.load(ck_path, map_location="cpu", weights_only=True)
        data_cfg = data_config_from_config(ck["config"])
        bundle = get_bundle(data_cfg)

        x = bundle["datasets"]["test"].x.numpy()
        y = bundle["datasets"]["test"].y.numpy()
        model = build_model(ck["model_config"])
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        with torch.no_grad():
            p = model(torch.as_tensor(x, dtype=torch.float32)).numpy()

        preds, gts = denorm_ot(p, scaler), denorm_ot(y, scaler)
        m = compute_metrics(preds, gts)
        mse_c = m["overall"]["RMSE"] ** 2
        runs.append({
            "seed": s, "MAE": m["overall"]["MAE"], "RMSE": m["overall"]["RMSE"],
            "norm_MSE": mse_c / float(scaler["std"][target_index(scaler)]) ** 2,
            "test_windows": int(x.shape[0]), "seq_len": data_cfg.seq_len,
            "n_params": sum(pp.numel() for pp in model.parameters()),
        })
        info = {"test_windows": int(x.shape[0]), "seq_len": data_cfg.seq_len,
                "n_params": runs[-1]["n_params"]}
    return runs, info


def agg(runs: List[Dict], key: str) -> Tuple[float, float]:
    vals = [r[key] for r in runs]
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def fmt(mean: float, std: float, nd: int = 4) -> str:
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="正式消融汇总（多 seed）")
    ap.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "base.yaml"))
    args = ap.parse_args(argv)

    cfg = load_data_config(args.config)
    scaler = load_scaler(cfg.scaler_path)
    t_idx = target_index(scaler)
    ot_std = float(scaler["std"][t_idx])

    print(BAR)
    print(f"正式消融（主切分 time_sequential，每档 5 seed；"
          f"scaler 目标列 #{t_idx} std = {ot_std:.4f}）")
    print(BAR)

    results: Dict[Tuple[str, str], Dict] = {}
    for dim, label, ckdir, is_ref in ARMS:
        runs, info = eval_arm(ckdir, scaler)
        if not runs:
            print(f"  !! {dim}={label}: 无 ckpt（{ckdir}），跳过")
            continue
        results[(dim, label)] = {"runs": runs, "is_ref": is_ref, **info}
        print(f"  {dim:<11s} {label:<6s} seeds={len(runs)}  test_win={info['test_windows']:<5d}"
              f" seq_len={info['seq_len']:<3d} params={info['n_params']:,}")

    # ---------------- 主表 ----------------
    print()
    print(BAR)
    print("消融汇总表（摄氏度；mean ± std over 5 seeds）")
    print(BAR)
    head = (f"{'维度':<12s}{'配置':<8s}{'MAE':>19s}{'RMSE':>19s}{'norm_MSE':>19s}{'params':>10s}")
    print(head)
    print("-" * len(head))
    for dim, label, ckdir, is_ref in ARMS:
        r = results.get((dim, label))
        if not r:
            continue
        ma = agg(r["runs"], "MAE")
        rm = agg(r["runs"], "RMSE")
        nm = agg(r["runs"], "norm_MSE")
        mark = " (ref)" if is_ref else ""
        print(f"{dim:<12s}{label + mark:<8s}{fmt(*ma):>19s}{fmt(*rm):>19s}{fmt(*nm):>19s}"
              f"{r['n_params']:>10,d}")

    # ---------------- 逐维度判定 ----------------
    print()
    print(BAR)
    print("逐维度 2σ 判定（delta = 该配置 − 参考配置；MAE 越小越好，delta > 0 表示更差）")
    print(BAR)

    def verdict(dim: str, ref_label: str, arm_label: str) -> None:
        ref, arm = results.get((dim, ref_label)), results.get((dim, arm_label))
        if not ref or not arm:
            return
        ref_m, ref_s = agg(ref["runs"], "MAE")
        arm_m, arm_s = agg(arm["runs"], "MAE")
        delta = arm_m - ref_m
        thr = 2 * max(ref_s, arm_s)
        ok = abs(delta) >= thr
        print(f"  · {dim}={arm_label} vs {dim}={ref_label}: delta = {delta:+.4f} °C"
              f"（{arm_m:.4f} − {ref_m:.4f}）, 2×max(σ) = {thr:.4f}  → "
              f"{'超过噪声' if ok else '落在噪声内'}")

    verdict("use_pe", "True", "False")
    verdict("nhead", "4", "8")
    verdict("num_layers", "2", "1")
    verdict("input_len", "24", "48")
    verdict("input_len", "24", "96")

    # ---------------- 落盘 JSON（供 eval/make_figs.py 复用，避免重复推理）----------------
    fig_dir = os.path.join(PROJECT_ROOT, "outputs", "figs")
    os.makedirs(fig_dir, exist_ok=True)
    payload: Dict = {"ot_std": ot_std, "seeds": list(SEEDS), "arms": [], "verdicts": []}
    for dim, label, ckdir, is_ref in ARMS:
        r = results.get((dim, label))
        if not r:
            continue
        payload["arms"].append({
            "dimension": dim, "label": label, "is_ref": bool(is_ref),
            "n_seeds": len(r["runs"]), "n_params": r["n_params"],
            "test_windows": r["test_windows"], "seq_len": r["seq_len"],
            "MAE": agg(r["runs"], "MAE"), "RMSE": agg(r["runs"], "RMSE"),
            "norm_MSE": agg(r["runs"], "norm_MSE"), "per_seed": r["runs"],
        })
    for dim, ref_label, arm_label in (("use_pe", "True", "False"),
                                      ("nhead", "4", "8"),
                                      ("num_layers", "2", "1"),
                                      ("input_len", "24", "48"),
                                      ("input_len", "24", "96")):
        ref, arm = results.get((dim, ref_label)), results.get((dim, arm_label))
        if not ref or not arm:
            continue
        ref_m, ref_s = agg(ref["runs"], "MAE")
        arm_m, arm_s = agg(arm["runs"], "MAE")
        payload["verdicts"].append({
            "dimension": dim, "arm": arm_label, "ref": ref_label,
            "delta": arm_m - ref_m, "two_sigma": 2 * max(ref_s, arm_s),
            "exceeds": bool(abs(arm_m - ref_m) >= 2 * max(ref_s, arm_s)),
        })
    out_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, "ablation.json")
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"\n  JSON: {os.path.abspath(jpath)}")

    print()
    print("  提醒：input_len=48/96 会改变 seq_len，因而 **test 滑窗集合不同**"
          "（窗口数与目标对齐都变），")
    print("        跨 input_len 的对比不是同一批目标，MAE 差异含此成分，需谨慎解读。")
    print(BAR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
