#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Baseline 汇总：persistence / ARIMA / LSTM / DLinear / Transformer / Transformer+残差。

本脚本是 `eval/head_to_head.py` 的**薄封装**（聚合、表格与 2σ 判定只有一份实现），
只负责把「各基线产物放在哪」这件事写死。

评测目标与切分完全一致：同一份 test 滑窗、同一份 `outputs/scaler.npz`，
指标统一走 `utils.metrics.compute_metrics`（全部在摄氏度上）。
Transformer 的 5 个 seed 用 `outputs/ckpt_A_ms/seed{s}.pt` 现场推理。

用法
    python eval/summarize_baselines.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import head_to_head  # noqa: E402
from eval.common import PROJECT_ROOT  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="persistence / ARIMA / LSTM / DLinear / Transformer 汇总")
    p.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "base.yaml"))
    p.add_argument("--figs", default=os.path.join(PROJECT_ROOT, "outputs", "figs"))
    p.add_argument("--ckpt-dir", default=os.path.join(PROJECT_ROOT, "outputs", "ckpt_A_ms"))
    p.add_argument("--ckpt-res-dir", default=os.path.join(PROJECT_ROOT, "outputs", "ckpt_res"))
    p.add_argument("--out", default=os.path.join(PROJECT_ROOT, "outputs", "baselines_summary.json"))
    p.add_argument("--no-residual", action="store_true", help="不汇总残差变体")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    figs = args.figs

    argv_out: List[str] = [
        "--config", args.config,
        "--seed-json", f"LSTM={os.path.join(figs, 'metrics_lstm_seed{s}.json')}",
        "--seed-json", f"DLinear={os.path.join(figs, 'metrics_dlinear_seed{s}.json')}",
        "--fixed-json", f"ARIMA(1,1,0)={os.path.join(figs, 'metrics_arima.json')}",
        "--ckpt", f"Transformer={args.ckpt_dir}",
    ]
    if not args.no_residual:
        argv_out += ["--ckpt", f"Transformer+残差={args.ckpt_res_dir}"]
    argv_out += ["--out", args.out, "--tag", "Baselines head-to-head"]
    return head_to_head.main(argv_out)


if __name__ == "__main__":
    raise SystemExit(main())
