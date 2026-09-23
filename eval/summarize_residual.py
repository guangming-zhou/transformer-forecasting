#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""残差预测实验汇总：`residual=True`（5 seed）在 test 上 vs persistence 与 A 基线。

本脚本是 `eval/head_to_head.py` 的**薄封装**（聚合与 2σ 判定只有一份实现），
额外做一件 head_to_head 不做的自检：断言残差组 ckpt 的 `model_config.residual` 全为 True、
A 基线全为 False —— 否则「残差头有效」这个结论可能只是拿错目录了。

用法
    python eval/summarize_residual.py
    python eval/summarize_residual.py --config configs/base.yaml --out outputs/residual_summary.json
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.common import PROJECT_ROOT, SEEDS  # noqa: E402
from eval import head_to_head  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="残差预测实验汇总（head_to_head 薄封装）")
    p.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "base.yaml"))
    p.add_argument("--ckpt-base", default=os.path.join(PROJECT_ROOT, "outputs", "ckpt_A_ms"))
    p.add_argument("--ckpt-res", default=os.path.join(PROJECT_ROOT, "outputs", "ckpt_res"))
    p.add_argument("--out", default=os.path.join(PROJECT_ROOT, "outputs", "residual_summary.json"))
    return p.parse_args(argv)


def check_residual_flags(ckpt_base: str, ckpt_res: str, seeds=SEEDS) -> int:
    """断言两组的 residual 标志，返回检查到的 ckpt 数。"""
    def flags(d: str) -> List[bool]:
        out = []
        for s in seeds:
            p = os.path.join(d, f"seed{s}.pt")
            if os.path.isfile(p):
                ck = torch.load(p, map_location="cpu", weights_only=True)
                out.append(bool((ck["model_config"] or {}).get("residual", False)))
        return out

    base_flags, res_flags = flags(ckpt_base), flags(ckpt_res)
    print(f"  residual 标志自检: 残差组 {res_flags}  |  A 基线 {base_flags}")
    if base_flags and any(base_flags):
        raise SystemExit(f"A 基线目录 {ckpt_base} 里出现了 residual=True 的 ckpt，目录拿错了")
    if res_flags and not all(res_flags):
        raise SystemExit(f"残差目录 {ckpt_res} 里出现了 residual=False 的 ckpt，目录拿错了")
    return len(res_flags) + len(base_flags)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    n = check_residual_flags(args.ckpt_base, args.ckpt_res)
    if n == 0:
        raise SystemExit("两组 ckpt 都不存在，先跑 train/train.py 与残差变体")

    return head_to_head.main([
        "--config", args.config,
        "--ckpt", f"Transformer A={args.ckpt_base}",
        "--ckpt", f"Transformer+残差={args.ckpt_res}",
        "--per-seed",
        "--out", args.out,
        "--tag", "Residual-Prediction Fix: residual=True vs A vs persistence",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
