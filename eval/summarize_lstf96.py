#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""历史自定义 6:2:2 设定（96 -> 96）汇总。

动机
    README 的主结果做的是 24 -> 24。ETT 数据集在长时序预测（LSTF）文献里的标准口径是
    96 / 192 / 336 / 720 步，所以「persistence 打不过」有可能只是短 horizon 的产物。
    这个脚本把同一套协议搬到 96 -> 96 上，用一个独立设定检验那个结论。

口径
    * 训练配方与 24 -> 24 **逐字段相同**（configs/lstf96.yaml 只改了窗口长度），
      由 tests/test_configs.py 断言。
    * scaler 仍只用原 train 段拟合，因此与 outputs/scaler.npz 逐元素相同
      （另存为 outputs/scaler96.npz，避免覆盖被 CI / Demo 依赖的那份）。
    * 与 `eval/head_to_head.py` 同一套聚合与 2σ 判定。

用法
    python scripts/sweep.py --config configs/lstf96.yaml --out-root <dir> \\
        --seeds 42 0 1 2 9999 \\
        --global-override output.scaler_path=outputs/scaler96.npz \\
        --run "lstf96|model.residual=false" --run "lstf96_res|model.residual=true"
    python eval/summarize_lstf96.py --sweep-root <dir> --dlinear-dir outputs/_runs/dl96
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
    p = argparse.ArgumentParser(description="96 -> 96（历史自定义 6:2:2）head-to-head 汇总")
    p.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "lstf96.yaml"))
    p.add_argument("--sweep-root",
                   default=os.path.join(PROJECT_ROOT, "outputs", "_runs", "sweep96"),
                   help="scripts/sweep.py --out-root 的取值")
    p.add_argument("--dlinear-dir",
                   default=os.path.join(PROJECT_ROOT, "outputs", "_runs", "dl96"))
    p.add_argument("--out", default=os.path.join(PROJECT_ROOT, "outputs", "lstf96_summary.json"))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    argv_out: List[str] = [
        "--config", args.config,
        "--ckpt", f"Transformer={os.path.join(args.sweep_root, 'lstf96', 'ckpt')}",
        "--ckpt", f"Transformer+残差={os.path.join(args.sweep_root, 'lstf96_res', 'ckpt')}",
        "--seed-json", f"DLinear={os.path.join(args.dlinear_dir, 'metrics_dlinear_seed{s}.json')}",
        "--out", args.out,
        "--tag", "ETTh1 96->96 (legacy custom 60/20/20)",
    ]
    return head_to_head.main(argv_out)


if __name__ == "__main__":
    raise SystemExit(main())
