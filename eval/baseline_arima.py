#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ARIMA 基线（单变量 OT，滚动预测）。

设计
* 只对 OT 建模；每个测试窗口用其**之前 history 小时**的 OT 重新拟合，再预报未来 pred_len 步。
* **确定性算法**：同输入同 order，结果完全可复现。因此只跑一次，不参与 2σ 判定。
* 窗口级并行用 joblib，`--n-jobs` 固定（默认 4）以便复现。
* 评测目标、指标函数与 `eval/evaluate.py` 完全一致：直接复用 `preds.npz` 的 `gts`，
  指标走 `utils/metrics.compute_metrics`。

用法
    python eval/baseline_arima.py --config configs/base.yaml \
        --preds outputs/figs/preds.npz --out outputs/figs --n-jobs 4

产物
    <out>/preds_arima.npz
    <out>/metrics_arima.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import clean_dataframe, load_data_config, load_raw, split_ranges  # noqa: E402
from utils.metrics import compute_metrics, format_metrics  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAR = "=" * 78
warnings.filterwarnings("ignore")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ARIMA 基线（滚动预测 OT）")
    p.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "base.yaml"))
    p.add_argument("--preds", default=os.path.join(PROJECT_ROOT, "outputs", "figs", "preds.npz"),
                   help="提供 gts / target_times 的参考文件（与 Transformer 评测目标一致）")
    p.add_argument("--out", default=os.path.join(PROJECT_ROOT, "outputs", "figs"))
    p.add_argument("--order", default="auto",
                   help="ARIMA order，如 '2,1,2'；'auto' 表示在验证段样本上用 AIC 选阶")
    p.add_argument("--history", type=int, default=168,
                   help="每个窗口用于拟合的历史小时数（默认 168 = 7 天）")
    p.add_argument("--n-jobs", type=int, default=4, help="joblib 并行度（固定值以便复现）")
    p.add_argument("--backend", default="threading",
                   help="joblib 后端。默认 threading：loky/多进程需要创建命名管道，"
                        "在受限 sandbox 下会被拒绝；每个窗口彼此独立且确定性，"
                        "后端不影响数值结果，只影响速度")
    p.add_argument("--select-sample", type=int, default=20,
                   help="--order auto 时从验证段取的窗口数（固定顺序，保证可复现）")
    return p.parse_args(argv)


def _fit_forecast(hist: np.ndarray, order: Tuple[int, int, int], steps: int) -> Tuple[np.ndarray, bool]:
    """拟合并预报；失败时退化为 persistence 并标记失败。"""
    from statsmodels.tsa.arima.model import ARIMA
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARIMA(np.asarray(hist, dtype="float64"), order=order).fit()
            fc = np.asarray(res.forecast(steps=steps), dtype="float64")
        if fc.shape[0] != steps or not np.isfinite(fc).all():
            raise ValueError("forecast 形状/数值异常")
        return fc, False
    except Exception:
        return np.repeat(float(hist[-1]), steps), True


def _task(payload):
    i, hist, order, steps = payload
    fc, failed = _fit_forecast(hist, order, steps)
    return i, fc, failed


def select_order(histories: List[np.ndarray], d: int = 1, max_pq: int = 2) -> Tuple[Tuple[int, int, int], Dict]:
    """在固定样本上按平均 AIC 选阶（确定性；替代未安装的 auto_arima）。"""
    from statsmodels.tsa.arima.model import ARIMA
    scores: Dict[str, float] = {}
    for p in range(max_pq + 1):
        for q in range(max_pq + 1):
            aics = []
            for h in histories:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        aics.append(float(ARIMA(h, order=(p, d, q)).fit().aic))
                except Exception:
                    continue
            if aics:
                scores[f"({p},{d},{q})"] = float(np.mean(aics))
    if not scores:
        return (1, d, 1), {}
    best = min(scores, key=scores.get)
    order = tuple(int(v) for v in best.strip("()").split(","))
    return order, scores


def validation_order_histories(ot: np.ndarray, index: pd.DatetimeIndex, cfg,
                               history: int, limit: int) -> List[np.ndarray]:
    """Build ARIMA order-selection histories from validation forecast origins only."""
    if history < 1 or limit < 1:
        raise ValueError("history 和 select-sample 必须为正整数")
    result: List[np.ndarray] = []
    for start, end in split_ranges(index, cfg)["val"]:
        for origin in range(start + cfg.seq_len, end - cfg.pred_len + 1):
            if origin < history:
                continue
            result.append(ot[origin - history:origin])
            if len(result) == limit:
                return result
    if not result:
        raise ValueError("验证段没有足够窗口用于 ARIMA 选阶")
    return result


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_data_config(args.config)

    ref = np.load(args.preds, allow_pickle=False)
    gts = ref["gts"]                                  # (N, 24) 摄氏度
    target_times = ref["target_times"]                # (N, 24) datetime64
    n_win, pred_len = gts.shape

    clean, _ = clean_dataframe(load_raw(cfg), cfg)
    ot = clean[cfg.target].to_numpy(dtype="float64")
    pos = clean.index.get_indexer(pd.DatetimeIndex(target_times[:, 0]))   # 每个窗口的预报起点
    if (pos < 0).any():
        raise SystemExit("有窗口起点无法在主序列中定位，检查 preds.npz 是否来自同一配置")
    if pos.min() < args.history:
        raise SystemExit(f"--history={args.history} 过大：最早的窗口之前只有 {pos.min()} 小时历史")

    histories = [ot[p - args.history:p] for p in pos]

    print(BAR)
    print("ARIMA 基线配置")
    print(BAR)
    print(f"  config   = {os.path.abspath(args.config)}")
    print(f"  目标     = {cfg.target} 单变量，滚动预测")
    print(f"  windows  = {n_win} （与 Transformer 评测完全相同的 test 窗口）")
    print(f"  history  = {args.history} 小时/窗口，每窗口重新拟合")
    print(f"  horizon  = {pred_len} 步")
    print(f"  n_jobs   = {args.n_jobs}（固定以便复现）")

    t0 = time.time()
    if str(args.order).strip().lower() == "auto":
        order_histories = validation_order_histories(
            ot, clean.index, cfg, args.history, args.select_sample)
        order, scores = select_order(order_histories)
        print(f"  order    = {order}（AIC 在验证段 {len(order_histories)} 个窗口上选出）")
        for k in sorted(scores, key=scores.get)[:4]:
            print(f"             AIC {k}: {scores[k]:.2f}")
    else:
        order = tuple(int(v) for v in str(args.order).split(","))
        print(f"  order    = {order}（手动指定）")

    from joblib import Parallel, delayed
    payloads = [(i, histories[i], order, pred_len) for i in range(n_win)]
    results = Parallel(n_jobs=args.n_jobs, backend=args.backend)(
        delayed(_task)(p) for p in payloads)

    preds = np.empty((n_win, pred_len), dtype="float64")
    n_failed = 0
    for i, fc, failed in results:
        preds[i] = fc
        n_failed += int(failed)
    elapsed = time.time() - t0

    metrics = compute_metrics(preds, gts)
    print(f"\n  拟合失败退化为 persistence 的窗口: {n_failed} / {n_win}")
    print(f"  用时: {elapsed:.1f}s")
    print()
    print(BAR)
    print(f"ARIMA test 指标（摄氏度，order={order}）")
    print(BAR)
    print(format_metrics(metrics))

    os.makedirs(args.out, exist_ok=True)
    npz_path = os.path.join(args.out, "preds_arima.npz")
    np.savez(
        npz_path,
        preds=preds,
        gts=gts,
        target_times=target_times,
        horizons=ref["horizons"],
        meta=np.array(json.dumps({
            "model": "ARIMA", "order": list(order), "history": args.history,
            "n_jobs": args.n_jobs, "n_windows": int(n_win), "pred_len": int(pred_len),
            "deterministic": True, "n_failed_fallback": int(n_failed),
            "elapsed_sec": round(elapsed, 2), "metrics": metrics,
        }, ensure_ascii=False)),
    )
    json_path = os.path.join(args.out, "metrics_arima.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"model": "ARIMA", "order": list(order), "history": args.history,
                   "deterministic": True, "n_seeds": 1, "n_windows": int(n_win),
                   "elapsed_sec": round(elapsed, 2), "metrics": metrics},
                  fh, indent=2, ensure_ascii=False)

    print()
    print(f"  preds_arima.npz : {os.path.abspath(npz_path)}")
    print(f"  metrics_arima.json : {os.path.abspath(json_path)}")
    print(BAR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
