# -*- coding: utf-8 -*-
"""评估指标（接口冻结，eval 与后续 baseline 共用）。

约定
* 所有指标都在**反归一化后的原单位（摄氏度）**上计算。
* `preds` / `gts` 形状均为 `(N, H)`，`H` 为预测步数。
* `compute_metrics` 返回结构：
      {"overall": {"MAE": float, "RMSE": float},
       "h1": {...}, "h6": {...}, "h12": {...}, "h24": {...}}
  其中 `hK` 表示第 K 步（1-based）的指标，即 `preds[:, K-1]`。

horizon 集合随预测步数自适应（`horizons_for`）
* `pred_len=24`（主设定）→ `(1, 6, 12, 24)`，与历史产出一致。
* `pred_len=96`（LSTF 标准设定）→ `(1, 24, 48, 96)`。
* 其它步数 → 从 `(1, 6, 12, 24, 48, 96)` 里取不超过该步数的部分。
调用方也可以显式传 `horizons=` 覆盖。
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np

__all__ = ["HORIZONS", "HORIZON_SETS", "horizons_for", "target_index",
           "compute_metrics", "denorm_ot", "format_metrics"]

# 需要单独汇报的 horizon 步数（1-based）——主设定 pred_len=24 的默认值
HORIZONS: Tuple[int, ...] = (1, 6, 12, 24)

# 已知预测步数 -> 该设定下要单独汇报的 horizon
HORIZON_SETS: Dict[int, Tuple[int, ...]] = {
    24: (1, 6, 12, 24),
    96: (1, 24, 48, 96),
    336: (1, 48, 168, 336),
    720: (1, 96, 336, 720),
}
_CANDIDATE_HORIZONS: Tuple[int, ...] = (1, 6, 12, 24, 48, 96, 168, 336, 720)


def horizons_for(n_steps: int) -> Tuple[int, ...]:
    """按预测步数返回要单独汇报的 horizon 集合（1-based，升序）。"""
    n_steps = int(n_steps)
    if n_steps in HORIZON_SETS:
        return HORIZON_SETS[n_steps]
    picked = tuple(h for h in _CANDIDATE_HORIZONS if h <= n_steps)
    return picked or (1,)

# OT 在 7 列特征里的下标（HUFL, HULL, MUFL, MULL, LUFL, LULL, OT）——默认数据集的取值
TARGET_IDX = 6


def _as_scaler(scaler: Union[Dict, str, os.PathLike]) -> Dict:
    """接受 scaler 字典（utils.dataset.load_scaler 的返回值）或 npz 路径。"""
    if isinstance(scaler, (str, os.PathLike)):
        with np.load(scaler, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    if not isinstance(scaler, dict) or "mean" not in scaler or "std" not in scaler:
        raise TypeError("scaler 需为含 'mean'/'std' 的字典，或 npz 文件路径")
    return scaler


def target_index(scaler: Union[Dict, str, os.PathLike]) -> int:
    """取目标列下标：优先用 scaler 里记录的那一个，缺省时退回模块常量。

    `outputs/scaler.npz` 由 `utils.dataset.save_scaler` 写出时会带上 `target_idx`，
    所以换一份数据（目标列不在第 7 列）时反归一化依然正确，不需要改代码。
    """
    stats = _as_scaler(scaler)
    if "target_idx" in stats:
        return int(np.asarray(stats["target_idx"]).reshape(-1)[0])
    return int(TARGET_IDX)


def denorm_ot(x: np.ndarray, scaler: Union[Dict, str, os.PathLike]) -> np.ndarray:
    """用 scaler 里目标列的 mean/std 把归一化值还原成原单位。

    scaler 一律取自 `outputs/scaler.npz`（train 段拟合），**不在此处重算**；
    目标列下标同样从 scaler 里读（见 `target_index`）。
    """
    stats = _as_scaler(scaler)
    idx = target_index(stats)
    return (np.asarray(x, dtype="float64") * float(stats["std"][idx])
            + float(stats["mean"][idx]))


def _mae_rmse(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    err = pred - gt
    return {"MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2)))}


def compute_metrics(preds: np.ndarray, gts: np.ndarray,
                    horizons: Optional[Iterable[int]] = None) -> Dict[str, Dict[str, float]]:
    """在**反归一化后的原单位**上计算 overall 与分 horizon 的 MAE / RMSE。

    preds / gts: (N, H) —— 传入前请先用 `denorm_ot` 还原。
    horizons: 需要单独汇报的步数；None 时按 `horizons_for(H)` 自适应。
    """
    preds = np.asarray(preds, dtype="float64")
    gts = np.asarray(gts, dtype="float64")
    if preds.shape != gts.shape:
        raise ValueError(f"preds/gts 形状不一致: {preds.shape} vs {gts.shape}")
    if preds.ndim != 2:
        raise ValueError(f"preds/gts 需为 (N, H) 二维，收到 {preds.shape}")
    if not np.isfinite(preds).all() or not np.isfinite(gts).all():
        raise ValueError("preds/gts 含 NaN/Inf")

    out: Dict[str, Dict[str, float]] = {"overall": _mae_rmse(preds, gts)}
    n_steps = preds.shape[1]
    for h in (horizons_for(n_steps) if horizons is None else horizons):
        h = int(h)
        if 1 <= h <= n_steps:
            out[f"h{h}"] = _mae_rmse(preds[:, h - 1], gts[:, h - 1])
    return out


def metric_keys(metrics: Dict[str, Dict[str, float]]) -> list:
    """返回 metrics 里出现的 `hK` 键，按 K 升序（用于排版，不依赖全局 HORIZONS）。"""
    keys = [k for k in metrics if len(k) > 1 and k[0] == "h" and k[1:].isdigit()]
    return sorted(keys, key=lambda k: int(k[1:]))


def format_metrics(metrics: Dict[str, Dict[str, float]], title: str = "") -> str:
    """把 compute_metrics 的结果排成可直接打印的表格。"""
    lines = []
    if title:
        lines.append(title)
    lines.append(f"{'horizon':>10s}{'MAE(°C)':>12s}{'RMSE(°C)':>12s}")
    lines.append("-" * 34)
    for key in ["overall"] + metric_keys(metrics):
        label = "overall" if key == "overall" else f"{key[1:]} 步"
        m = metrics[key]
        lines.append(f"{label:>10s}{m['MAE']:>12.4f}{m['RMSE']:>12.4f}")
    return "\n".join(lines)
