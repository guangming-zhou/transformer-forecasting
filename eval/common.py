# -*- coding: utf-8 -*-
"""评估脚本共用组件：test 集构造、ckpt 推理、多 seed 聚合、2σ 判定、表格渲染。

抽出来是为了让 `evaluate.py` / `head_to_head.py` / 两个 summarize 脚本共用**同一份**
测试集构造与同一套判定口径 —— 口径一旦分叉，表里的数字就没法互相对照了。
"""

from __future__ import annotations

import json
import math
import os
import statistics as st
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from models.transformer import build_model
from utils.dataset import (
    apply_scaler, build_windows_ranges, clean_dataframe, load_raw, load_scaler,
    split_ranges,
)
from utils.metrics import compute_metrics, denorm_ot, horizons_for, metric_keys

# 多 seed 规范里固定的 seed 集合（README「报告口径」一节）
SEEDS: Tuple[int, ...] = (42, 0, 1, 2, 9999)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAR = "=" * 96


def require_complete_seeds(used: Iterable[int], expected: Iterable[int] = SEEDS,
                           label: str = "model") -> Tuple[int, ...]:
    """要求随机模型结果包含完整且唯一的预注册 seed 集合。"""
    used_tuple = tuple(int(s) for s in used)
    expected_tuple = tuple(int(s) for s in expected)
    if len(set(used_tuple)) != len(used_tuple):
        raise ValueError(f"{label} 的 seed 列表包含重复值: {used_tuple}")
    missing = [s for s in expected_tuple if s not in used_tuple]
    extra = [s for s in used_tuple if s not in expected_tuple]
    if missing or extra:
        details = []
        if missing:
            details.append("缺少 " + ", ".join(map(str, missing)))
        if extra:
            details.append("多出 " + ", ".join(map(str, extra)))
        raise ValueError(f"{label} 的 seed 集不完整（{'；'.join(details)}）")
    return expected_tuple


def resolve_device(spec: str) -> torch.device:
    spec = str(spec or "auto").strip().lower()
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


# --------------------------------------------------------------------------- #
# test 集
# --------------------------------------------------------------------------- #
def load_test_set(cfg, scaler: Dict) -> Dict[str, np.ndarray]:
    """用**磁盘上的 scaler** 归一化并切出 test 滑窗，同时取回原始 OT 与时间戳。

    只读 test 段（两种 split_mode 下相同），不重算 scaler、不写任何文件。
    """
    raw = load_raw(cfg)
    clean, _ = clean_dataframe(raw, cfg)
    index = clean.index

    scaled, _ = apply_scaler(clean.to_numpy(dtype="float64"), scaler)
    ranges = split_ranges(index, cfg)["test"]
    x, y = build_windows_ranges(
        scaled, ranges, cfg.seq_len, cfg.pred_len, target_idx=cfg.target_idx)

    ot_raw = clean[cfg.target].to_numpy(dtype="float64")
    times = index.to_numpy()
    hist_rows, last_rows, tgt_rows = [], [], []
    for start, end in ranges:
        m = end - start - cfg.seq_len - cfg.pred_len + 1
        if m <= 0:
            continue
        base = start + np.arange(m)
        hist_rows.append(base[:, None] + np.arange(cfg.seq_len)[None, :])
        last_rows.append(base + cfg.seq_len - 1)
        tgt_rows.append(base[:, None] + cfg.seq_len + np.arange(cfg.pred_len)[None, :])

    return {
        "x": x,                                                          # (N, seq_len, C)
        "y_norm": y,                                                     # (N, pred_len)
        "gts": denorm_ot(y, scaler),                                     # (N, pred_len) 摄氏度
        "input_ot": ot_raw[np.concatenate(hist_rows)],                   # (N, seq_len)
        "last_input_ot": ot_raw[np.concatenate(last_rows)],              # (N,)
        "target_times": times[np.concatenate(tgt_rows)],                 # (N, pred_len)
    }


def persistence_preds(last_input_ot: np.ndarray, pred_len: int) -> np.ndarray:
    """平凡基线：把输入段最后一步的 OT 重复 pred_len 次。"""
    return np.repeat(np.asarray(last_input_ot)[:, None], pred_len, axis=1)


def ckpt_predictions(ckpt_path: str, x: np.ndarray, scaler: Dict,
                     device: Optional[torch.device] = None,
                     batch_size: int = 256) -> np.ndarray:
    """加载 Transformer ckpt 在给定滑窗上推理，返回**摄氏度**预测。"""
    device = device or torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = build_model(ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])       # 必须加载权重
    model.eval()
    if model.input_len != x.shape[1]:
        raise ValueError(f"{ckpt_path} 的 input_len={model.input_len} 与滑窗长度 {x.shape[1]} 不一致")
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须为正整数，收到 {batch_size}")
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(
                x[start:start + batch_size], dtype=torch.float32, device=device)
            outputs.append(model(batch).cpu().numpy())
    out = np.concatenate(outputs, axis=0) if outputs else np.empty((0, model.pred_len))
    return denorm_ot(out, scaler)


# --------------------------------------------------------------------------- #
# 多 seed 聚合与判定
# --------------------------------------------------------------------------- #
def agg(runs: Sequence[Dict], key_path: Sequence[str]) -> Tuple[float, float]:
    """对多 seed 的某个指标取 (mean, 样本标准差)。单次运行时 std 记为 0。"""
    vals = []
    for m in runs:
        node = m
        for k in key_path:
            node = node[k]
        vals.append(float(node))
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def fmt(mean: float, std: float, nd: int = 4) -> str:
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


def verdict_text(delta: float, threshold: float, credible: Optional[bool] = None) -> str:
    ok = abs(delta) >= threshold if credible is None else credible
    return "超过噪声，差异可信" if ok else "差异落在种子噪声范围内"


def compare(name: str, mean: float, std: float, ref_name: str, ref_mean: float,
            ref_std: float = 0.0, deterministic: bool = False) -> str:
    """按 2σ 规范给出一条对比结论。"""
    delta = mean - ref_mean
    if deterministic:
        return (f"  · {name} vs {ref_name}: delta = {delta:+.4f} —— "
                f"{name} 为确定性算法，单次运行，**不参与 2σ 判定**")
    threshold = 2 * max(std, ref_std)
    return (f"  · {name} vs {ref_name}: delta = {delta:+.4f}, 2σ = {threshold:.4f}  → "
            f"{verdict_text(delta, threshold)}")


def compare_pair(name_a: str, runs_a: Sequence[Dict], name_b: str,
                 runs_b: Sequence[Dict]) -> str:
    ma, sa = agg(runs_a, ["overall", "MAE"])
    mb, sb = agg(runs_b, ["overall", "MAE"])
    threshold = 2 * max(sa, sb)
    return (f"  · {name_a} vs {name_b}: delta = {ma - mb:+.4f}, "
            f"2×max(σ_A, σ_B) = {threshold:.4f}  → {verdict_text(ma - mb, threshold)}")


def _paired_ci(differences: Sequence[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """返回配对差值的 (mean, lower, upper) Student-t 置信区间。"""
    values = [float(v) for v in differences]
    if len(values) < 2:
        raise ValueError("配对置信区间至少需要 2 次运行")
    mean = st.mean(values)
    sd = st.stdev(values)
    if sd == 0.0:
        return mean, mean, mean
    from scipy.stats import t
    critical = float(t.ppf((1.0 + confidence) / 2.0, df=len(values) - 1))
    half = critical * sd / math.sqrt(len(values))
    return mean, mean - half, mean + half


def _metric_value(metrics: Dict, key_path: Sequence[str]) -> float:
    node = metrics
    for key in key_path:
        node = node[key]
    return float(node)


def _ci_verdict(lower: float, upper: float) -> str:
    return "excludes 0" if lower > 0.0 or upper < 0.0 else "includes 0"


def compare_pair_ci(name_a: str, runs_a: Sequence[Dict], name_b: str,
                    runs_b: Sequence[Dict],
                    key_path: Sequence[str] = ("overall", "MAE")) -> str:
    """用相同顺序的 seed 做配对差值并报告 95% Student-t CI。"""
    if len(runs_a) != len(runs_b):
        raise ValueError(f"配对比较要求运行数一致，收到 {len(runs_a)} 和 {len(runs_b)}")
    diffs = [_metric_value(a, key_path) - _metric_value(b, key_path)
             for a, b in zip(runs_a, runs_b)]
    mean, lower, upper = _paired_ci(diffs)
    return (f"  · {name_a} vs {name_b}: paired delta = {mean:+.4f}, "
            f"paired 95% CI = [{lower:+.4f}, {upper:+.4f}] → {_ci_verdict(lower, upper)}")


def compare_runs_to_fixed_ci(name: str, runs: Sequence[Dict], ref_name: str,
                             ref_mean: float,
                             key_path: Sequence[str] = ("overall", "MAE")) -> str:
    """随机模型各 seed 相对固定基线的差值及 95% Student-t CI。"""
    diffs = [_metric_value(run, key_path) - float(ref_mean) for run in runs]
    mean, lower, upper = _paired_ci(diffs)
    return (f"  · {name} vs {ref_name}: paired delta = {mean:+.4f}, "
            f"paired 95% CI = [{lower:+.4f}, {upper:+.4f}] → {_ci_verdict(lower, upper)}")


# --------------------------------------------------------------------------- #
# 表格
# --------------------------------------------------------------------------- #
def table_columns(metrics: Dict) -> List[str]:
    return metric_keys(metrics)


def render_table(rows: List[Dict], columns: Sequence[str]) -> str:
    """rows: [{"name": str, "MAE": (mean,std), "RMSE": (mean,std),
               "h": {key: (mean,std)}, "note": str}]"""
    head = (f"{'模型':<30s}{'MAE':>19s}{'RMSE':>19s}"
            + "".join(f"{c:>17s}" for c in columns))
    lines = [head, "-" * len(head)]
    for r in rows:
        cells = "".join(f"{fmt(*r['h'][c]):>17s}" for c in columns)
        note = f"  {r['note']}" if r.get("note") else ""
        lines.append(f"{r['name']:<30s}{fmt(*r['MAE']):>19s}{fmt(*r['RMSE']):>19s}{cells}{note}")
    return "\n".join(lines)


def metrics_from_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)["metrics"]


def collect_seed_metrics(pattern: str, seeds: Iterable[int] = SEEDS,
                         expected_seq_len: Optional[int] = None,
                         expected_pred_len: Optional[int] = None,
                         expected_split_mode: Optional[str] = None) -> Tuple[List[int], List[Dict]]:
    """收集多 seed 指标，并可拒绝混入不同窗口协议的 JSON。"""
    used, runs = [], []
    for s in seeds:
        path = pattern.format(s=s)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            for key, expected in (("seq_len", expected_seq_len),
                                  ("pred_len", expected_pred_len),
                                  ("split_mode", expected_split_mode)):
                if expected is None:
                    continue
                actual = payload.get(key)
                matches = str(actual) == str(expected) if key == "split_mode" else (
                    actual is not None and int(actual) == int(expected))
                if not matches:
                    raise ValueError(
                        f"{path} 的 {key}={actual!r}，与当前协议 {expected} 不一致")
            used.append(s)
            runs.append(payload["metrics"])
    return used, runs


def collect_ckpt_metrics(ckpt_dir: str, x: np.ndarray, gts: np.ndarray, scaler: Dict,
                         seeds: Iterable[int] = SEEDS,
                         device: Optional[torch.device] = None) -> Tuple[List[int], List[Dict]]:
    """对 `ckpt_dir/seed{s}.pt` 逐个现场推理并算指标，缺哪个 seed 就跳过哪个。"""
    used, runs = [], []
    for s in seeds:
        path = os.path.join(ckpt_dir, f"seed{s}.pt")
        if os.path.isfile(path):
            preds = ckpt_predictions(path, x, scaler, device)
            used.append(s)
            runs.append(compute_metrics(preds, gts))
    return used, runs


def rel_to_project(path: str) -> str:
    """转成相对仓库根的路径，用于写进**入库**的 JSON。

    入库文件里出现本机绝对路径，别人 clone 下来一眼就知道「这些产物是在哪台机器上跑的」，
    也会让 diff 无意义地抖动 —— 所以统一转成相对路径；实在在仓库外的原样返回绝对路径。
    """
    absolute = os.path.abspath(str(path))
    try:
        rel = os.path.relpath(absolute, PROJECT_ROOT)
    except ValueError:                     # 不同盘符，relpath 会抛
        return absolute
    return absolute if rel.startswith("..") else rel.replace(os.sep, "/")


def write_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


__all__ = [
    "SEEDS", "PROJECT_ROOT", "BAR", "resolve_device", "load_test_set",
    "persistence_preds", "ckpt_predictions", "agg", "fmt", "verdict_text",
    "compare", "compare_pair", "compare_pair_ci", "compare_runs_to_fixed_ci",
    "table_columns", "render_table",
    "metrics_from_json", "collect_seed_metrics", "collect_ckpt_metrics", "write_json",
    "rel_to_project", "require_complete_seeds", "compute_metrics", "denorm_ot", "horizons_for",
]
