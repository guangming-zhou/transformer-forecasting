# -*- coding: utf-8 -*-
"""ETTh1 数据管道：清洗 → 按时间 6:2:2 划分 → 只用 train 拟合归一化/裁剪 → 滑窗 Dataset。

任务定义
    x: 过去 24 小时的全部 7 个变量  -> (24, 7) float32
    y: 未来 24 小时的 OT            -> (24,)  float32

清洗规则（叠加顺序：1 → 2 → 3，全部在原始数值上检测，最后统一插值一次）
    1. 重复块整段（含首行）        -> 视为缺失
    2. 行级零值：一行中 >= zero_min_cols 列为 0 -> 该行内恰为 0 的单元格视为缺失
    3. 单列连续 0 失效段（分级阈值）-> 输入列 >= 12h、目标列 OT >= 6h 的整段视为缺失

设计决策（依据数据探查结论，详见 README.md）
1. **不删除任何行**。数据探查确认本文件与官方 ETTh1 逐字节一致，14 段「整日冻结」的
   24 行重复块是数据集原生特性。删行会在小时网格上凿洞，使 746/10405 个 train 滑窗
   跨越时间洞、x 与 y 时间错位；因此改为「标记为缺失 + 按列线性插值」，保持网格连续。
2. 规则 2 **按行**判定，避免误杀单列合法的 0 值（HUFL 89 个、HULL 410 个…）。
3. 规则 3 补齐行级规则的盲区：MUFL 73h / MULL 72h / LULL 141h 的真实失效段中，多数
   行只有 1 列为 0，行级规则覆盖不到；分级阈值（输入 12h / OT 6h）保守，不会误伤
   夜间低负载真实贴 0 的短段。清洗策略统一生效，不做默认关闭的开关，保证多 seed 与
   消融实验的数据分布一致、可复现。
4. 归一化的 mean/std 与 3σ 裁剪边界**只在 train 段拟合**，val/test 复用同一组参数，
   不做任何 val/test 统计量泄漏。裁剪后归一化值天然落在 [-3, 3]。
5. 三个 split 各自独立切窗，窗口**不跨越 split 边界**。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields as dataclass_fields
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "FEATURES", "TARGET", "TARGET_IDX", "DataConfig", "ETTh1Dataset",
    "load_data_config", "data_config_from_dict", "data_config_from_config", "load_raw",
    "find_identical_runs", "find_long_zero_runs",
    "build_anomaly_mask", "clean_dataframe", "split_dataframe",
    "split_ranges", "season_of_month", "season_composition", "SEASONS", "SPLIT_MODES",
    "fit_scaler", "apply_scaler", "build_windows", "build_windows_ranges", "prepare_data",
    "make_dataloaders", "load_scaler", "denormalize_ot", "max_zero_run_lengths",
]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "base.yaml")

FEATURES: Tuple[str, ...] = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")
INPUT_FEATURES: Tuple[str, ...] = FEATURES[:-1]          # 6 个输入列
TARGET: str = "OT"
TARGET_IDX: int = FEATURES.index(TARGET)
TIME_COL: str = "date"
# ---- 切分模式 ----
SPLIT_MODES: Tuple[str, ...] = ("time_sequential", "season_stratified", "official_etth1")
ETTH1_OFFICIAL_TRAIN_ROWS = 12 * 30 * 24
ETTH1_OFFICIAL_VAL_ROWS = 4 * 30 * 24
ETTH1_OFFICIAL_TEST_ROWS = 4 * 30 * 24
ETTH1_OFFICIAL_END = (
    ETTH1_OFFICIAL_TRAIN_ROWS + ETTH1_OFFICIAL_VAL_ROWS + ETTH1_OFFICIAL_TEST_ROWS)

# ---- 季节（按月份划分，北半球）----
SEASONS: Tuple[str, ...] = ("spring", "summer", "autumn", "winter")
SEASON_MONTHS: Dict[str, Tuple[int, ...]] = {
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
    "winter": (12, 1, 2),
}


def season_of_month(month: int) -> str:
    """按月份返回季节标签。"""
    for name, months in SEASON_MONTHS.items():
        if month in months:
            return name
    raise ValueError(f"非法月份: {month}")


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    """数据管道配置。相对路径一律相对项目根目录解析。"""

    data_path: str = "data/ETTh1.csv"
    scaler_path: str = "outputs/scaler.npz"

    seq_len: int = 24          # 输入窗口长度（小时）
    pred_len: int = 24         # 预测长度（小时）

    # 列名与目标列。留空（None）时使用模块级默认值 FEATURES / TARGET（=ETTh1 的 7 列）。
    # 想让项目跑自己的 CSV，只需在 yaml 里给出 feature_cols 与 target_col，
    # 数据管道、scaler、模型输入维度都会跟着走，不需要改代码。
    feature_cols: Optional[Tuple[str, ...]] = None
    target_col: str = TARGET

    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2

    # 切分模式
    #   time_sequential   严格按时间 6:2:2（主切分，默认，逻辑不受 season 模式影响）
    #   season_stratified 对照实验：在**原 train 时间段内**按季节分层，
    #                     每季前 season_train_ratio 归 season_train、其余归 season_val；
    #                     test 仍用原 test 段，原 val 段不参与。
    #                     归一化 mean/std 仍用原 train 段拟合，绝不重新拟合。
    split_mode: str = "time_sequential"
    season_train_ratio: float = 0.75

    # 滑窗降采样：只作用于 **train** split 的窗口索引（每 window_stride 个取 1 个），
    # val/test 永远保持完整，便于逐时评估。1 = 不降采样（原行为）。
    # 动机：相邻窗口共享 48 小时中的 47 小时，独立样本量其实只有 rows/48 左右。
    window_stride: int = 1

    # ---- 清洗规则 1：连续完全相同的行 ----
    dedup_policy: str = "mask"  # "mask" = 重复行当缺失插值；"keep" = 不做处理
    # ---- 清洗规则 2：行级零值 ----
    zero_min_cols: int = 5      # 一行中 >= 该数量的 0 值列 -> 该行 0 值视为缺失
    # ---- 清洗规则 3：单列连续 0（分级阈值，统一生效） ----
    zero_run_clean: bool = True
    zero_run_input_hours: int = 12   # 6 个输入列：连续 0 >= 12h 视为失效段
    zero_run_target_hours: int = 6   # 目标列 OT ：连续 0 >= 6h 视为失效段

    clip_sigma: float = 3.0     # 3σ 裁剪；设为 0 或负数则关闭
    eps: float = 1e-8           # std 下限保护

    def __post_init__(self) -> None:
        if self.feature_cols is not None:
            self.feature_cols = tuple(str(c) for c in self.feature_cols)
        self.target_col = str(self.target_col)
        if self.feature_cols is not None and len(self.feature_cols) == 0:
            raise ValueError("feature_cols 不能为空列表")
        if self.target_col not in self.features:
            raise ValueError(
                f"target_col={self.target_col!r} 不在 feature_cols={self.features} 里")
        self.data_path = self.resolve(self.data_path)
        self.scaler_path = self.resolve(self.scaler_path)
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train/val/test 比例之和必须为 1，当前为 {total}")
        if self.dedup_policy not in ("mask", "keep"):
            raise ValueError("dedup_policy 只支持 'mask' 或 'keep'（不推荐 'drop'，见模块文档）")
        if self.split_mode not in SPLIT_MODES:
            raise ValueError(f"split_mode 只支持 {SPLIT_MODES}，收到 {self.split_mode!r}")
        if not 0.0 < self.season_train_ratio < 1.0:
            raise ValueError(f"season_train_ratio 必须在 (0,1) 内，收到 {self.season_train_ratio}")
        if self.window_stride < 1:
            raise ValueError(f"window_stride 必须 >= 1，收到 {self.window_stride}")
        if self.zero_run_clean:
            if self.zero_run_input_hours <= 0 or self.zero_run_target_hours <= 0:
                raise ValueError("zero_run_*_hours 必须为正整数小时数")

    @staticmethod
    def resolve(path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)

    @property
    def csv_path(self) -> str:
        """向后兼容别名（配置文件里该字段名为 data_path）。"""
        return self.data_path

    @property
    def features(self) -> Tuple[str, ...]:
        """实际使用的特征列（None 时回落到模块默认 FEATURES）。"""
        return tuple(self.feature_cols) if self.feature_cols is not None else FEATURES

    @property
    def target(self) -> str:
        return self.target_col

    @property
    def target_idx(self) -> int:
        """目标列在 features 里的下标（不写死 6，换数据也不用改代码）。"""
        return self.features.index(self.target_col)

    def cleaning_summary(self) -> str:
        if self.zero_run_clean:
            rule3 = (f"单列连续0(输入>={self.zero_run_input_hours}h, "
                     f"OT>={self.zero_run_target_hours}h)")
        else:
            rule3 = "单列连续0(off)"
        return (f"重复块[{self.dedup_policy}] -> 行级>={self.zero_min_cols}列为0 -> {rule3}")


def load_data_config(path: Optional[str] = None, **overrides) -> DataConfig:
    """从 configs/base.yaml 读取数据管道配置。

    兼容两种写法：顶层平铺，或统一放在 `data:` 段下（存在 data 段时优先取该段）。
    与 DataConfig 无关的键（例如同一文件里的 `model:` 段）会被忽略，
    方便模型超参与数据配置共存于一个 yaml。
    """
    path = path or DEFAULT_CONFIG_PATH
    if not os.path.isfile(path):
        return DataConfig(**overrides)

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} 的内容不是键值对")

    return data_config_from_config(raw, **overrides)


def data_config_from_config(config: Dict, **overrides) -> DataConfig:
    """从完整配置（含 `data:` / `output:` 段）构造 DataConfig。

    `scaler_path` 按约定写在 `output:` 段，这里做一次跨段取值；
    同时兼容它直接写在 `data:` 段，以及整份配置平铺（无 data 段）的写法。
    """
    if isinstance(config.get("data"), dict):
        section = dict(config["data"])
    else:                                   # 平铺写法：顶层非字典键即数据字段
        section = {k: v for k, v in config.items()
                   if k not in ("model", "train", "output") and not isinstance(v, dict)}

    output = config.get("output") or {}
    if "scaler_path" not in section and isinstance(output, dict) and output.get("scaler_path"):
        section["scaler_path"] = output["scaler_path"]

    return data_config_from_dict(section, **overrides)


def data_config_from_dict(section: Optional[Dict], **overrides) -> DataConfig:
    """从配置段（通常是 yaml 的 `data:`）过滤出 DataConfig 字段并构造。"""
    allowed = {f.name for f in dataclass_fields(DataConfig)}
    kwargs = {k: v for k, v in (section or {}).items() if k in allowed}
    kwargs.update(overrides)
    return DataConfig(**kwargs)


# --------------------------------------------------------------------------- #
# 1. 读取
# --------------------------------------------------------------------------- #
def load_raw(cfg: DataConfig) -> pd.DataFrame:
    """读取 CSV，返回以时间为索引、7 个 float64 变量为列的 DataFrame。"""
    if not os.path.isfile(cfg.data_path):
        raise FileNotFoundError(f"找不到数据文件: {cfg.data_path}")

    df = pd.read_csv(cfg.data_path)
    cols = cfg.features
    missing = [c for c in (TIME_COL, *cols) if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必需列: {missing}；实际列为 {list(df.columns)}")

    df = df[[TIME_COL, *cols]].copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="raise")   # pandas 3.0 读成 str，必须显式解析
    if df[TIME_COL].duplicated().any():
        raise ValueError("时间列存在重复时间戳")
    df = df.set_index(TIME_COL).sort_index().astype("float64")

    if not df.index.is_monotonic_increasing:
        raise ValueError("时间索引不是单调递增")
    if df.isna().any().any():
        raise ValueError("原始数据出现 NaN —— 本数据集预期无缺失，请先排查")
    return df


# --------------------------------------------------------------------------- #
# 2. 异常检测与清洗
# --------------------------------------------------------------------------- #
def find_identical_runs(values: np.ndarray) -> np.ndarray:
    """规则 1：找出「连续 >=2 行完全相同」的段落，返回覆盖整段的布尔行掩码。

    注意：整段（含首行）都会被标记，而不是只标记后续重复行——那 14 段 24 小时的
    「整日冻结」块整体都是可疑的，从块前一行插值到块后一行更合理。
    """
    n = values.shape[0]
    mask = np.zeros(n, dtype=bool)
    i = 0
    while i < n - 1:
        if np.array_equal(values[i], values[i + 1]):
            j = i + 1
            while j + 1 < n and np.array_equal(values[j + 1], values[i]):
                j += 1
            mask[i:j + 1] = True
            i = j + 1
        else:
            i += 1
    return mask


def find_long_zero_runs(values: np.ndarray,
                        thresholds: Sequence[float]) -> Tuple[np.ndarray, List[List[Dict]]]:
    """规则 3：单列连续为 0 且长度 >= 该列阈值 -> 整段标记为缺失。

    thresholds[j] <= 0 表示第 j 列不启用该规则。
    返回 (cell_mask, runs)：runs[j] 是第 j 列的失效段清单（start_idx/end_idx/length）。
    """
    n, n_cols = values.shape
    if len(thresholds) != n_cols:
        raise ValueError(f"thresholds 长度 {len(thresholds)} 与列数 {n_cols} 不一致")

    mask = np.zeros((n, n_cols), dtype=bool)
    runs: List[List[Dict]] = []
    for j in range(n_cols):
        thr = float(thresholds[j])
        col_runs: List[Dict] = []
        if thr > 0:
            is_zero = values[:, j] == 0.0
            i = 0
            while i < n:
                if is_zero[i]:
                    k = i
                    while k + 1 < n and is_zero[k + 1]:
                        k += 1
                    length = k - i + 1
                    if length >= thr:
                        mask[i:k + 1, j] = True
                        col_runs.append({"start_idx": i, "end_idx": k, "length": length})
                    i = k + 1
                else:
                    i += 1
        runs.append(col_runs)
    return mask, runs


def build_anomaly_mask(values: np.ndarray, cfg: DataConfig,
                       index: Optional[pd.DatetimeIndex] = None) -> Tuple[np.ndarray, Dict]:
    """按 1 → 2 → 3 的顺序构造异常掩码，返回 (cell_mask, info)。

    三条规则都在**原始数值**上检测（规则之间不互相影响段长判断），最后取并集，
    由 clean_dataframe 统一插值一次。
    """
    n, n_cols = values.shape
    zeros_per_row = (values == 0.0).sum(axis=1)

    # ---- 规则 1：重复块 ----
    if cfg.dedup_policy == "mask":
        dup_rows = find_identical_runs(values)
    else:
        dup_rows = np.zeros(n, dtype=bool)

    # ---- 规则 2：行级零值 ----
    zero_rows = zeros_per_row >= cfg.zero_min_cols
    zero_cells = zero_rows[:, None] & (values == 0.0)

    # ---- 规则 3：单列连续 0（分级阈值） ----
    if cfg.zero_run_clean:
        thresholds = np.full(n_cols, float(cfg.zero_run_input_hours))
        thresholds[cfg.target_idx] = float(cfg.zero_run_target_hours)
    else:
        thresholds = np.zeros(n_cols)
    zero_run_mask, zero_run_runs = find_long_zero_runs(values, thresholds)

    cell_mask = dup_rows[:, None] | zero_cells | zero_run_mask

    dup_cells = int(dup_rows.sum()) * n_cols
    zero_run_segments = int(sum(len(r) for r in zero_run_runs))
    zero_run_hours = int(sum(s["length"] for r in zero_run_runs for s in r))

    info: Dict = {
        "n_rows": int(n),
        # 规则 1
        "dup_rows": int(dup_rows.sum()),
        "dup_segments": int(_count_segments(dup_rows)),
        "dup_cells": dup_cells,
        # 规则 2
        "zero_rows": int(zero_rows.sum()),
        "zero_segments": int(_count_segments(zero_rows)),
        "zero_cells": int(zero_cells.sum()),
        "all_zero_rows": int((zeros_per_row == n_cols).sum()),
        # 规则 3
        "zero_run_rows": int(zero_run_mask.any(axis=1).sum()),
        "zero_run_segments": zero_run_segments,
        "zero_run_hours": zero_run_hours,
        "zero_run_cells": int(zero_run_mask.sum()),
        "zero_run_per_column": {
            cfg.features[j]: {
                "threshold_hours": int(thresholds[j]),
                "segments": len(zero_run_runs[j]),
                "hours": int(sum(s["length"] for s in zero_run_runs[j])),
                "runs": zero_run_runs[j],
            }
            for j in range(n_cols)
        },
        # 汇总
        "cells_masked": int(cell_mask.sum()),
        "rows_touched": int(cell_mask.any(axis=1).sum()),
    }

    info["rules"] = [
        {"name": "1. 重复块整段（含首行）", "segments": info["dup_segments"],
         "hours": info["dup_rows"], "cells": info["dup_cells"], "unit": "行"},
        {"name": f"2. 行级 >= {cfg.zero_min_cols} 列为 0", "segments": info["zero_segments"],
         "hours": info["zero_rows"], "cells": info["zero_cells"], "unit": "行"},
        {"name": (f"3. 单列连续 0（输入>={cfg.zero_run_input_hours}h / "
                  f"OT>={cfg.zero_run_target_hours}h）"),
         "segments": zero_run_segments, "hours": zero_run_hours,
         "cells": info["zero_run_cells"], "unit": "段长之和"},
    ]

    if index is not None:
        info["dup_segment_list"] = describe_segments(dup_rows, index)
        info["zero_segment_list"] = describe_segments(zero_rows, index)
        for j, col in enumerate(cfg.features):
            info["zero_run_per_column"][col]["list"] = describe_segments(
                zero_run_mask[:, j], index)
    return cell_mask, info


def _count_segments(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    return int(np.count_nonzero(mask[1:] & ~mask[:-1]) + int(mask[0]))


def describe_segments(mask: np.ndarray, index: pd.DatetimeIndex) -> List[Dict]:
    """把布尔行掩码展开成带起止时间的段落清单，便于打印审计报告。"""
    out: List[Dict] = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append({
                "start_idx": i, "end_idx": j, "length": j - i + 1,
                "start_time": index[i], "end_time": index[j],
            })
            i = j + 1
        else:
            i += 1
    return out


def max_zero_run_lengths(values: np.ndarray) -> List[int]:
    """每列最长的「连续精确 0」长度（小时），用于清洗后自检。"""
    n, n_cols = values.shape
    out: List[int] = []
    for j in range(n_cols):
        is_zero = values[:, j] == 0.0
        best = cur = 0
        for flag in is_zero:
            cur = cur + 1 if flag else 0
            if cur > best:
                best = cur
        out.append(int(best))
    return out


def clean_dataframe(df: pd.DataFrame, cfg: DataConfig) -> Tuple[pd.DataFrame, Dict]:
    """把异常单元格置为 NaN 后按列线性插值。行数保持不变（小时网格不被破坏）。"""
    values = df.to_numpy(dtype="float64", copy=True)
    # 三条规则都在原始数值上检测；段落清单在写入 NaN 之前生成
    # （NaN != NaN，写入后会破坏重复行的判定）。
    cell_mask, info = build_anomaly_mask(values, cfg, index=df.index)

    values[cell_mask] = np.nan
    masked = pd.DataFrame(values, index=df.index, columns=df.columns)
    cleaned = masked.interpolate(method="linear", limit_direction="both", axis=0)

    if cleaned.isna().any().any():
        bad = cleaned.columns[cleaned.isna().any()].tolist()
        raise RuntimeError(f"插值后仍有缺失值: {bad}")

    info["interpolated_cells"] = int(cell_mask.sum())
    info["cell_mask"] = cell_mask          # (N, 7) bool，供自检与后续审计使用
    info["after"] = {
        "rows": int(len(cleaned)),
        "target": cfg.target,
        "target_min": float(cleaned[cfg.target].min()),
        "target_max": float(cleaned[cfg.target].max()),
        "target_mean": float(cleaned[cfg.target].mean()),
        "target_std": float(cleaned[cfg.target].std(ddof=0)),
    }
    info["residual_zero_runs"] = dict(
        zip(cfg.features, max_zero_run_lengths(cleaned.to_numpy(dtype="float64"))))
    return cleaned, info


# --------------------------------------------------------------------------- #
# 3. 按时间 6:2:2 划分（不打乱）
# --------------------------------------------------------------------------- #
def split_dataframe(df: pd.DataFrame, cfg: DataConfig) -> Dict[str, pd.DataFrame]:
    """严格按时间先后切分，绝不打乱；返回的 DataFrame 保留原时间索引。"""
    if cfg.split_mode == "official_etth1":
        if len(df) < ETTH1_OFFICIAL_END:
            raise ValueError(
                f"official_etth1 至少需要 {ETTH1_OFFICIAL_END} 行，当前只有 {len(df)} 行")
        train_end = ETTH1_OFFICIAL_TRAIN_ROWS
        val_end = train_end + ETTH1_OFFICIAL_VAL_ROWS
        return {
            "train": df.iloc[:train_end],
            "val": df.iloc[train_end:val_end],
            "test": df.iloc[val_end:ETTH1_OFFICIAL_END],
        }

    n = len(df)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(f"数据量 {n} 太小，无法按 {cfg.train_ratio}:{cfg.val_ratio}:{cfg.test_ratio} 划分")

    return {
        "train": df.iloc[:n_train],
        "val": df.iloc[n_train:n_train + n_val],
        "test": df.iloc[n_train + n_val:],
    }


def _mask_to_ranges(mask: np.ndarray) -> List[Tuple[int, int]]:
    """布尔行掩码 -> 半开区间 [start, end) 列表（连续 True 合为一段）。"""
    ranges: List[Tuple[int, int]] = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            ranges.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return ranges


def split_ranges(index: pd.DatetimeIndex, cfg: DataConfig) -> Dict[str, List[Tuple[int, int]]]:
    """返回每个 split 的**行区间列表**（半开区间，按时间升序）。

    通常只有 season_stratified 模式会产生多个区间；滑窗必须**逐区间**构建，
    否则窗口会跨越不连续的时间段（例如从 3 月直接跳到 12 月）。
    """
    n = len(index)
    if cfg.split_mode == "official_etth1":
        if n < ETTH1_OFFICIAL_END:
            raise ValueError(
                f"official_etth1 至少需要 {ETTH1_OFFICIAL_END} 行，当前只有 {n} 行")
        train_end = ETTH1_OFFICIAL_TRAIN_ROWS
        val_end = train_end + ETTH1_OFFICIAL_VAL_ROWS
        # val/test 向前借 seq_len 行作为已观测上下文；被评分目标仍从各自边界开始。
        return {
            "train": [(0, train_end)],
            "val": [(train_end - cfg.seq_len, val_end)],
            "test": [(val_end - cfg.seq_len, ETTH1_OFFICIAL_END)],
        }

    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    test_range = (n_train + n_val, n)

    if cfg.split_mode == "time_sequential":
        # 主切分：严格按时间先后，单区间
        return {
            "train": [(0, n_train)],
            "val": [(n_train, n_train + n_val)],
            "test": [test_range],
        }

    # ---- season_stratified：只在原 train 时间段内做季节分层 ----
    months = np.asarray(index[:n_train].month)
    seasons = np.array([season_of_month(int(m)) for m in months])
    train_mask = np.zeros(n_train, dtype=bool)
    val_mask = np.zeros(n_train, dtype=bool)

    for name in SEASONS:
        pos = np.flatnonzero(seasons == name)        # 已按时间升序
        if pos.size == 0:
            continue
        if pos.size == 1:
            train_mask[pos] = True                   # 单点季节无法对半，全部给 train
            continue
        k = int(pos.size * cfg.season_train_ratio)   # 前 75% -> train
        k = min(max(k, 1), pos.size - 1)             # 保证两侧都非空
        train_mask[pos[:k]] = True
        val_mask[pos[k:]] = True                     # 后 25% -> val

    return {
        "train": _mask_to_ranges(train_mask),
        "val": _mask_to_ranges(val_mask),
        "test": [test_range],                        # test 永远是原 test 段
    }


def season_composition(index: pd.DatetimeIndex) -> Dict[str, int]:
    """统计某段行的时间在各季节上的分布（用于核对分层是否生效）。"""
    months = np.asarray(index.month)
    seasons = [season_of_month(int(m)) for m in months]
    return {name: int(seasons.count(name)) for name in SEASONS}


# --------------------------------------------------------------------------- #
# 4. 归一化 + 3σ 裁剪（只用 train 拟合）
# --------------------------------------------------------------------------- #
def fit_scaler(train_values: np.ndarray, cfg: DataConfig) -> Dict[str, np.ndarray]:
    """用 train 段拟合 mean/std 与 3σ 裁剪边界（ddof=0，与 sklearn 默认一致）。"""
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std = np.where(std < cfg.eps, 1.0, std)

    if cfg.clip_sigma and cfg.clip_sigma > 0:
        low = mean - cfg.clip_sigma * std
        high = mean + cfg.clip_sigma * std
    else:
        low = np.full_like(mean, -np.inf)
        high = np.full_like(mean, np.inf)

    return {"mean": mean.astype("float64"), "std": std.astype("float64"),
            "low": low.astype("float64"), "high": high.astype("float64")}


def apply_scaler(values: np.ndarray, scaler: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """先裁剪到 train 拟合的 [low, high]，再标准化。返回 (结果, 每列被裁剪的单元格数)。"""
    clipped = np.clip(values, scaler["low"], scaler["high"])
    n_clipped = (clipped != values).sum(axis=0)
    out = (clipped - scaler["mean"]) / scaler["std"]
    return out, n_clipped


def save_scaler(path: str, scaler: Dict[str, np.ndarray], cfg: DataConfig) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(
        path,
        mean=scaler["mean"], std=scaler["std"], low=scaler["low"], high=scaler["high"],
        features=np.array(cfg.features), target=np.array(cfg.target),
        target_idx=np.array(cfg.target_idx),
        seq_len=np.array(cfg.seq_len), pred_len=np.array(cfg.pred_len),
        train_ratio=np.array(cfg.train_ratio), val_ratio=np.array(cfg.val_ratio),
        test_ratio=np.array(cfg.test_ratio), clip_sigma=np.array(cfg.clip_sigma),
        # 清洗策略一并落盘，保证后续反归一化/复现时能追溯数据是怎么洗出来的
        zero_min_cols=np.array(cfg.zero_min_cols),
        zero_run_clean=np.array(int(cfg.zero_run_clean)),
        zero_run_input_hours=np.array(cfg.zero_run_input_hours),
        zero_run_target_hours=np.array(cfg.zero_run_target_hours),
        # 切分模式只作追溯用：scaler 永远用原 train 段拟合，与 split_mode 无关
        split_mode=np.array(cfg.split_mode),
        season_train_ratio=np.array(cfg.season_train_ratio),
        window_stride=np.array(cfg.window_stride),
    )


def load_scaler(path: str) -> Dict[str, np.ndarray]:
    """读取 outputs/scaler.npz，供反归一化使用。"""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def denormalize_ot(y_norm: np.ndarray, scaler: Dict[str, np.ndarray]) -> np.ndarray:
    """把归一化后的目标列还原成原始量纲（摄氏度）。

    目标列下标优先取 scaler 里记录的 `target_idx`，缺省时回落到模块常量 TARGET_IDX。
    """
    idx = int(np.asarray(scaler["target_idx"]).reshape(-1)[0]) if "target_idx" in scaler \
        else TARGET_IDX
    return np.asarray(y_norm) * scaler["std"][idx] + scaler["mean"][idx]


# --------------------------------------------------------------------------- #
# 5. 滑窗与 Dataset
# --------------------------------------------------------------------------- #
def build_windows(data: np.ndarray, seq_len: int, pred_len: int,
                  target_idx: int = TARGET_IDX) -> Tuple[np.ndarray, np.ndarray]:
    """在一个 split 内部切窗（不跨边界）。

    data: (N, C) -> x: (M, seq_len, C) float32, y: (M, pred_len) float32
    """
    n = data.shape[0]
    m = n - seq_len - pred_len + 1
    if m <= 0:
        return (np.empty((0, seq_len, data.shape[1]), dtype="float32"),
                np.empty((0, pred_len), dtype="float32"))

    span = np.arange(seq_len + pred_len)
    idx = np.arange(m)[:, None] + span[None, :]          # (M, L+H)
    windows = data[idx]                                   # (M, L+H, C)
    x = windows[:, :seq_len, :].astype("float32")
    y = windows[:, seq_len:, target_idx].astype("float32")
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def build_windows_ranges(data: np.ndarray, ranges: List[Tuple[int, int]],
                         seq_len: int, pred_len: int,
                         target_idx: int = TARGET_IDX) -> Tuple[np.ndarray, np.ndarray]:
    """对多个**不连续**行区间分别切窗再拼接，窗口不会跨越区间边界。

    区间之间断开处的 (seq_len + pred_len - 1) 个位置被丢弃，这是为了保证
    「过去 24h」和「未来 24h」在真实时间上连续、与主切分口径一致。
    """
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for start, end in ranges:
        x, y = build_windows(data[start:end], seq_len, pred_len, target_idx)
        if len(x):
            xs.append(x)
            ys.append(y)

    if not xs:
        return (np.empty((0, seq_len, data.shape[1]), dtype="float32"),
                np.empty((0, pred_len), dtype="float32"))
    return np.ascontiguousarray(np.concatenate(xs)), np.ascontiguousarray(np.concatenate(ys))


class ETTh1Dataset(Dataset):
    """返回 (x, y)：x=(seq_len, 7) float32，y=(pred_len,) float32（OT）。"""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        if len(x) != len(y):
            raise ValueError(f"x/y 样本数不一致: {len(x)} vs {len(y)}")
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[i], self.y[i]

    def __repr__(self) -> str:
        return (f"ETTh1Dataset(n={len(self)}, x={tuple(self.x.shape[1:])}, "
                f"y={tuple(self.y.shape[1:])})")


# --------------------------------------------------------------------------- #
# 6. 一站式入口
# --------------------------------------------------------------------------- #
def prepare_data(cfg: Optional[DataConfig] = None, save: bool = True,
                 verbose: bool = True) -> Dict:
    """跑完整条管道，返回 datasets / scaler / 审计信息。

    cfg 为 None 时自动读取 configs/base.yaml。

    归一化 mean/std 与裁剪边界一律只用 `split_dataframe(...)["train"]` 拟合。
    time_sequential/season_stratified 使用时间顺序前 60%；official_etth1 使用固定前 12 个月。
    season_stratified 只换用哪些行训练/验证，不重新拟合 scaler。
    """
    cfg = cfg or load_data_config()

    raw = load_raw(cfg)
    clean, clean_info = clean_dataframe(raw, cfg)
    index = clean.index
    values = clean.to_numpy(dtype="float64")

    # 先确定协议的训练段；scaler 永远只在该训练段拟合。
    seq_splits = split_dataframe(clean, cfg)
    scaler = fit_scaler(seq_splits["train"].to_numpy(dtype="float64"), cfg)
    scaled, _ = apply_scaler(values, scaler)
    # 逐单元格的裁剪掩码（与 apply_scaler 内部的判断一致），用于按 split 统计
    clipped_mask = np.clip(values, scaler["low"], scaler["high"]) != values

    ranges = split_ranges(index, cfg)

    datasets: Dict[str, ETTh1Dataset] = {}
    stats: Dict[str, Dict] = {}
    split_frames: Dict[str, List[pd.DataFrame]] = {}
    for name in ("train", "val", "test"):
        rng = ranges[name]
        # 逐区间切窗，窗口不跨区间边界
        x, y = build_windows_ranges(scaled, rng, cfg.seq_len, cfg.pred_len,
                                    target_idx=cfg.target_idx)
        windows_unstrided = int(len(x))
        # 滑窗降采样只作用于 train：val/test 保持完整，保证逐时评估口径不变
        stride = cfg.window_stride if name == "train" else 1
        if stride > 1:
            x = np.ascontiguousarray(x[::stride])
            y = np.ascontiguousarray(y[::stride])
        datasets[name] = ETTh1Dataset(x, y)

        row_mask = np.zeros(len(index), dtype=bool)
        for start, end in rng:
            row_mask[start:end] = True
        sub = clean.loc[row_mask]
        split_frames[name] = [clean.iloc[start:end] for start, end in rng]
        stats[name] = {
            "rows": int(row_mask.sum()),
            "segments": len(rng),
            "ranges": [(int(a), int(b)) for a, b in rng],
            "windows": int(len(x)),
            "windows_unstrided": windows_unstrided,
            "window_stride": int(stride),
            "eff_samples": int(row_mask.sum() // (cfg.seq_len + cfg.pred_len)),
            "start_time": index[rng[0][0]],
            "end_time": index[rng[-1][1] - 1],
            "x_shape": tuple(x.shape),
            "y_shape": tuple(y.shape),
            "cells_clipped": int(clipped_mask[row_mask].sum()),
            "clipped_per_col": dict(zip(cfg.features, clipped_mask[row_mask].sum(axis=0).tolist())),
            "seasons": season_composition(sub.index),
        }

    if save:
        save_scaler(cfg.scaler_path, scaler, cfg)

    bundle = {"datasets": datasets, "scaler": scaler, "config": cfg,
              "clean_info": clean_info, "stats": stats,
              "splits": seq_splits,            # 主切分（时间顺序）的 DataFrame，语义不变
              "split_frames": split_frames,    # 当前 split_mode 下的分段 DataFrame
              "ranges": ranges,
              "clean_frame": clean}

    if verbose:
        print(render_report(bundle))
    return bundle


def render_report(bundle: Dict) -> str:
    ci, st, sc = bundle["clean_info"], bundle["stats"], bundle["scaler"]
    cfg = bundle["config"]
    bar = "=" * 78
    lines: List[str] = []
    lines.append(bar)
    lines.append("ETTh1 数据管道报告")
    lines.append(bar)
    lines.append(f"清洗规则叠加顺序: {cfg.cleaning_summary()}")
    lines.append(f"行数保持 {ci['n_rows']} 不变（小时网格连续，滑窗无时间错位）")
    lines.append("-" * 78)
    lines.append("各规则触发统计")
    lines.append(f"  {'规则':<34s}{'段数':>6s}{'小时':>8s}{'单元格':>9s}")
    for r in ci["rules"]:
        lines.append(f"  {r['name']:<34s}{r['segments']:>6d}{r['hours']:>8d}{r['cells']:>9d}")
    lines.append(f"  {'合并去重后置缺失单元格':<34s}{'':>6s}{'':>8s}{ci['cells_masked']:>9d}")
    lines.append(f"  被触及的行数: {ci['rows_touched']}"
                 f"  |  整行 7 列全 0: {ci['all_zero_rows']}")
    lines.append("-" * 78)
    lines.append("规则 3 逐列明细（阈值 / 段数 / 段长之和）")
    for col, d in ci["zero_run_per_column"].items():
        lines.append(f"  {col:>4s}  阈值>={d['threshold_hours']:>3d}h   "
                     f"段数={d['segments']:>2d}   合计={d['hours']:>4d}h")
    lines.append("-" * 78)
    lines.append("清洗后各列最长连续精确 0（小时，用于确认失效段已消除）")
    lines.append("  " + "  ".join(f"{c}={v}h" for c, v in ci["residual_zero_runs"].items()))
    lines.append("-" * 78)
    mode_note = (f" (season_train_ratio={cfg.season_train_ratio})"
                 if cfg.split_mode == "season_stratified" else "")
    lines.append(f"切分模式: {cfg.split_mode}{mode_note}"
                 f"   |  scaler 只用当前协议的 train 段拟合"
                 f"   |  train window_stride={cfg.window_stride}")
    lines.append(f"{'split':>6s} {'rows':>6s} {'seg':>4s} {'windows':>8s}  {'x_shape':>16s}  "
                 f"{'y_shape':>12s}  {'clipped':>7s}")
    for name in ("train", "val", "test"):
        s = st[name]
        lines.append(f"{name:>6s} {s['rows']:>6d} {s['segments']:>4d} {s['windows']:>8d}  "
                     f"{str(s['x_shape']):>16s}  {str(s['y_shape']):>12s}  "
                     f"{s['cells_clipped']:>7d}")
    lines.append("-" * 78)
    for name in ("train", "val", "test"):
        s = st[name]
        lines.append(f"  {name:>5s}  {s['start_time']} -> {s['end_time']}")
        lines.append(f"         季节分布: {s['seasons']}")
    lines.append("-" * 78)
    lines.append("scaler（只用 train 段拟合）")
    for i, c in enumerate(cfg.features):
        lines.append(f"  {c:>4s}  mean={sc['mean'][i]:>9.4f}  std={sc['std'][i]:>8.4f}"
                     f"  clip=[{sc['low'][i]:>9.4f}, {sc['high'][i]:>9.4f}]")
    lines.append(bar)
    return "\n".join(lines)


def make_dataloaders(cfg: Optional[DataConfig] = None, batch_size: int = 32,
                     num_workers: int = 0, shuffle_train: bool = True,
                     bundle: Optional[Dict] = None) -> Dict[str, DataLoader]:
    """构建 DataLoader。

    shuffle_train 只打乱 **batch 顺序**，不会打乱 split 本身，也不影响 train/val/test
    的时间切分，因此不构成泄漏。val/test 恒为顺序读取，便于逐时对齐评估。
    """
    bundle = bundle or prepare_data(cfg, verbose=False)
    ds = bundle["datasets"]
    loaders = {
        "train": DataLoader(ds["train"], batch_size=batch_size, shuffle=shuffle_train,
                            num_workers=num_workers, drop_last=False),
        "val": DataLoader(ds["val"], batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, drop_last=False),
        "test": DataLoader(ds["test"], batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, drop_last=False),
    }
    return loaders
