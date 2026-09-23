# -*- coding: utf-8 -*-
"""`utils/dataset.py` 单元测试。

三类内容：
1. 清洗三条规则在**合成数据**上的行为（可精确断言，不依赖真实数据）；
2. 切分的不变量（时间顺序 / 分层模式和主切分的关系 / 滑窗不跨区间）；
3. scaler 只用 train 拟合、与窗口长度无关（这是 96->96 复用同一套均值方差的依据）。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from utils.dataset import (
    FEATURES, TARGET_IDX, DataConfig, apply_scaler, build_anomaly_mask, build_windows,
    build_windows_ranges, clean_dataframe, find_identical_runs, find_long_zero_runs,
    fit_scaler, load_data_config, load_raw, load_scaler, max_zero_run_lengths, prepare_data,
    season_of_month, split_dataframe, split_ranges,
)


def _cfg(**kw) -> DataConfig:
    """合成数据用的配置：默认关掉规则 1 与规则 3，只留想测的那条规则。"""
    base = dict(data_path="data/ETTh1.csv", dedup_policy="keep",
                zero_run_input_hours=10 ** 6, zero_run_target_hours=10 ** 6)
    base.update(kw)
    return DataConfig(**base)


def _frame(values: np.ndarray, start: str = "2016-07-01",
           columns=None) -> pd.DataFrame:
    return pd.DataFrame(values, columns=list(columns) if columns is not None else FEATURES,
                        index=pd.date_range(start, periods=len(values), freq="h"))


# --------------------------------------------------------------------------- #
# 规则 1：重复块
# --------------------------------------------------------------------------- #
def test_find_identical_runs_marks_whole_block_including_first_row():
    v = np.array([[1, 2], [1, 2], [1, 2], [3, 4], [5, 6], [5, 6]], dtype="float64")
    assert find_identical_runs(v).tolist() == [True, True, True, False, True, True]


def test_find_identical_runs_ignores_single_rows():
    v = np.arange(20, dtype="float64").reshape(10, 2)
    assert not find_identical_runs(v).any()


def test_dedup_policy_keep_disables_rule1():
    v = np.ones((4, 7))
    mask_mask, info_mask = build_anomaly_mask(v, _cfg(dedup_policy="mask"))
    mask_keep, _ = build_anomaly_mask(v, _cfg(dedup_policy="keep"))
    assert mask_mask.all()
    assert info_mask["dup_segments"] == 1 and info_mask["dup_rows"] == 4
    assert not mask_keep.any()


# --------------------------------------------------------------------------- #
# 规则 2：行级零值
# --------------------------------------------------------------------------- #
def test_row_level_zero_rule_only_touches_zero_cells():
    v = np.ones((4, 7))
    v[2, :5] = 0.0                      # 一行 5 个 0 -> 触发
    mask, info = build_anomaly_mask(v, _cfg(zero_min_cols=5))
    assert mask[2].sum() == 5
    assert mask.sum() == 5
    assert info["zero_rows"] == 1 and info["zero_cells"] == 5


def test_single_column_zeros_are_not_killed_by_row_rule():
    v = np.ones((8, 7))
    v[:, 3] = 0.0                       # 每行只有 1 个 0 -> 不触发行级规则
    mask, info = build_anomaly_mask(v, _cfg(zero_min_cols=5))
    assert not mask.any()
    assert info["zero_rows"] == 0


# --------------------------------------------------------------------------- #
# 规则 3：单列长 0 段
# --------------------------------------------------------------------------- #
def test_find_long_zero_runs_threshold_and_disable():
    v = np.zeros((6, 2))
    v[:, 0] = [1, 0, 0, 0, 1, 1]
    mask, runs = find_long_zero_runs(v, [3, 10])
    assert mask[:, 0].tolist() == [False, True, True, True, False, False]
    assert mask[:, 1].sum() == 0                     # 6 < 阈值 10
    assert runs[0][0]["length"] == 3 and runs[0][0]["start_idx"] == 1

    mask_off, runs_off = find_long_zero_runs(v, [0, 0])
    assert not mask_off.any() and runs_off == [[], []]


def test_find_long_zero_runs_rejects_wrong_threshold_count():
    with pytest.raises(ValueError):
        find_long_zero_runs(np.zeros((3, 7)), [1, 2, 3])


def test_target_column_uses_its_own_threshold():
    v = np.ones((20, 7))
    v[5:12, TARGET_IDX] = 0.0             # OT 连续 7 小时
    v[0:14, 0] = 0.0                      # 输入列连续 14 小时
    cfg = _cfg(zero_run_input_hours=12, zero_run_target_hours=6)
    mask, info = build_anomaly_mask(v, cfg)
    assert info["zero_run_per_column"]["OT"]["segments"] == 1
    assert info["zero_run_per_column"]["OT"]["threshold_hours"] == 6
    assert info["zero_run_per_column"]["HUFL"]["threshold_hours"] == 12
    assert mask[5:12, TARGET_IDX].all() and mask[0:14, 0].all()


# --------------------------------------------------------------------------- #
# 统一插值
# --------------------------------------------------------------------------- #
def test_clean_dataframe_interpolates_and_keeps_row_count():
    v = np.ones((6, 7))
    v[2:4] = 0.0                          # 两行全 0 -> 规则 2
    df = _frame(v)
    clean, info = clean_dataframe(df, _cfg())
    assert len(clean) == 6                # 行数不变（小时网格不被破坏）
    assert not clean.isna().any().any()
    assert info["interpolated_cells"] == 14
    assert np.allclose(clean.to_numpy(), 1.0)


def test_clean_dataframe_removes_long_zero_runs():
    v = np.ones((40, 7))
    v[10:30, 2] = 0.0
    clean, info = clean_dataframe(_frame(v), _cfg(zero_run_input_hours=12))
    assert info["zero_run_per_column"]["MUFL"]["hours"] == 20
    assert max_zero_run_lengths(clean.to_numpy())[2] == 0


def test_max_zero_run_lengths():
    v = np.ones((10, 2))
    v[2:5, 0] = 0.0                       # 3 连 0
    v[5:6, 1] = 0.0                       # 1 个 0
    assert max_zero_run_lengths(v) == [3, 1]


# --------------------------------------------------------------------------- #
# 切分
# --------------------------------------------------------------------------- #
def test_split_dataframe_is_time_ordered():
    idx = pd.date_range("2016-07-01", periods=100, freq="h")
    df = _frame(np.arange(100 * 7, dtype="float64").reshape(100, 7))
    cfg = _cfg()
    parts = split_dataframe(df, cfg)
    assert (len(parts["train"]), len(parts["val"]), len(parts["test"])) == (60, 20, 20)
    assert parts["train"].index.max() < parts["val"].index.min() < parts["test"].index.min()


def test_split_ranges_time_sequential_is_contiguous():
    idx = pd.date_range("2016-07-01", periods=100, freq="h")
    r = split_ranges(idx, _cfg())
    assert r["train"] == [(0, 60)] and r["val"] == [(60, 80)] and r["test"] == [(80, 100)]


def test_official_etth1_split_uses_fixed_benchmark_boundaries():
    n = 17_420
    df = _frame(np.arange(n * 7, dtype="float64").reshape(n, 7))
    cfg = _cfg(split_mode="official_etth1", seq_len=96, pred_len=96)

    parts = split_dataframe(df, cfg)
    ranges = split_ranges(df.index, cfg)

    assert tuple(len(parts[name]) for name in ("train", "val", "test")) == (8640, 2880, 2880)
    assert ranges == {
        "train": [(0, 8640)],
        "val": [(8640 - 96, 8640 + 2880)],
        "test": [(8640 + 2880 - 96, 8640 + 2880 + 2880)],
    }
    # context 可以来自前一 split，但首个被评分的目标必须严格位于当前 split 内。
    assert ranges["val"][0][0] + cfg.seq_len == 8640
    assert ranges["test"][0][0] + cfg.seq_len == 8640 + 2880


def test_official_etth1_split_rejects_truncated_data():
    idx = pd.date_range("2016-07-01", periods=14_399, freq="h")
    with pytest.raises(ValueError, match="14400"):
        split_ranges(idx, _cfg(split_mode="official_etth1", seq_len=96, pred_len=96))


def test_official_config_preserves_raw_values_and_expected_window_counts(project_root):
    cfg = load_data_config(os.path.join(project_root, "configs", "official_etth1_96.yaml"))
    raw = load_raw(cfg)
    bundle = prepare_data(cfg, save=False, verbose=False)

    assert np.array_equal(bundle["clean_frame"].to_numpy(), raw.to_numpy())
    assert bundle["clean_info"]["cells_masked"] == 0
    assert tuple(len(bundle["datasets"][name]) for name in ("train", "val", "test")) == \
        (8449, 2785, 2785)
    for name in ("train", "val", "test"):
        assert bundle["stats"][name]["cells_clipped"] == 0


def test_split_ranges_season_stratified_covers_only_original_train():
    idx = pd.date_range("2016-07-01", periods=100, freq="h")
    r = split_ranges(idx, _cfg(split_mode="season_stratified"))
    assert r["test"] == [(80, 100)]                      # test 永远是原 test 段
    tr, va = set(), set()
    for a, b in r["train"]:
        tr |= set(range(a, b))
    for a, b in r["val"]:
        va |= set(range(a, b))
    assert tr and va
    assert not (tr & va)                                 # 不重叠
    assert max(tr | va) < 60                             # 绝不碰原 val / test 段


def test_season_of_month():
    assert season_of_month(1) == "winter" and season_of_month(4) == "spring"
    assert season_of_month(7) == "summer" and season_of_month(10) == "autumn"
    with pytest.raises(ValueError):
        season_of_month(13)


# --------------------------------------------------------------------------- #
# 滑窗
# --------------------------------------------------------------------------- #
def test_build_windows_shapes_and_alignment():
    data = np.arange(50 * 3, dtype="float64").reshape(50, 3)
    x, y = build_windows(data, seq_len=5, pred_len=4, target_idx=2)
    assert x.shape == (42, 5, 3) and y.shape == (42, 4)
    assert np.array_equal(x[0], data[:5])
    assert np.array_equal(y[0], data[5:9, 2])
    assert np.array_equal(x[-1], data[41:46])


def test_build_windows_returns_empty_when_too_short():
    data = np.zeros((8, 3))
    x, y = build_windows(data, seq_len=5, pred_len=4, target_idx=2)
    assert len(x) == 0 and len(y) == 0


def test_build_windows_ranges_never_crosses_gap():
    data = np.tile(np.arange(50, dtype="float64")[:, None], (1, 3))   # 每列都是 0..49
    x, y = build_windows_ranges(data, [(0, 20), (30, 50)], 5, 4, target_idx=2)
    assert len(x) == 12 + 12                             # 每段 20-5-4+1
    for i in range(len(x)):
        assert np.allclose(np.diff(x[i][:, 2]), 1.0)     # 段内时间连续，没有跳变
        assert np.allclose(np.diff(y[i]), 1.0)
    # 第二段的第一个窗口必须从第 30 行开始（没有跨过 (20,30) 的缺口）
    assert x[12, 0, 2] == 30.0


# --------------------------------------------------------------------------- #
# 归一化：只用 train、与窗口长度无关
# --------------------------------------------------------------------------- #
def test_fit_scaler_uses_only_the_given_array():
    train = np.zeros((10, 7))
    train[:, TARGET_IDX] = np.arange(10, dtype="float64")
    scaler = fit_scaler(train, _cfg())
    assert scaler["mean"][TARGET_IDX] == pytest.approx(4.5)
    assert scaler["std"][TARGET_IDX] == pytest.approx(np.std(np.arange(10.0)))


def test_fit_scaler_is_independent_of_window_length():
    """96->96 设定必须复用与 24->24 完全相同的均值/方差（scaler 只认 train 段）。"""
    train = np.random.default_rng(0).normal(size=(500, 7))
    a = fit_scaler(train, _cfg(seq_len=24, pred_len=24))
    b = fit_scaler(train, _cfg(seq_len=96, pred_len=96))
    for key in ("mean", "std", "low", "high"):
        assert np.array_equal(a[key], b[key])


def test_apply_scaler_clips_to_train_bounds():
    train = np.zeros((10, 7))
    train[:, TARGET_IDX] = np.arange(10, dtype="float64")
    scaler = fit_scaler(train, _cfg(clip_sigma=1.0))
    out, n_clipped = apply_scaler(np.full((4, 7), 1e6), scaler)
    assert n_clipped[TARGET_IDX] == 4
    assert out[:, TARGET_IDX].max() == pytest.approx(1.0)


def test_clip_sigma_zero_disables_clipping():
    train = np.random.default_rng(0).normal(size=(50, 7))
    scaler = fit_scaler(train, _cfg(clip_sigma=0.0))
    assert np.isneginf(scaler["low"]).all() and np.isposinf(scaler["high"]).all()


# --------------------------------------------------------------------------- #
# 配置校验
# --------------------------------------------------------------------------- #
def test_data_config_validation():
    with pytest.raises(ValueError):
        _cfg(train_ratio=0.7)                            # 比例和不为 1
    with pytest.raises(ValueError):
        _cfg(split_mode="nope")
    with pytest.raises(ValueError):
        _cfg(window_stride=0)
    with pytest.raises(ValueError):
        _cfg(season_train_ratio=1.5)
    with pytest.raises(ValueError):
        _cfg(zero_run_input_hours=0)                     # 开启规则 3 时阈值必须为正


def test_data_config_resolves_relative_paths_to_project_root():
    cfg = _cfg()
    assert os.path.isabs(cfg.data_path)
    assert os.path.normpath(cfg.data_path).endswith(os.path.join("data", "ETTh1.csv"))


# --------------------------------------------------------------------------- #
# 列名可配置：换一份数据不必改代码
# --------------------------------------------------------------------------- #
def test_default_features_are_etth1_columns():
    cfg = _cfg()
    assert cfg.features == FEATURES
    assert cfg.target == "OT" and cfg.target_idx == TARGET_IDX


def test_custom_feature_cols_and_target_col():
    cfg = _cfg(feature_cols=["a", "b", "c"], target_col="b")
    assert cfg.features == ("a", "b", "c")
    assert cfg.target == "b" and cfg.target_idx == 1


def test_target_col_must_be_one_of_the_features():
    with pytest.raises(ValueError):
        _cfg(feature_cols=["a", "b"], target_col="z")
    with pytest.raises(ValueError):
        _cfg(feature_cols=[])


def test_save_scaler_records_features_and_target(work_tmp):
    from utils.dataset import save_scaler

    cfg = _cfg(feature_cols=["a", "b", "c"], target_col="b")
    path = os.path.join(work_tmp, "scaler_custom.npz")
    save_scaler(path, fit_scaler(np.random.default_rng(0).normal(size=(20, 3)), cfg), cfg)
    z = load_scaler(path)
    assert list(z["features"]) == ["a", "b", "c"]
    assert str(z["target"]) == "b" and int(z["target_idx"]) == 1


def test_custom_target_column_drives_windows_and_clean_info():
    values = np.random.default_rng(0).normal(size=(40, 3))
    values[10:20, 1] = 0.0                               # 自定义目标列出现长 0 段
    cfg = _cfg(feature_cols=["a", "b", "c"], target_col="b",
               zero_run_input_hours=12, zero_run_target_hours=6)
    clean, info = clean_dataframe(_frame(values, columns=["a", "b", "c"]), cfg)
    assert info["zero_run_per_column"]["b"]["threshold_hours"] == 6     # 目标列用 6h 阈值
    assert info["zero_run_per_column"]["a"]["threshold_hours"] == 12
    assert set(info["residual_zero_runs"]) == {"a", "b", "c"}
    assert info["after"]["target"] == "b"
    # 切窗时取的是第 2 列
    x, y = build_windows(clean.to_numpy(dtype="float64"), 5, 4, target_idx=cfg.target_idx)
    assert np.allclose(y[0], clean.to_numpy(dtype="float64")[5:9, 1])


# --------------------------------------------------------------------------- #
# 真实数据管线的不变量
# --------------------------------------------------------------------------- #
def test_bundle_window_counts_match_ranges(base_bundle):
    cfg = base_bundle["config"]
    for name in ("train", "val", "test"):
        st = base_bundle["stats"][name]
        per_range = [max(0, b - a - cfg.seq_len - cfg.pred_len + 1) for a, b in st["ranges"]]
        expected = sum(per_range)
        if name == "train":
            expected = len(range(0, expected, st["window_stride"]))
        assert st["windows"] == expected, f"{name}: {st['windows']} != {expected}"
        assert st["x_shape"][0] == st["windows"]


def test_bundle_shapes_match_config(base_bundle):
    cfg = base_bundle["config"]
    ds = base_bundle["datasets"]
    for name in ("train", "val", "test"):
        x, y = ds[name].x, ds[name].y
        assert tuple(x.shape[1:]) == (cfg.seq_len, len(FEATURES))
        assert tuple(y.shape[1:]) == (cfg.pred_len,)
        assert x.dtype.is_floating_point and y.dtype.is_floating_point


def test_bundle_cleaning_does_not_change_row_count(base_bundle):
    assert base_bundle["clean_info"]["n_rows"] == len(base_bundle["clean_frame"])
    assert not base_bundle["clean_frame"].isna().any().any()


def test_disk_scaler_matches_refit(base_bundle):
    """磁盘上的 scaler 必须等于按同一份配置重新拟合的结果（否则评估口径就漂了）。"""
    path = base_bundle["config"].scaler_path
    if not os.path.isfile(path):
        pytest.skip(f"{path} 不存在")
    disk = load_scaler(path)
    for key in ("mean", "std", "low", "high"):
        assert np.allclose(disk[key], base_bundle["scaler"][key], rtol=0, atol=1e-12), key


def test_lstf96_reuses_identical_statistics(lstf96_bundle, project_root):
    """96->96 与 24->24 的归一化统计量必须逐元素相同，否则两个设定不可比。"""
    scaler24 = os.path.join(project_root, "outputs", "scaler.npz")
    scaler96 = os.path.join(project_root, "outputs", "scaler96.npz")
    if not (os.path.isfile(scaler24) and os.path.isfile(scaler96)):
        pytest.skip("outputs/scaler.npz 或 outputs/scaler96.npz 不存在")
    a, b = load_scaler(scaler24), load_scaler(scaler96)
    for key in ("mean", "std", "low", "high"):
        assert np.array_equal(a[key], b[key]), key
    assert int(a["seq_len"]) == 24 and int(b["seq_len"]) == 96
