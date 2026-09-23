# -*- coding: utf-8 -*-
"""`utils/metrics.py` 单元测试：horizon 自适应、指标数值、输入校验、反归一化。"""

from __future__ import annotations

import os

import numpy as np
import pytest

from utils.metrics import (
    HORIZONS, HORIZON_SETS, compute_metrics, denorm_ot, format_metrics,
    horizons_for, metric_keys, target_index,
)


# --------------------------------------------------------------------------- #
# horizon 集合
# --------------------------------------------------------------------------- #
def test_horizons_default_is_24_step_set():
    assert HORIZONS == (1, 6, 12, 24)
    assert HORIZON_SETS[24] == HORIZONS


def test_horizons_for_known_settings():
    assert horizons_for(24) == (1, 6, 12, 24)
    assert horizons_for(96) == (1, 24, 48, 96)
    assert horizons_for(336) == (1, 48, 168, 336)
    assert horizons_for(720) == (1, 96, 336, 720)


def test_horizons_for_falls_back_to_subset():
    assert horizons_for(12) == (1, 6, 12)
    assert horizons_for(1) == (1,)
    for n in (2, 13, 50, 100, 200, 1000):
        picked = horizons_for(n)
        assert picked and all(h <= n for h in picked)
        assert list(picked) == sorted(set(picked))


# --------------------------------------------------------------------------- #
# compute_metrics
# --------------------------------------------------------------------------- #
def test_perfect_prediction_is_exactly_zero():
    rng = np.random.default_rng(0)
    gts = rng.normal(size=(32, 24))
    m = compute_metrics(gts, gts)
    assert m["overall"]["MAE"] == 0.0
    assert m["overall"]["RMSE"] == 0.0
    for key in ("h1", "h6", "h12", "h24"):
        assert m[key]["MAE"] == 0.0


def test_hand_computed_values():
    preds = np.array([[0.0, 1.0, 2.0, 3.0]])
    gts = np.array([[0.0, 3.0, 0.0, 3.0]])
    err = np.array([0.0, -2.0, 2.0, 0.0])
    m = compute_metrics(preds, gts, horizons=(1, 2, 3, 4))
    assert m["overall"]["MAE"] == pytest.approx(np.mean(np.abs(err)))
    assert m["overall"]["RMSE"] == pytest.approx(np.sqrt(np.mean(err ** 2)))
    assert m["h2"]["MAE"] == pytest.approx(2.0)
    assert m["h3"]["MAE"] == pytest.approx(2.0)      # 显式请求了 h3
    assert "h6" not in m                             # 未请求就不出现


def test_metric_keys_sorted_numerically_not_lexically():
    m = compute_metrics(np.zeros((2, 12)), np.zeros((2, 12)))
    assert metric_keys(m) == ["h1", "h6", "h12"]
    assert sorted(metric_keys(m)) == ["h1", "h12", "h6"]      # 字典序会把 h12 排到 h6 前面


def test_96_step_default_horizons():
    gts = np.zeros((5, 96))
    preds = np.ones((5, 96))
    m = compute_metrics(preds, gts)
    assert metric_keys(m) == ["h1", "h24", "h48", "h96"]
    for key in metric_keys(m):
        assert m[key]["MAE"] == pytest.approx(1.0)


def test_horizon_beyond_pred_len_is_ignored():
    m = compute_metrics(np.zeros((2, 4)), np.zeros((2, 4)), horizons=(1, 6))
    assert metric_keys(m) == ["h1"]


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        compute_metrics(np.zeros((2, 3)), np.zeros((2, 4)))       # 形状不一致
    with pytest.raises(ValueError):
        compute_metrics(np.zeros(3), np.zeros(3))                 # 不是二维
    bad = np.zeros((2, 3))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        compute_metrics(bad, np.zeros((2, 3)))                    # NaN


# --------------------------------------------------------------------------- #
# 反归一化 / 排版
# --------------------------------------------------------------------------- #
def test_denorm_ot_is_inverse_of_normalization():
    mean = np.array([0.0] * 6 + [17.25])
    std = np.array([1.0] * 6 + [8.5])
    scaler = {"mean": mean, "std": std}
    z = np.array([[-1.0, 0.0, 2.0]])
    assert np.allclose(denorm_ot(z, scaler), z * 8.5 + 17.25)


def test_denorm_ot_accepts_npz_path(work_tmp):
    path = os.path.join(work_tmp, "scaler_denorm.npz")
    np.savez(path, mean=np.zeros(7), std=np.full(7, 2.0))
    assert np.allclose(denorm_ot(np.array([1.0]), path), 2.0)


def test_denorm_ot_reads_target_idx_from_scaler():
    scaler = {"mean": np.array([1.0, 2.0, 3.0]), "std": np.ones(3), "target_idx": np.array(1)}
    assert denorm_ot(np.array([10.0]), scaler) == pytest.approx(12.0)
    # 没有 target_idx 时回落到模块常量 6
    fallback = {"mean": np.zeros(7), "std": np.ones(7)}
    assert target_index(fallback) == 6


def test_denorm_ot_rejects_bad_scaler():
    with pytest.raises(TypeError):
        denorm_ot(np.zeros(3), {"mean": np.zeros(7)})


def test_format_metrics_lists_every_key():
    m = compute_metrics(np.zeros((4, 24)), np.zeros((4, 24)))
    text = format_metrics(m, title="标题")
    assert text.splitlines()[0] == "标题"
    for label in ("overall", "1 步", "6 步", "12 步", "24 步"):
        assert label in text
