# -*- coding: utf-8 -*-
"""`eval/common.py` 的单元测试。

这个模块是**所有结论的出口**（聚合、2σ 判定、表格渲染），却最容易在重构里被悄悄改坏，
所以单独补一组测试。只测纯函数，不加载 ckpt、不跑数据管道。
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from eval.common import (
    SEEDS, agg, ckpt_predictions, collect_seed_metrics, compare, compare_pair, compare_pair_ci,
    compare_runs_to_fixed_ci, fmt, load_test_set, metrics_from_json, persistence_preds,
    require_complete_seeds,
    rel_to_project, render_table, resolve_device, table_columns, verdict_text, write_json,
)
from utils.dataset import DataConfig, load_scaler, prepare_data
from utils.metrics import compute_metrics


def _metrics(mae: float) -> dict:
    """构造一个 overall MAE = mae 的最小 metrics 结构。"""
    preds = np.full((4, 24), mae)
    gts = np.zeros((4, 24))
    return compute_metrics(preds, gts)


# --------------------------------------------------------------------------- #
# 聚合
# --------------------------------------------------------------------------- #
def test_seed_set_is_the_documented_one():
    assert SEEDS == (42, 0, 1, 2, 9999)


def test_require_complete_seeds_rejects_partial_or_unexpected_sets():
    assert require_complete_seeds([42, 0, 1, 2, 9999], label="model") == SEEDS
    with pytest.raises(ValueError, match="缺少.*9999"):
        require_complete_seeds([42, 0, 1, 2], label="model")
    with pytest.raises(ValueError, match="多出.*7"):
        require_complete_seeds([42, 0, 1, 2, 9999, 7], label="model")


def test_agg_takes_mean_and_sample_std():
    runs = [_metrics(v) for v in (1.0, 2.0, 3.0)]
    mean, std = agg(runs, ["overall", "MAE"])
    assert mean == pytest.approx(2.0)
    assert std == pytest.approx(1.0)          # 样本标准差（ddof=1）


def test_agg_single_run_has_zero_std():
    mean, std = agg([_metrics(1.5)], ["overall", "MAE"])
    assert mean == pytest.approx(1.5) and std == 0.0


def test_agg_reads_nested_horizon_keys():
    runs = [_metrics(v) for v in (1.0, 3.0)]
    assert agg(runs, ["h1", "MAE"])[0] == pytest.approx(2.0)
    assert agg(runs, ["h24", "RMSE"])[0] == pytest.approx(2.0)


def test_fmt_format():
    assert fmt(1.28881, 0.00196) == "1.2888 ± 0.0020"
    assert fmt(1.5, 0.0, nd=2) == "1.50 ± 0.00"


# --------------------------------------------------------------------------- #
# 2σ 判定：整个项目的结论闸门
# --------------------------------------------------------------------------- #
def test_verdict_text_threshold_is_inclusive():
    assert "超过噪声" in verdict_text(0.20, 0.20)        # |delta| == 2σ 判为可信
    assert "超过噪声" in verdict_text(-0.20, 0.20)
    assert "噪声范围" in verdict_text(0.19, 0.20)


def test_compare_against_a_deterministic_baseline():
    line = compare("DLinear(5 seed)", 1.2888, 0.0020, "persistence", 1.4439)
    assert "delta = -0.1551" in line
    assert "2σ = 0.0040" in line                       # 2 × max(0.0020, 0.0)
    assert "超过噪声" in line


def test_compare_uses_the_larger_of_the_two_stds():
    line = compare("A", 1.0, 0.20, "B", 1.5, 0.10)
    assert "2σ = 0.4000" in line                       # 2 × 0.20，不是 2 × 0.10
    assert "超过噪声" in line
    # 同样两个 mean，把 σ 换大一点就翻成「不显著」—— 证明判据确实卡在 σ 上
    weak = compare("A", 1.0, 0.40, "B", 1.5, 0.10)
    assert "2σ = 0.8000" in weak and "噪声范围" in weak


def test_compare_marks_deterministic_algorithm_as_not_participating():
    line = compare("ARIMA", 1.4491, 0.0, "persistence", 1.4439, deterministic=True)
    assert "不参与 2σ 判定" in line


def test_compare_pair_symmetric_threshold():
    a = [_metrics(v) for v in (1.0, 1.1, 1.2)]
    b = [_metrics(v) for v in (1.5, 1.6, 1.7)]
    forward = compare_pair("A", a, "B", b)
    assert "2×max(σ_A, σ_B)" in forward
    assert "+0.5000" in forward or "-0.5000" in forward


def test_paired_seed_ci_uses_within_seed_differences():
    a = [_metrics(v) for v in (1.00, 1.10, 1.20, 1.30, 1.40)]
    b = [_metrics(v) for v in (1.50, 1.60, 1.70, 1.80, 1.90)]
    line = compare_pair_ci("A", a, "B", b)
    assert "paired 95% CI" in line
    assert "-0.5000" in line
    assert "excludes 0" in line


def test_fixed_baseline_ci_requires_multiple_complete_runs():
    runs = [_metrics(v) for v in (1.0, 1.1, 1.2, 1.3, 1.4)]
    line = compare_runs_to_fixed_ci("model", runs, "persistence", 2.0)
    assert "paired 95% CI" in line and "excludes 0" in line
    with pytest.raises(ValueError, match="2"):
        compare_runs_to_fixed_ci("model", runs[:1], "persistence", 2.0)


# --------------------------------------------------------------------------- #
# 表格
# --------------------------------------------------------------------------- #
def test_table_columns_follows_horizons_for():
    assert table_columns(_metrics(1.0)) == ["h1", "h6", "h12", "h24"]
    assert table_columns(compute_metrics(np.zeros((2, 96)), np.zeros((2, 96)))) == \
        ["h1", "h24", "h48", "h96"]


def test_render_table_contains_every_number():
    metrics = _metrics(1.25)
    columns = table_columns(metrics)
    rows = [{"name": "dummy", "MAE": (1.25, 0.01), "RMSE": (1.75, 0.02),
             "h": {c: (metrics[c]["MAE"], 0.0) for c in columns}, "note": "(确定性)"}]
    text = render_table(rows, columns)
    assert "dummy" in text and "1.2500 ± 0.0100" in text and "(确定性)" in text
    for c in columns:
        assert c in text


# --------------------------------------------------------------------------- #
# 平凡基线 / IO
# --------------------------------------------------------------------------- #
def test_persistence_repeats_last_observation():
    out = persistence_preds(np.array([1.0, 2.0, 3.0]), 4)
    assert out.shape == (3, 4)
    assert np.array_equal(out, np.array([[1.0] * 4, [2.0] * 4, [3.0] * 4]))


def test_metrics_from_json_reads_the_metrics_key(work_tmp):
    path = os.path.join(work_tmp, "m.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"metrics": {"overall": {"MAE": 1.0, "RMSE": 2.0}}}, fh)
    assert metrics_from_json(path)["overall"]["MAE"] == 1.0


def test_collect_seed_metrics_rejects_wrong_window_protocol(work_tmp):
    pattern = os.path.join(work_tmp, "protocol_seed{s}.json")
    with open(pattern.format(s=42), "w", encoding="utf-8") as fh:
        json.dump({"seq_len": 24, "pred_len": 96, "metrics": _metrics(1.0)}, fh)
    with pytest.raises(ValueError, match="pred_len"):
        collect_seed_metrics(pattern, seeds=[42], expected_seq_len=24, expected_pred_len=24)


def test_write_json_and_rel_to_project(work_tmp):
    path = os.path.join(work_tmp, "out.json")
    write_json(path, {"config": "configs/base.yaml"})
    with open(path, "r", encoding="utf-8") as fh:
        assert json.load(fh)["config"] == "configs/base.yaml"

    assert rel_to_project("configs/base.yaml") == "configs/base.yaml"
    assert rel_to_project(os.path.join(work_tmp, "x.json")) == "outputs/_runs/_pytest_tmp/x.json"
    # 仓库外的绝对路径原样返回（不硬凑成 ../..）
    outside = os.path.abspath(os.path.join(os.sep, "definitely", "outside", "x.json"))
    assert rel_to_project(outside) == outside


def test_resolve_device():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("").type in ("cpu", "cuda")     # 空串等同 auto
    assert resolve_device("auto").type in ("cpu", "cuda")


def test_load_test_set_uses_configured_target_column(work_tmp):
    """统一评估入口不能把自定义数据的目标列写死为 ETTh1 的第 7 列。"""
    import pandas as pd

    n = 30
    csv_path = os.path.join(work_tmp, "custom_eval.csv")
    scaler_path = os.path.join(work_tmp, "custom_eval_scaler.npz")
    pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n, freq="h"),
        "a": np.linspace(1.0, 2.0, n),
        "target": np.arange(n, dtype="float64"),
        "c": np.linspace(100.0, 200.0, n),
    }).to_csv(csv_path, index=False)
    cfg = DataConfig(
        data_path=csv_path,
        scaler_path=scaler_path,
        feature_cols=("a", "target", "c"),
        target_col="target",
        seq_len=3,
        pred_len=2,
        dedup_policy="keep",
        zero_run_clean=False,
        clip_sigma=0.0,
    )
    prepare_data(cfg, save=True, verbose=False)

    test = load_test_set(cfg, load_scaler(scaler_path))

    assert test["gts"].shape == (2, 2)
    assert np.allclose(test["gts"], np.array([[27.0, 28.0], [28.0, 29.0]]), atol=1e-6)


def test_ckpt_predictions_batches_inference_without_changing_values(work_tmp):
    import torch
    from models.transformer import TransformerForecaster

    model = TransformerForecaster(
        input_dim=3, d_model=4, nhead=1, num_layers=1, dim_ff=8,
        dropout=0.0, input_len=3, pred_len=2, target_idx=1).eval()
    ckpt_path = os.path.join(work_tmp, "tiny_model.pt")
    torch.save({"model_config": model.config(), "model_state_dict": model.state_dict()}, ckpt_path)
    x = np.random.default_rng(0).normal(size=(5, 3, 3)).astype("float32")
    scaler = {"mean": np.array([0.0, 10.0, 0.0]),
              "std": np.array([1.0, 2.0, 1.0]), "target_idx": np.array(1)}
    with torch.no_grad():
        expected = model(torch.as_tensor(x)).numpy() * 2.0 + 10.0

    actual = ckpt_predictions(ckpt_path, x, scaler, batch_size=2)

    assert np.allclose(actual, expected, atol=1e-6)
