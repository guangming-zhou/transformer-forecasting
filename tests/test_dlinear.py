# -*- coding: utf-8 -*-
"""`eval/baseline_dlinear.py` 的单元测试：分解重构、目标通道选择、参数量、pad 行为。"""

from __future__ import annotations

import torch
import pytest

from eval.baseline_dlinear import DLinearForecaster, MovingAvg, SeriesDecomp


def test_moving_average_keeps_length_and_smooths():
    x = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1)
    avg = MovingAvg(3)(x)
    assert avg.shape == x.shape
    # 两端用边界值复制补齐（与官方实现一致），所以边界窗口不是简单平均
    assert avg[0, 5, 0] == pytest.approx(5.0)         # 线性序列内部：平均还是自身
    assert avg[0, 0, 0] == pytest.approx(1.0 / 3.0)   # 窗口 (0, 0, 1)
    assert avg[0, -1, 0] == pytest.approx(26.0 / 3.0)  # 窗口 (8, 9, 9)


def test_series_decomposition_reconstructs_input():
    x = torch.randn(4, 24, 7)
    seasonal, trend = SeriesDecomp(25)(x)
    assert seasonal.shape == x.shape == trend.shape
    assert torch.allclose(seasonal + trend, x, atol=1e-6)


def test_forward_shape_and_multi_step_direct_output():
    model = DLinearForecaster(seq_len=24, pred_len=24).eval()
    with torch.no_grad():
        assert tuple(model(torch.randn(5, 24, 7)).shape) == (5, 24)
    model96 = DLinearForecaster(seq_len=96, pred_len=96).eval()
    with torch.no_grad():
        assert tuple(model96(torch.randn(2, 96, 7)).shape) == (2, 96)


def test_output_is_taken_from_the_target_channel():
    model = DLinearForecaster(seq_len=6, pred_len=3, kernel_size=3, target_idx=1).eval()
    with torch.no_grad():
        model.linear_seasonal.weight.copy_(torch.eye(3, 6))
        model.linear_seasonal.bias.zero_()
        model.linear_trend.weight.zero_()
        model.linear_trend.bias.zero_()
        x = torch.randn(2, 6, 4)
        out = model(x)
        seasonal, _ = model.decomp(x)
    expected = seasonal[:, :, 1] @ torch.eye(3, 6).T            # 只取第 1 号通道
    assert torch.allclose(out, expected, atol=1e-6)


def test_parameter_count_is_channel_independent():
    model = DLinearForecaster(seq_len=24, pred_len=24, kernel_size=25)
    n = sum(p.numel() for p in model.parameters())
    assert n == 2 * (24 * 24 + 24) == 1200
    # 通道数不进入参数量：通道独立 + 权重共享
    any_channel = DLinearForecaster(seq_len=24, pred_len=24, target_idx=0).eval()
    with torch.no_grad():
        assert any_channel(torch.randn(2, 24, 3)).shape == (2, 24)
        assert any_channel(torch.randn(2, 24, 9)).shape == (2, 24)


def test_rejects_wrong_input_length_and_ndim():
    model = DLinearForecaster(seq_len=24, pred_len=24).eval()
    with pytest.raises(ValueError):
        model(torch.randn(24, 7))
    with pytest.raises(ValueError):
        model(torch.randn(2, 12, 7))


def test_kernel_larger_than_seq_len_is_padded_ok():
    """kernel_size=25 > seq_len=24 是原论文默认；靠两端复制 pad 凑够长度。"""
    model = DLinearForecaster(seq_len=24, pred_len=12, kernel_size=25).eval()
    with torch.no_grad():
        assert model(torch.randn(3, 24, 7)).shape == (3, 12)


def test_config_records_construction_arguments():
    model = DLinearForecaster(seq_len=96, pred_len=96, kernel_size=25, target_idx=6)
    assert model.config == {"seq_len": 96, "pred_len": 96, "kernel_size": 25,
                            "target_idx": 6, "individual": False}
