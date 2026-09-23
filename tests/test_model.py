# -*- coding: utf-8 -*-
"""`models/transformer.py` 单元测试：参数量、形状校验、残差语义、注意力路径等价性。"""

from __future__ import annotations

import torch
import pytest

from models.transformer import (
    TransformerForecaster, build_model, count_parameters, model_kwargs_from_dict,
    parameter_breakdown,
)
from utils.dataset import FEATURES, TARGET_IDX
from utils.seed import set_seed


def _model(**kw) -> TransformerForecaster:
    set_seed(0)
    return TransformerForecaster(**kw)


# --------------------------------------------------------------------------- #
# 参数量：README 里对外公布的数字，改动必须显式失败
# --------------------------------------------------------------------------- #
def test_default_param_count_matches_readme():
    total, trainable = count_parameters(_model())
    assert total == 105_880
    assert trainable == 105_880


def test_use_pe_false_removes_exactly_max_len_times_d_model():
    with_pe, _ = count_parameters(_model(use_pe=True))
    without_pe, _ = count_parameters(_model(use_pe=False))
    assert with_pe - without_pe == 24 * 64
    assert without_pe == 104_344


def test_parameter_breakdown_covers_all_submodules():
    model = _model()
    breakdown = parameter_breakdown(model)
    # pos_embed 是顶层 Parameter（不是子 module），所以单独加回来才对得上总数
    assert set(breakdown) == {"input_proj", "pos_drop", "encoder", "head"}
    assert sum(breakdown.values()) + model.pos_embed.numel() == count_parameters(model)[0]


def test_d_model_must_be_divisible_by_nhead():
    with pytest.raises(ValueError):
        TransformerForecaster(d_model=64, nhead=5)


def test_target_idx_must_be_within_input_dim():
    with pytest.raises(ValueError):
        TransformerForecaster(input_dim=7, target_idx=7)


def test_max_len_cannot_be_smaller_than_input_len():
    with pytest.raises(ValueError):
        TransformerForecaster(input_len=24, max_len=12)


# --------------------------------------------------------------------------- #
# 前向
# --------------------------------------------------------------------------- #
def test_forward_shape():
    model = _model().eval()
    x = torch.randn(4, 24, len(FEATURES))
    with torch.no_grad():
        assert tuple(model(x).shape) == (4, 24)


def test_forward_accepts_other_window_lengths():
    model = _model(input_len=96, pred_len=96).eval()
    with torch.no_grad():
        assert tuple(model(torch.randn(2, 96, 7)).shape) == (2, 96)


def test_forward_rejects_bad_shapes():
    model = _model().eval()
    with pytest.raises(ValueError):
        model(torch.randn(24, 7))                     # 不是三维
    with pytest.raises(ValueError):
        model(torch.randn(2, 12, 7))                  # 长度不对
    with pytest.raises(ValueError):
        model(torch.randn(2, 24, 8))                  # 通道数不对


def test_train_mode_is_stochastic_but_equals_eval_when_dropout_is_zero():
    x = torch.randn(4, 24, 7)
    stochastic = _model(dropout=0.5).train()
    with torch.no_grad():
        assert not torch.allclose(stochastic(x), stochastic(x))

    set_seed(0)
    deterministic = TransformerForecaster(dropout=0.0).train()
    with torch.no_grad():
        assert torch.allclose(deterministic(x), deterministic(x))


# --------------------------------------------------------------------------- #
# 残差头语义：Δ≡0 时输出必须**逐元素**等于「重复输入最后一步的 OT」
# --------------------------------------------------------------------------- #
def test_residual_with_zero_head_equals_persistence():
    model = _model(residual=True, pred_len=24).eval()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        x = torch.randn(5, 24, len(FEATURES))
        out = model(x)
    expected = x[:, -1, TARGET_IDX].unsqueeze(-1).expand(-1, 24)
    assert torch.allclose(out, expected, atol=0.0, rtol=0.0)


def test_residual_equals_base_plus_last_step_when_head_is_shared():
    """同一个 head 权重下，残差版输出 - 基础版输出 == 输入最后一步 OT。"""
    set_seed(0)
    base = TransformerForecaster(residual=False).eval()
    res = TransformerForecaster(residual=True).eval()
    res.load_state_dict(base.state_dict())
    x = torch.randn(3, 24, len(FEATURES))
    with torch.no_grad():
        diff = res(x) - base(x)
    assert torch.allclose(diff, x[:, -1, TARGET_IDX].unsqueeze(-1).expand(-1, 24), atol=1e-6)


def test_residual_default_is_off():
    assert TransformerForecaster().residual is False
    assert TransformerForecaster().target_idx == TARGET_IDX


# --------------------------------------------------------------------------- #
# 手工注意力路径必须与 nn.TransformerEncoder 数值等价
# --------------------------------------------------------------------------- #
def test_forward_with_attn_matches_forward():
    model = _model().eval()
    x = torch.randn(3, 24, len(FEATURES))
    with torch.no_grad():
        plain = model(x)
        with_attn, attn = model.forward_with_attn(x)
    assert torch.allclose(plain, with_attn, atol=1e-6)


def test_attention_weights_have_expected_shape_and_are_row_stochastic():
    model = _model().eval()
    x = torch.randn(2, 24, len(FEATURES))
    with torch.no_grad():
        _, attn = model.forward_with_attn(x)
    assert len(attn) == model.num_layers
    w = attn[0]
    assert tuple(w.shape) == (2, model.nhead, 24, 24)
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5)
    assert (w >= 0).all()


def test_forward_with_attn_carries_residual_term():
    model = _model(residual=True).eval()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        x = torch.randn(2, 24, 7)
        out, _ = model.forward_with_attn(x)
    assert torch.allclose(out, x[:, -1, TARGET_IDX].unsqueeze(-1).expand(-1, 24), atol=0.0)


# --------------------------------------------------------------------------- #
# 配置往返
# --------------------------------------------------------------------------- #
def test_config_roundtrip_reproduces_the_same_model():
    model = _model(d_model=32, nhead=8, num_layers=1, dim_ff=64, use_pe=False, residual=True)
    rebuilt = build_model(model_kwargs_from_dict(model.config()))
    assert rebuilt.config() == model.config()
    assert count_parameters(rebuilt)[0] == count_parameters(model)[0]


def test_model_kwargs_from_dict_ignores_unknown_keys():
    kwargs = model_kwargs_from_dict({"d_model": 32, "name": "transformer", "nope": 1})
    assert kwargs == {"d_model": 32}
