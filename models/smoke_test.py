# -*- coding: utf-8 -*-
"""Transformer 前向 smoke test（不涉及训练）。

运行：
    python -m models.smoke_test
    python models/smoke_test.py
"""

from __future__ import annotations

import os
import sys

import torch

if __package__ in (None, ""):                      # 允许直接 python models/smoke_test.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from models.transformer import (  # noqa: E402
    build_model, count_parameters, load_model_config, parameter_breakdown,
)

BAR = "=" * 78


def main() -> int:
    torch.manual_seed(0)
    device = torch.device("cpu")

    # ---------------- 配置来源 ----------------
    cfg = load_model_config()                       # 读取 configs/base.yaml 的 model 段
    print(f"config = configs/base.yaml [model]")
    print(f"kwargs = {cfg}\n")

    # ---------------- 构建模型 ----------------
    model = build_model(cfg).to(device)
    print(model)
    print()

    total, trainable = count_parameters(model)
    print(BAR)
    print("[参数量]")
    print(BAR)
    print(f"  总参数量     : {total:,}")
    print(f"  可训练参数量 : {trainable:,}")
    print("  按模块拆解:")
    for name, n in parameter_breakdown(model).items():
        print(f"    {name:<12s} {n:>10,}")
    if model.pos_embed is not None:
        print(f"    {'pos_embed':<12s} {model.pos_embed.numel():>10,}"
              f"   shape={tuple(model.pos_embed.shape)}")
    print(f"  可训练占比   : {trainable / total * 100:.1f}%")

    # ---------------- 默认配置前向 ----------------
    x = torch.randn(2, 24, 7, device=device)
    model.eval()
    with torch.no_grad():
        y = model(x)

    print(f"\n{BAR}")
    print("[前向 smoke test] 默认配置（use_pe=True）")
    print(BAR)
    print(f"  输入 x.shape = {tuple(x.shape)}")
    print(f"  输出 y.shape = {tuple(y.shape)}")
    print(f"  输出 dtype   = {y.dtype}")
    print(f"  输出统计     : min={y.min().item():.4f}  max={y.max().item():.4f}  "
          f"mean={y.mean().item():.4f}  std={y.std().item():.4f}")
    assert tuple(y.shape) == (2, 24), f"输出 shape 应为 (2, 24)，实际 {tuple(y.shape)}"
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all(), "输出出现 NaN/Inf"
    print("  ✓ 输出 shape == (2, 24)")

    # train 模式下也要能跑通（Dropout 生效）
    model.train()
    y_train_mode = model(x)
    assert tuple(y_train_mode.shape) == (2, 24)
    print("  ✓ train() 模式（Dropout 生效）同样输出 (2, 24)")

    # ---------------- 残差开关（residual）语义验证 ----------------
    print(f"\n{BAR}")
    print("[语义验证] residual=True：Δ≡0 时必须精确退化为 persistence")
    print(BAR)
    import torch.nn as nn
    from models.transformer import TransformerForecaster

    m_res = TransformerForecaster(use_pe=False, residual=True).eval()
    with torch.no_grad():
        m_res.head.weight.zero_()
        m_res.head.bias.zero_()
        y_res = m_res(x)
    last_ot = x[:, -1, m_res.target_idx].unsqueeze(-1)      # (B, 1)
    expect = last_ot.expand(-1, 24)
    err = float((y_res - expect).abs().max())
    print(f"  head 全零时 输出 与 输入最后一步 OT 的最大偏差 = {err:.3e}")
    assert err < 1e-6, "residual=True 且 Δ=0 时应精确等于「重复最后一步观测」"
    assert tuple(y_res.shape) == (2, 24)
    print("  ✓ residual=True 且 Δ≡0 时输出 == 重复最后一步观测（persistence 等价）")
    print("  ✓ 这意味着 persistence 就是这个模型的起点，模型只需学「变化量」")

    m_def = TransformerForecaster(use_pe=False, residual=False).eval()
    print(f"  ✓ residual 默认值 = {m_def.residual}（默认行为不变）")
    assert m_def.residual is False

    # ---------------- use_pe=False ----------------
    print(f"\n{BAR}")
    print("[前向 smoke test] use_pe=False")
    print(BAR)
    model_no_pe = build_model(cfg, use_pe=False).to(device).eval()
    with torch.no_grad():
        y_no_pe = model_no_pe(x)
    total_np, trainable_np = count_parameters(model_no_pe)
    print(f"  输出 y.shape = {tuple(y_no_pe.shape)}")
    print(f"  总参数量     : {total_np:,}   （比 use_pe=True 少 {total - total_np:,}）")
    print(f"  可训练参数量 : {trainable_np:,}")
    assert tuple(y_no_pe.shape) == (2, 24), f"use_pe=False 输出 shape 异常 {tuple(y_no_pe.shape)}"
    assert torch.isfinite(y_no_pe).all()
    assert model_no_pe.pos_embed is None
    assert total - total_np == 24 * 64, "差值应等于位置编码参数量 24*64"
    print("  ✓ use_pe=False 可正常前向，输出 shape == (2, 24)")
    print(f"  ✓ pos_embed is None；参数量差值 = {24 * 64:,}（= max_len 24 × d_model 64）")

    # ---------------- 消融参数可调 ----------------
    print(f"\n{BAR}")
    print("[消融参数可调]")
    print(BAR)
    for kw in ({"d_model": 32, "nhead": 4}, {"num_layers": 1}, {"nhead": 8, "d_model": 64},
               {"pred_len": 48, "input_len": 24}, {"dim_ff": 256}):
        m = build_model(cfg, **kw).eval()
        in_len, out_len = kw.get("input_len", 24), kw.get("pred_len", 24)
        with torch.no_grad():
            out = m(torch.randn(2, in_len, 7))
        t, _ = count_parameters(m)
        assert tuple(out.shape) == (2, out_len)
        label = ", ".join(f"{k}={v}" for k, v in kw.items())
        print(f"  {label:<34s} -> 输出 {tuple(out.shape)}   参数量 {t:,}")
    print("  ✓ use_pe / nhead / num_layers / d_model / input_len / pred_len 均可调")

    # ---------------- 反传可达性 ----------------
    model.train()
    out = model(x)
    loss = out.mean()
    loss.backward()
    grad_ok = model.input_proj.weight.grad is not None and \
        torch.isfinite(model.input_proj.weight.grad).all() and \
        float(model.input_proj.weight.grad.abs().sum()) > 0
    assert grad_ok, "梯度未正确回传"
    print(f"\n  ✓ 反传可达：input_proj.weight.grad 非空且有限"
          f"（|grad| 之和 = {float(model.input_proj.weight.grad.abs().sum()):.3e}）")
    model.zero_grad(set_to_none=True)

    print("\n全部断言通过 [OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
