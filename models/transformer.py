# -*- coding: utf-8 -*-
"""Encoder-only Transformer 多步预测模型。

    x: (B, 24, 7)  ->  y: (B, 24)

结构
    1. Linear(input_dim -> d_model)
    2. 可学习位置编码 nn.Parameter(zeros(1, max_len, d_model)) + trunc_normal_ 初始化，
       外面套 Dropout（use_pe=False 时不加位置编码，但 Dropout 仍然生效，
       保证消融时 use_pe 是唯一变量）
    3. nn.TransformerEncoder(norm_first=True, batch_first=True, activation="gelu")
    4. flatten -> Linear(d_model * input_len, pred_len)

d_model / nhead / num_layers / dim_ff / dropout / use_pe / input_len / pred_len
全部是构造参数，供后续消融实验使用。
"""

from __future__ import annotations

import inspect
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import yaml

__all__ = [
    "TransformerForecaster", "count_parameters", "parameter_breakdown",
    "load_model_config", "build_model",
]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "base.yaml")


class TransformerForecaster(nn.Module):
    """Encoder-only Transformer：输入 (B, input_len, input_dim)，输出 (B, pred_len)。"""

    def __init__(
        self,
        input_dim: int = 7,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.1,
        input_len: int = 24,
        pred_len: int = 24,
        use_pe: bool = True,
        activation: str = "gelu",
        norm_first: bool = True,
        max_len: Optional[int] = None,
        residual: bool = False,
        target_idx: int = 6,
    ) -> None:
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model({d_model}) 必须能被 nhead({nhead}) 整除")
        if input_len <= 0 or pred_len <= 0:
            raise ValueError("input_len / pred_len 必须为正整数")

        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_ff = int(dim_ff)
        self.dropout_p = float(dropout)
        self.input_len = int(input_len)
        self.pred_len = int(pred_len)
        self.use_pe = bool(use_pe)
        self.activation = str(activation)
        self.norm_first = bool(norm_first)
        # 残差预测：输出 = 输入最后一步的目标值 + Δ，模型只学「变化量」。
        # persistence 等价于 Δ≡0，因此这个开关把平凡基线变成模型的起点。
        # 默认 False —— 保持 forward() 的默认行为不变。
        self.residual = bool(residual)
        self.target_idx = int(target_idx)
        if not 0 <= self.target_idx < self.input_dim:
            raise ValueError(f"target_idx({self.target_idx}) 超出 input_dim({self.input_dim})")
        # max_len 默认等于 input_len；给更大的值可以复用同一个模型跑更长序列
        self.max_len = int(max_len) if max_len is not None else self.input_len
        if self.max_len < self.input_len:
            raise ValueError(f"max_len({self.max_len}) 不能小于 input_len({self.input_len})")

        # 1) 输入投影
        self.input_proj = nn.Linear(self.input_dim, self.d_model)

        # 2) 可学习位置编码（use_pe=False 时不创建参数）
        if self.use_pe:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.max_len, self.d_model))
        else:
            self.register_parameter("pos_embed", None)
        self.pos_drop = nn.Dropout(self.dropout_p)

        # 3) Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_ff,
            dropout=self.dropout_p,
            activation=self.activation,
            batch_first=True,
            norm_first=self.norm_first,
        )
        # norm_first=True 时嵌套张量优化本来就不会启用，显式关掉以消除 PyTorch 警告
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers, enable_nested_tensor=False)

        # 4) 预测头：flatten 后一次性映射到 pred_len
        self.head = nn.Linear(self.d_model * self.input_len, self.pred_len)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"输入需为 (B, L, C) 三维张量，收到 {tuple(x.shape)}")
        b, length, channels = x.shape
        if length != self.input_len:
            raise ValueError(f"输入长度需为 {self.input_len}，收到 {length}")
        if channels != self.input_dim:
            raise ValueError(f"输入特征数需为 {self.input_dim}，收到 {channels}")

        h = self.input_proj(x)                       # (B, L, d_model)
        if self.pos_embed is not None:
            h = h + self.pos_embed[:, :length, :]    # 广播到 batch
        h = self.pos_drop(h)                         # 位置编码外层的 Dropout
        h = self.encoder(h)                          # (B, L, d_model)
        h = h.reshape(b, length * self.d_model)      # flatten
        out = self.head(h)                           # (B, pred_len)
        if self.residual:
            # 残差预测：out = 输入最后一步的目标值 + Δ（Δ 由 head 给出）。
            # 归一化空间里相加，反归一化后等价于「在最后一步观测值上叠加预测的变化量」。
            out = out + x[:, -1, self.target_idx].unsqueeze(-1)
        return out                                   # (B, pred_len)

    def forward_with_attn(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """与 forward() 数值等价，但额外返回每层 attention 权重。

        返回 `(output, attn_list)`，其中 `attn_list[i]` 形状 **(B, nhead, L, L)**
        （未对 head 取平均）。`forward()` 的默认行为完全不变。

        实现说明：`nn.TransformerEncoderLayer` 内部以 `need_weights=False` 调用自注意力
        （为了走 fast path），因此 register_forward_hook 拿不到权重。这里按 PyTorch
        `norm_first=True` 的定义手工跑一遍 encoder 层，并在自注意力处显式
        `need_weights=True, average_attn_weights=False`。在 `norm_first=True` 且无 mask 时，
        该手工路径与 `nn.TransformerEncoder` 逐层等价（norm → 自注意力残差 → FFN 残差），
        数值一致性由 `eval/visualize_attn.py` 断言校验。
        """
        if x.dim() != 3:
            raise ValueError(f"输入需为 (B, L, C) 三维张量，收到 {tuple(x.shape)}")
        b, length, channels = x.shape
        if length != self.input_len:
            raise ValueError(f"输入长度需为 {self.input_len}，收到 {length}")
        if channels != self.input_dim:
            raise ValueError(f"输入特征数需为 {self.input_dim}，收到 {channels}")

        h = self.input_proj(x)
        if self.pos_embed is not None:
            h = h + self.pos_embed[:, :length, :]
        h = self.pos_drop(h)

        attn_list: list = []
        for layer in self.encoder.layers:
            normed = layer.norm1(h)
            attn_out, attn_w = layer.self_attn(
                normed, normed, normed,
                need_weights=True, average_attn_weights=False)   # (B, nhead, L, L)
            attn_list.append(attn_w.detach())
            h = h + layer.dropout1(attn_out)
            h = h + layer.dropout2(
                layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(h))))))

        h = h.reshape(b, length * self.d_model)
        out = self.head(h)
        if self.residual:
            out = out + x[:, -1, self.target_idx].unsqueeze(-1)
        return out, attn_list

    # ---- 便于消融/日志 ----
    def config(self) -> Dict:
        return {
            "input_dim": self.input_dim, "d_model": self.d_model, "nhead": self.nhead,
            "num_layers": self.num_layers, "dim_ff": self.dim_ff, "dropout": self.dropout_p,
            "input_len": self.input_len, "pred_len": self.pred_len, "use_pe": self.use_pe,
            "activation": self.activation, "norm_first": self.norm_first,
            "max_len": self.max_len, "residual": self.residual,
            "target_idx": self.target_idx,
        }

    def extra_repr(self) -> str:
        c = self.config()
        return (f"input=({c['input_len']}, {c['input_dim']}) -> output=({c['pred_len']},)  "
                f"d_model={c['d_model']} nhead={c['nhead']} layers={c['num_layers']} "
                f"dim_ff={c['dim_ff']} dropout={c['dropout']} use_pe={c['use_pe']} "
                f"activation={c['activation']} norm_first={c['norm_first']}")


# --------------------------------------------------------------------------- #
# 参数统计 / 配置
# --------------------------------------------------------------------------- #
def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """返回 (总参数量, 可训练参数量)。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def parameter_breakdown(model: nn.Module) -> Dict[str, int]:
    """按顶层子模块统计参数量，便于确认结构。"""
    out: Dict[str, int] = {}
    for name, module in model.named_children():
        out[name] = sum(p.numel() for p in module.parameters())
    return out


def model_kwargs_from_dict(section: Optional[Dict], **overrides) -> Dict:
    """从配置段（通常是 yaml 的 `model:`）过滤出模型构造参数。"""
    allowed = set(inspect.signature(TransformerForecaster.__init__).parameters)
    kwargs = {k: v for k, v in (section or {}).items()
              if k in allowed and k not in ("self", "name")}
    kwargs.update(overrides)
    return kwargs


def load_model_config(path: Optional[str] = None, **overrides) -> Dict:
    """从 configs/base.yaml 的 `model:` 段读取模型超参（未知键会被忽略）。"""
    path = path or DEFAULT_CONFIG_PATH
    if not os.path.isfile(path):
        return dict(overrides)

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    section = raw.get("model", {}) if isinstance(raw, dict) else {}
    if not isinstance(section, dict):
        raise ValueError(f"{path} 的 model 段内容不是键值对")

    return model_kwargs_from_dict(section, **overrides)


def build_model(cfg: Optional[Dict] = None, **overrides) -> TransformerForecaster:
    """按 configs/base.yaml 的 model 段构建模型，overrides 优先级最高。"""
    kwargs = dict(cfg or load_model_config())
    kwargs.update(overrides)
    kwargs.pop("name", None)
    return TransformerForecaster(**kwargs)
