# -*- coding: utf-8 -*-
"""`utils/seed.py` 单元测试：种子可复现、DataLoader 生成器与全局 RNG 隔离。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from utils.seed import make_generator, set_seed, worker_init_fn


def test_set_seed_makes_torch_and_numpy_reproducible():
    set_seed(123)
    a_t = torch.randn(8)
    a_n = np.random.rand(8)
    set_seed(123)
    assert torch.allclose(a_t, torch.randn(8))
    assert np.allclose(a_n, np.random.rand(8))


def test_set_seed_rejects_negative():
    with pytest.raises(ValueError):
        set_seed(-1)


def test_make_generator_is_isolated_from_global_rng():
    """打乱顺序不应被模型初始化等操作消耗的随机数影响。"""
    set_seed(0)
    g1 = make_generator(7)
    torch.randn(1000)                       # 模拟模型初始化消耗随机数
    g2 = make_generator(7)
    assert torch.randperm(50, generator=g1).tolist() == \
        torch.randperm(50, generator=g2).tolist()


def test_make_generator_different_seeds_give_different_order():
    a = torch.randperm(50, generator=make_generator(1)).tolist()
    b = torch.randperm(50, generator=make_generator(2)).tolist()
    assert a != b


def test_worker_init_fn_does_not_crash():
    worker_init_fn(0)                       # num_workers>0 时由 DataLoader 调用
