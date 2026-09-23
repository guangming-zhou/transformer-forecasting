# -*- coding: utf-8 -*-
"""随机种子 / 可复现性工具。"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch

__all__ = ["set_seed", "make_generator", "worker_init_fn"]


def set_seed(seed: int, deterministic: bool = True) -> int:
    """固定 python / numpy / torch(CPU+CUDA) 的随机种子。

    deterministic=True 时同时关掉 cudnn 的自动算法选择，保证同一 seed 可复现。
    """
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"seed 必须为非负整数，收到 {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def make_generator(seed: Optional[int] = None) -> torch.Generator:
    """给 DataLoader 的 shuffle 用的独立生成器。

    与全局 RNG 隔离，避免模型初始化等操作消耗随机数后改变打乱顺序。
    """
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(int(seed))
    return generator


def worker_init_fn(worker_id: int) -> None:
    """DataLoader 多进程 worker 的种子设置（num_workers>0 时使用）。"""
    base = torch.initial_seed() % (2 ** 32)
    np.random.seed((base + worker_id) % (2 ** 32))
    random.seed((base + worker_id) % (2 ** 32))
