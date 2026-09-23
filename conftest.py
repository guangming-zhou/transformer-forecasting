# -*- coding: utf-8 -*-
"""pytest 共享配置与 fixture。

放在项目根目录有两个作用：
1. 让 `python -m pytest` 从仓库根运行时能 import `utils` / `models`（把根目录塞进 sys.path）；
2. 提供只跑一次的数据管道 fixture（`prepare_data` 约 2-4 秒，不宜每个用例都跑）。
"""

from __future__ import annotations

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(scope="session")
def project_root() -> str:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def base_bundle() -> dict:
    """主设定（configs/base.yaml，24 -> 24）的完整数据管道产物。

    `save=False`：**绝不**让测试覆盖仓库里那份被 evaluate / demo / CI 依赖的
    `outputs/scaler.npz`。
    """
    from utils.dataset import load_data_config, prepare_data

    return prepare_data(load_data_config(), save=False, verbose=False)


@pytest.fixture(scope="session")
def lstf96_bundle() -> dict:
    """96 -> 96 设定的数据管道产物（同样 save=False）。"""
    from utils.dataset import load_data_config, prepare_data

    cfg_path = os.path.join(PROJECT_ROOT, "configs", "lstf96.yaml")
    if not os.path.isfile(cfg_path):
        pytest.skip("configs/lstf96.yaml 不存在")
    return prepare_data(load_data_config(cfg_path), save=False, verbose=False)


@pytest.fixture()
def work_tmp() -> str:
    """可写的临时目录。

    刻意不用 pytest 内置的 `tmp_path`：在受限沙箱里系统临时目录可能不可写
    （`PermissionError: [WinError 5]`），而 `outputs/_runs/` 一定在仓库内且被 gitignore。
    """
    path = os.path.join(PROJECT_ROOT, "outputs", "_runs", "_pytest_tmp")
    os.makedirs(path, exist_ok=True)
    return path
