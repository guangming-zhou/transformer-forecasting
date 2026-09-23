# -*- coding: utf-8 -*-
"""配置文件契约测试。

最有价值的一条是 `test_lstf96_differs_from_base_only_in_window_length`：
96->96 与 24->24 的对比只有在「配方完全相同、只有窗口长度不同」时才成立，
这条测试把这个前提钉死，任何人不小心改了其中一个 yaml 的 lr/epochs 都会立刻失败。
"""

from __future__ import annotations

import os

import pytest
import yaml

from utils.dataset import data_config_from_config
from models.transformer import model_kwargs_from_dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(PROJECT_ROOT, "configs", "base.yaml")
LSTF96 = os.path.join(PROJECT_ROOT, "configs", "lstf96.yaml")
OFFICIAL96 = os.path.join(PROJECT_ROOT, "configs", "official_etth1_96.yaml")


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def base_cfg() -> dict:
    return _load(BASE)


@pytest.fixture(scope="module")
def lstf96_cfg() -> dict:
    if not os.path.isfile(LSTF96):
        pytest.skip("configs/lstf96.yaml 不存在")
    return _load(LSTF96)


@pytest.fixture(scope="module")
def official96_cfg() -> dict:
    return _load(OFFICIAL96)


@pytest.mark.parametrize("section", ["data", "model", "train", "output"])
def test_base_yaml_has_all_sections(base_cfg, section):
    assert isinstance(base_cfg.get(section), dict) and base_cfg[section]


def test_base_yaml_data_and_model_windows_agree(base_cfg):
    data, model = base_cfg["data"], base_cfg["model"]
    assert data["seq_len"] == model["input_len"] == 24
    assert data["pred_len"] == model["pred_len"] == 24


def test_base_yaml_train_recipe_is_the_documented_one(base_cfg):
    train = base_cfg["train"]
    assert train["lr"] == pytest.approx(1e-3)
    assert train["weight_decay"] == pytest.approx(1e-4)
    assert train["batch_size"] == 64
    assert train["epochs"] == 30
    assert train["grad_clip"] == pytest.approx(1.0)
    assert train["scheduler"] == "cosine"
    assert train["patience"] == 8
    assert train["warmup_epochs"] == 0          # 默认关闭：历史主结果不含 warmup
    assert train["seed"] == 42


def test_base_yaml_builds_a_constructible_config(base_cfg):
    cfg = data_config_from_config(base_cfg)
    assert (cfg.seq_len, cfg.pred_len) == (24, 24)
    assert cfg.split_mode == "time_sequential"
    assert cfg.features == ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")
    assert cfg.target == "OT" and cfg.target_idx == 6
    kwargs = model_kwargs_from_dict(base_cfg["model"])
    assert kwargs["input_len"] == cfg.seq_len and kwargs["pred_len"] == cfg.pred_len


def test_lstf96_differs_from_base_only_in_window_length(base_cfg, lstf96_cfg):
    assert base_cfg["train"] == lstf96_cfg["train"], "两个设定的训练配方必须逐字段相同"
    assert base_cfg["model"]["name"] == lstf96_cfg["model"]["name"]

    data_diff = {k for k in set(base_cfg["data"]) | set(lstf96_cfg["data"])
                 if base_cfg["data"].get(k) != lstf96_cfg["data"].get(k)}
    assert data_diff == {"seq_len", "pred_len"}

    model_diff = {k for k in set(base_cfg["model"]) | set(lstf96_cfg["model"])
                  if base_cfg["model"].get(k) != lstf96_cfg["model"].get(k)}
    assert model_diff == {"input_len", "pred_len"}

    out_diff = {k for k in set(base_cfg["output"]) | set(lstf96_cfg["output"])
                if base_cfg["output"].get(k) != lstf96_cfg["output"].get(k)}
    assert out_diff == {"scaler_path", "ckpt_dir", "log_dir"}


def test_lstf96_windows_are_96(lstf96_cfg):
    assert lstf96_cfg["data"]["seq_len"] == lstf96_cfg["model"]["input_len"] == 96
    assert lstf96_cfg["data"]["pred_len"] == lstf96_cfg["model"]["pred_len"] == 96
    cfg = data_config_from_config(lstf96_cfg)
    assert (cfg.seq_len, cfg.pred_len) == (96, 96)
    assert cfg.scaler_path.endswith("scaler96.npz"), "不得让训练覆盖 outputs/scaler.npz"


def test_residual_flag_defaults_to_false_in_both_configs(base_cfg, lstf96_cfg):
    assert base_cfg["model"]["residual"] is False
    assert lstf96_cfg["model"]["residual"] is False


def test_official96_config_is_raw_leakage_free_protocol(official96_cfg):
    data = official96_cfg["data"]
    model = official96_cfg["model"]
    assert data["split_mode"] == "official_etth1"
    assert data["seq_len"] == model["input_len"] == 96
    assert data["pred_len"] == model["pred_len"] == 96
    assert data["dedup_policy"] == "keep"
    assert data["zero_run_clean"] is False
    assert data["zero_min_cols"] > len(data["feature_cols"])
    assert data["clip_sigma"] == 0.0
    assert official96_cfg["output"]["scaler_path"].endswith("scaler_official96.npz")
