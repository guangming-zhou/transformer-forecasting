#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Streamlit 前端：上传 CSV 或从 test 集随机抽窗口，调用 FastAPI /predict 并画图。

启动（需要 FastAPI 服务已在跑）
    python demo/app.py                                     # 终端 1
    streamlit run demo/streamlit_app.py                    # 终端 2

功能
* 数据来源二选一：上传 CSV（复用 demo/app.py 的格式约定）/ 从 test 集随机抽一个窗口
  （本地用 `utils.dataset` 取原始窗口，顺便带上真值便于对照）
* 可选 seed（默认列出 outputs/ckpt/ 下的全部 seed），直观展示不同 seed 的预测差异
* 画输入历史 OT + 模型预测（+ 真值，若来自 test 集）
"""

from __future__ import annotations

import io
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402

from utils.dataset import (  # noqa: E402
    build_windows_ranges, clean_dataframe, load_data_config, load_raw, split_ranges,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_API = os.environ.get("DSH_API", "http://127.0.0.1:8000")
_CFG = load_data_config()
FEATURES = _CFG.features
TARGET_IDX = _CFG.target_idx


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_test_windows() -> Tuple[np.ndarray, np.ndarray]:
    """取 test 集的**原始（摄氏度）**窗口：x (N, 24, 7) 与真值 y (N, 24)。"""
    cfg = _CFG
    clean, _ = clean_dataframe(load_raw(cfg), cfg)
    ranges = split_ranges(clean.index, cfg)["test"]
    x, y = build_windows_ranges(clean.to_numpy(dtype="float64"), ranges,
                                cfg.seq_len, cfg.pred_len, cfg.target_idx)
    return x.astype("float64"), y.astype("float64")


def window_to_csv(x_raw: np.ndarray) -> bytes:
    return pd.DataFrame(x_raw, columns=list(FEATURES)).to_csv(index=False).encode("utf-8")


def call_predict(api: str, csv_bytes: bytes, seed: int) -> Dict:
    resp = requests.post(f"{api.rstrip('/')}/predict",
                         params={"seed": seed},
                         files={"file": ("window.csv", csv_bytes, "text/csv")},
                         timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
    return resp.json()


def plot_result(input_ot: np.ndarray, preds: np.ndarray,
                gts: Optional[np.ndarray], title: str):
    seq_len, pred_len = len(input_ot), len(preds)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(np.arange(-seq_len + 1, 1), input_ot, color="tab:gray", lw=1.6,
            marker="o", ms=2.5, label="输入历史 OT")
    if gts is not None:
        ax.plot(np.arange(1, pred_len + 1), gts, color="tab:blue", lw=1.8,
                marker="o", ms=3, label="真值")
    ax.plot(np.arange(1, pred_len + 1), preds, color="tab:red", lw=1.8, ls="--",
            marker="s", ms=3, label="模型预测")
    ax.axvline(0, color="k", lw=0.9, alpha=0.5)
    ax.axhline(float(input_ot[-1]), color="gray", ls=":", lw=1.0,
               label=f"persistence（重复最后一步 {input_ot[-1]:.2f}°C）")
    ax.set_xlabel("相对预测起点的小时数")
    ax.set_ylabel("OT (°C)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 页面
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="ETTh1 OT 24 步预测 Demo", layout="wide")
    st.title("ETTh1 · OT 未来 24 小时预测 Demo")

    with st.sidebar:
        st.header("设置")
        api = st.text_input("FastAPI 地址", DEFAULT_API)

        seeds: List[int] = [42]
        try:
            health = requests.get(f"{api.rstrip('/')}/health", timeout=5).json()
            seeds = health.get("seeds") or [42]
            st.success(f"服务在线 · seq_len={health['seq_len']} pred_len={health['pred_len']}")
        except Exception as exc:                       # noqa: BLE001
            st.error(f"连不上 FastAPI：{exc}\n\n先运行 `python demo/app.py`")

        seed = st.selectbox("seed（不同 seed 的预测差异）", seeds,
                            index=seeds.index(42) if 42 in seeds else 0)
        source = st.radio("数据来源", ["从 test 集随机抽一个窗口", "上传 CSV"])

    x_raw = gts = None
    if source.startswith("从 test"):
        x_all, y_all = load_test_windows()
        if st.button("🎲 换一个窗口") or "win_idx" not in st.session_state:
            st.session_state["win_idx"] = int(np.random.randint(len(x_all)))
        idx = st.session_state["win_idx"]
        x_raw, gts = x_all[idx], y_all[idx]
        st.caption(f"test 窗口 #{idx} / {len(x_all)}（x_raw {x_raw.shape}，真值 {gts.shape}）")
    else:
        up = st.file_uploader("上传 CSV：24 行 × 7 列（HUFL,HULL,MUFL,MULL,LUFL,LULL,OT）",
                              type=["csv"])
        if up is not None:
            raw = up.getvalue()
            try:
                x_raw = pd.read_csv(io.BytesIO(raw)).apply(
                    pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
            except Exception as exc:                   # noqa: BLE001
                st.error(f"CSV 解析失败：{exc}")
                return

    if x_raw is None:
        st.info("选择数据来源后开始预测。")
        return

    st.subheader("输入窗口（24 行 × 7 列）")
    st.dataframe(pd.DataFrame(x_raw, columns=list(FEATURES)).round(3), height=240)

    try:
        with st.spinner("调用 /predict ..."):
            out = call_predict(api, window_to_csv(x_raw), int(seed))
    except Exception as exc:                           # noqa: BLE001
        st.error(f"预测失败：{exc}")
        return

    preds = np.asarray(out["preds"], dtype="float64")
    input_ot = np.asarray(out["input_ot"], dtype="float64")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("输入末值", f"{out['last_input_ot']:.2f} °C")
    c2.metric("预测均值", f"{preds.mean():.2f} °C")
    c3.metric("预测范围", f"{preds.min():.2f} ~ {preds.max():.2f} °C")
    if gts is not None:
        c4.metric("本窗口 MAE（vs 真值）", f"{np.mean(np.abs(preds - gts)):.3f} °C")
    else:
        c4.metric("模型参数量", f"{out['model']['params']:,}")

    st.pyplot(plot_result(input_ot, preds, gts,
                          f"seed={out['seed']} · {out['model']['num_layers']} 层 · "
                          f"use_pe={out['model']['use_pe']}"))
    with st.expander("原始响应 JSON"):
        st.json(out)


if __name__ == "__main__":
    main()
