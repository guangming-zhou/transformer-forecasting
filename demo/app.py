#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FastAPI 推理服务：上传 24×7 的历史窗口 CSV，返回未来 24 小时 OT（摄氏度）。

设计约束
* **复用 `utils/dataset.py`**：清洗走 `clean_dataframe`、归一化/反归一化走
  `apply_scaler` / `utils.metrics.denorm_ot`，scaler 一律读 `outputs/scaler.npz`。
  本文件不重新实现任何清洗或缩放逻辑。
* 模型按 seed 惰性加载并缓存。

启动
    python demo/app.py                     # 默认 127.0.0.1:8000
    uvicorn demo.app:app --host 0.0.0.0 --port 8000

接口
    GET  /health          服务与可用 seed
    GET  /models          可用 ckpt 列表
    POST /predict?seed=42 上传 CSV（24 行 × 7 列）-> 24 步预测
"""

from __future__ import annotations

import io
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer import build_model  # noqa: E402
from utils.dataset import (  # noqa: E402
    apply_scaler, clean_dataframe, load_data_config, load_scaler,
)
from utils.metrics import denorm_ot, target_index  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="ETTh1 OT 24 步预测服务", version="1.0")

_CFG = load_data_config()
_SCALER = load_scaler(_CFG.scaler_path)

# 列名与目标列一律从 scaler 里读（`save_scaler` 会把 features/target/target_idx 一起落盘），
# 这样换一份数据只要换 yaml + 重训，Demo 不需要改代码。
FEATURES = tuple(str(c) for c in _SCALER["features"])
TARGET_IDX = target_index(_SCALER)
TARGET_NAME = str(_SCALER["target"])

_MODEL_CACHE: Dict[int, torch.nn.Module] = {}


# --------------------------------------------------------------------------- #
# 模型加载
# --------------------------------------------------------------------------- #
def ckpt_path_for(seed: int) -> str:
    """默认从 outputs/ckpt/seed{seed}.pt 取；可用环境变量 DSH_CKPT_DIR 覆盖。"""
    ckpt_dir = os.environ.get("DSH_CKPT_DIR", os.path.join(PROJECT_ROOT, "outputs", "ckpt"))
    return os.path.join(ckpt_dir, f"seed{seed}.pt")


def available_seeds() -> List[int]:
    ckpt_dir = os.path.dirname(ckpt_path_for(0))
    if not os.path.isdir(ckpt_dir):
        return []
    seeds: List[int] = []
    for name in os.listdir(ckpt_dir):
        if name.startswith("seed") and name.endswith(".pt"):
            try:
                seeds.append(int(name[4:-3]))
            except ValueError:
                pass
    return sorted(seeds)


def get_model(seed: int) -> torch.nn.Module:
    if seed in _MODEL_CACHE:
        return _MODEL_CACHE[seed]
    path = ckpt_path_for(seed)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404,
                            detail=f"找不到 seed={seed} 的 ckpt: {path}")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ck["model_config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    _MODEL_CACHE[seed] = model
    return model


# --------------------------------------------------------------------------- #
# 输入解析（只做格式校验，清洗交给 utils.dataset）
# --------------------------------------------------------------------------- #
def parse_window(raw: bytes, seq_len: int) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"CSV 解析失败: {exc}") from exc

    # 容忍多带一列时间戳
    if df.shape[1] == len(FEATURES) + 1:
        df = df.iloc[:, 1:]
    if df.shape[1] != len(FEATURES):
        raise HTTPException(
            status_code=400,
            detail=f"需要 {len(FEATURES)} 列 {list(FEATURES)}，收到 {df.shape[1]} 列")
    if len(df) != seq_len:
        raise HTTPException(status_code=400,
                            detail=f"需要 {seq_len} 行（小时），收到 {len(df)} 行")

    df = df.iloc[:, :len(FEATURES)].copy()
    df.columns = list(FEATURES)                       # 按位置对齐
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any():
        raise HTTPException(status_code=400, detail="CSV 含非数值或缺失值")
    # 上传数据没有时间戳，给一个连续小时索引（清洗里的时间字段只用于报告）
    df.index = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=seq_len, freq="h")
    return df


# --------------------------------------------------------------------------- #
# 路由
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> Dict:
    return {"status": "ok", "seq_len": _CFG.seq_len, "pred_len": _CFG.pred_len,
            "features": list(FEATURES), "seeds": available_seeds(),
            "scaler": os.path.abspath(_CFG.scaler_path),
            "ot_mean": float(_SCALER["mean"][TARGET_IDX]),
            "ot_std": float(_SCALER["std"][TARGET_IDX])}


@app.get("/models")
def models() -> Dict:
    out = []
    for s in available_seeds():
        p = ckpt_path_for(s)
        out.append({"seed": s, "path": os.path.abspath(p),
                    "size_kb": round(os.path.getsize(p) / 1024, 1)})
    return {"models": out}


@app.post("/predict")
async def predict(file: UploadFile = File(...),
                  seed: int = Query(42, description="使用哪个 seed 的 ckpt")) -> JSONResponse:
    raw = await file.read()
    df = parse_window(raw, _CFG.seq_len)

    # ---- 复用数据管道：清洗（重复块 / 行级零值 / 单列连续 0）----
    clean, clean_info = clean_dataframe(df, _CFG)

    # ---- 复用 scaler 归一化（train 段拟合，绝不重算）----
    scaled, _ = apply_scaler(clean.to_numpy(dtype="float64"), _SCALER)
    x = torch.as_tensor(scaled[None, :, :], dtype=torch.float32)

    model = get_model(seed)
    with torch.no_grad():
        pred_norm = model(x).numpy()
    preds = denorm_ot(pred_norm, _SCALER)[0]                  # (24,) 摄氏度

    last_ts = clean.index[-1]
    target_ts = pd.date_range(start=last_ts + pd.Timedelta(hours=1),
                              periods=_CFG.pred_len, freq="h")

    return JSONResponse({
        "seed": seed,
        "horizon": int(_CFG.pred_len),
        "unit": "degC",
        "preds": [round(float(v), 4) for v in preds],
        "input_ot": [round(float(v), 4) for v in clean[TARGET_NAME].to_numpy()],
        "raw_input_ot": [round(float(v), 4) for v in df[TARGET_NAME].to_numpy()],
        "last_input_ot": round(float(clean[TARGET_NAME].to_numpy()[-1]), 4),
        "target_times": [str(t) for t in target_ts],
        "model": {"seed": seed, "params": int(sum(p.numel() for p in model.parameters())),
                  "use_pe": bool(model.use_pe), "num_layers": int(model.num_layers),
                  "nhead": int(model.nhead), "input_len": int(model.input_len)},
        "cleaning": {"cells_masked": int(clean_info["cells_masked"]),
                     "rows_touched": int(clean_info["rows_touched"])},
        "scaler": {"ot_mean": float(_SCALER["mean"][TARGET_IDX]),
                   "ot_std": float(_SCALER["std"][TARGET_IDX])},
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
