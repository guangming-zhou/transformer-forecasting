# -*- coding: utf-8 -*-
"""FastAPI Demo 的接口自测（用 TestClient，不需要真的起服务器）。

覆盖：
1. GET /health、GET /models
2. POST /predict：用真实 test 窗口的原始 24×7 数值构造 CSV，校验返回 24 个预测
3. 返回的预测与「直接用 utils.dataset 管道算出来的结果」逐元素一致
   （证明 API 确实复用了数据管道，没有另写一套清洗/归一化）
4. 错误路径：行数不对、列数不对 -> 400
"""

from __future__ import annotations

import io
import os
import sys

import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, ""):
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)          # 脚本内使用 repo 相对路径，切到项目根保证到哪都能跑

from demo.app import app  # noqa: E402
from models.transformer import build_model  # noqa: E402
from utils.dataset import (  # noqa: E402
    FEATURES, TARGET_IDX, apply_scaler, build_windows_ranges, clean_dataframe,
    load_data_config, load_raw, load_scaler, split_ranges,
)
from utils.metrics import denorm_ot  # noqa: E402

client = TestClient(app)
cfg = load_data_config()
scaler = load_scaler(cfg.scaler_path)

print("=" * 78)
print("1) GET /health")
h = client.get("/health")
assert h.status_code == 200, h.text
hj = h.json()
print("   status =", hj["status"], " seq_len =", hj["seq_len"], " pred_len =", hj["pred_len"])
print("   seeds  =", hj["seeds"])
print("   ot_mean/std =", round(hj["ot_mean"], 4), round(hj["ot_std"], 4))
assert hj["status"] == "ok"

print("\n2) GET /models")
m = client.get("/models")
assert m.status_code == 200
print("   ", m.json())

print("\n3) 构造真实 test 窗口的 CSV 并 POST /predict")
clean, _ = clean_dataframe(load_raw(cfg), cfg)
ranges = split_ranges(clean.index, cfg)["test"]
x_raw, y_raw = build_windows_ranges(clean.to_numpy(dtype="float64"), ranges,
                                    cfg.seq_len, cfg.pred_len, TARGET_IDX)
win = 0
csv_bytes = pd.DataFrame(x_raw[win], columns=list(FEATURES)).to_csv(index=False).encode()
r = client.post("/predict", params={"seed": 42},
                files={"file": ("w.csv", csv_bytes, "text/csv")})
assert r.status_code == 200, r.text
out = r.json()
preds = np.asarray(out["preds"], dtype="float64")
print(f"   HTTP {r.status_code}; horizon={out['horizon']} unit={out['unit']}")
print(f"   预测前 5 个 = {preds[:5].round(3).tolist()}")
print(f"   预测范围   = [{preds.min():.3f}, {preds.max():.3f}] °C")
print(f"   输入末值   = {out['last_input_ot']:.3f} °C")
print(f"   cleaning   = {out['cleaning']}")
print(f"   model      = {out['model']}")
assert len(preds) == cfg.pred_len and np.isfinite(preds).all()

print("\n4) 交叉验证：API 结果 vs 直接走 utils.dataset 管道")
ck = torch.load("outputs/ckpt/seed42.pt", map_location="cpu", weights_only=True)
model = build_model(ck["model_config"])
model.load_state_dict(ck["model_state_dict"])
model.eval()
scaled, _ = apply_scaler(x_raw[win], scaler)
with torch.no_grad():
    ref = denorm_ot(model(torch.as_tensor(scaled[None], dtype=torch.float32)).numpy(), scaler)[0]
diff = float(np.abs(ref - preds).max())
print(f"   最大绝对偏差 = {diff:.3e}  ->  {'一致' if diff < 1e-4 else '不一致'}")
assert diff < 1e-4, "API 结果与数据管道结果不一致"

print(f"\n   顺带：本窗口 真值范围 = [{y_raw[win].min():.3f}, {y_raw[win].max():.3f}] °C，"
      f"模型 MAE = {np.mean(np.abs(preds - y_raw[win])):.4f} °C，"
      f"persistence MAE = {np.mean(np.abs(x_raw[win][-1, TARGET_IDX] - y_raw[win])):.4f} °C")

print("\n5) 错误路径")
bad_rows = pd.DataFrame(np.zeros((20, 7)), columns=list(FEATURES)).to_csv(index=False).encode()
r1 = client.post("/predict", files={"file": ("b.csv", bad_rows, "text/csv")})
print(f"   20 行 -> HTTP {r1.status_code}  {r1.json().get('detail')}")
assert r1.status_code == 400
bad_cols = pd.DataFrame(np.zeros((24, 5))).to_csv(index=False).encode()
r2 = client.post("/predict", files={"file": ("b.csv", bad_cols, "text/csv")})
print(f"   5 列  -> HTTP {r2.status_code}  {r2.json().get('detail')}")
assert r2.status_code == 400

print("\n全部断言通过 [OK]")
