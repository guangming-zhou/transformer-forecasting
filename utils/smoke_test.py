# -*- coding: utf-8 -*-
"""数据管道 smoke test：清洗（3 条规则）-> 划分 -> 归一化 -> 滑窗 -> 取一个 batch。

运行：
    python -m utils.smoke_test
    python utils/smoke_test.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

if __package__ in (None, ""):                      # 允许直接 python utils/smoke_test.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows 控制台默认 GBK，中文报告与符号会乱码/报错。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from utils.dataset import (  # noqa: E402
    FEATURES, INPUT_FEATURES, TARGET_IDX, denormalize_ot, load_data_config,
    load_raw, load_scaler, make_dataloaders, max_zero_run_lengths, prepare_data,
)

BAR = "=" * 78


def check_cleaning(raw, clean, info) -> None:
    """核对清洗行为是否符合既定策略。

    关键守恒律是「改写范围 ⊆ 标记范围」：只有被标记为异常的单元格才会被插值覆盖，
    其余单元格必须逐位不变。反过来不成立——被标记的单元格插值后可能恰好等于原值
    （例如某列前后都是 0，线性插值结果仍是 0），这属于正确行为，不算漏清洗。
    """
    raw_v = raw.to_numpy()
    clean_v = clean.to_numpy()
    changed = ~np.isclose(raw_v, clean_v, rtol=0.0, atol=1e-9)
    mask = info["cell_mask"]

    print("\n[清洗自检]")
    print("  标记为缺失的单元格 按列:", dict(zip(FEATURES, mask.sum(axis=0).tolist())))
    print("                      合计:", int(mask.sum()))
    print("  插值后数值真变化的 按列:", dict(zip(FEATURES, changed.sum(axis=0).tolist())))
    print("                      合计:", int(changed.sum()))

    assert not (changed & ~mask).any(), "改写了未被标记的单元格"
    assert int(mask.sum()) == info["cells_masked"]
    print("  ✓ 改写范围 ⊆ 标记范围（未标记的单元格逐位未动）")

    # 规则 1：每一段重复块的所有列都必须被标记
    for s in info["dup_segment_list"]:
        a, b = s["start_idx"], s["end_idx"]
        assert mask[a:b + 1].all(), f"重复块 [{a}..{b}] 未被完全标记"
    print(f"  ✓ 规则1: {info['dup_segments']} 段重复块（{info['dup_rows']} 行）全部被标记")

    # 规则 2：死列段的 OT 只在 OT 本身为 0，或所在行属重复块时才被标记
    dead = (raw.index >= "2016-12-05 09:00") & (raw.index <= "2016-12-07 18:00")
    dup_row = np.zeros(info["n_rows"], dtype=bool)
    for s in info["dup_segment_list"]:
        dup_row[s["start_idx"]:s["end_idx"] + 1] = True
    ot_is_zero = raw_v[:, TARGET_IDX] == 0.0
    assert (mask[:, TARGET_IDX][dead] == (ot_is_zero | dup_row)[dead]).all(), \
        "死列段 OT 的标记范围与预期不符"
    print(f"  ✓ 规则2: 死列段 {int(dead.sum())} 行中 OT 被标记 "
          f"{int(mask[dead, TARGET_IDX].sum())} 个，其余 "
          f"{int((~mask[dead, TARGET_IDX]).sum())} 个 OT 保留真实值")

    # 规则 3：每个失效段都必须被整段标记
    for col, d in info["zero_run_per_column"].items():
        j = FEATURES.index(col)
        for s in d["list"]:
            a, b = s["start_idx"], s["end_idx"]
            assert mask[a:b + 1, j].all(), f"{col} 的失效段 [{a}..{b}] 未被完全标记"
    print(f"  ✓ 规则3: {info['zero_run_segments']} 个失效段（合计 {info['zero_run_hours']}h）"
          f"全部被整段标记")

    untouched = ~changed.any(axis=1)
    assert int(untouched.sum()) == info["n_rows"] - info["rows_touched"]
    print(f"  ✓ 未被标记的 {int(untouched.sum())} 行数值逐位未变")

    noop = mask & ~changed
    if noop.any():
        print(f"  i 标记但插值后与原值相同 {int(noop.sum())} 个: "
              f"{dict(zip(FEATURES, noop.sum(axis=0).tolist()))}")
        print("    原因：该列异常窗口两侧的已知值本身就是 0，线性插值结果仍为 0。")


def check_zero_run_rule(info, cfg) -> None:
    """打印规则 3 的逐列明细：段数 + 总小时数 + 每段的起止时间。"""
    print(f"\n[规则 3 明细] 单列连续 0 失效段"
          f"（输入列 >= {cfg.zero_run_input_hours}h，OT >= {cfg.zero_run_target_hours}h）")
    for col, d in info["zero_run_per_column"].items():
        if d["segments"] == 0:
            print(f"  {col:>4s}  阈值>={d['threshold_hours']:>3d}h   未触发")
            continue
        print(f"  {col:>4s}  阈值>={d['threshold_hours']:>3d}h   "
              f"{d['segments']} 段 / 合计 {d['hours']}h")
        for s in d["list"]:
            print(f"        rows[{s['start_idx']:>5d}..{s['end_idx']:>5d}] "
                  f"{s['length']:>4d}h   {s['start_time']} -> {s['end_time']}")


def check_residual_zeros(clean, cfg) -> None:
    """断言清洗后不再存在超过阈值的连续精确 0 段。"""
    clean_v = clean.to_numpy(dtype="float64")
    runs_in = max_zero_run_lengths(clean_v[:, :TARGET_IDX])
    runs_ot = max_zero_run_lengths(clean_v[:, TARGET_IDX:TARGET_IDX + 1])[0]

    print("\n[清洗后残留零段自检] 各列最长连续精确 0")
    print("  输入列:", dict(zip(INPUT_FEATURES, runs_in)))
    print(f"  {'OT':>6s}: {runs_ot}h")

    assert max(runs_in) < cfg.zero_run_input_hours, \
        f"输入列仍存在连续 >= {cfg.zero_run_input_hours}h 的精确 0 段: {runs_in}"
    assert runs_ot < cfg.zero_run_target_hours, \
        f"OT 仍存在连续 >= {cfg.zero_run_target_hours}h 的精确 0 段: {runs_ot}h"
    print(f"  ✓ 全部输入列均不存在连续 >= {cfg.zero_run_input_hours}h 的精确 0 段"
          f"（最长 {max(runs_in)}h）")
    print(f"  ✓ OT 不存在连续 >= {cfg.zero_run_target_hours}h 的精确 0 段"
          f"（最长 {runs_ot}h）")


def show_block_demo(raw, clean) -> None:
    """打印一个 24 小时冻结块的前后对比，让插值效果可见。"""
    s = raw.loc["2016-07-30 20:00":"2016-08-01 04:00", "OT"]
    c = clean.loc["2016-07-30 20:00":"2016-08-01 04:00", "OT"]
    print("\n[插值效果示例] 2016-07-31 整日冻结块的 OT（原始 -> 清洗后）")
    for t in list(s.index)[:4] + list(s.index)[4:6] + list(s.index)[-3:]:
        flag = "  <== 块内" if "2016-07-31" in str(t) else ""
        print(f"  {t}  {s[t]:>8.3f} -> {c[t]:>8.3f}{flag}")


def main() -> int:
    cfg = load_data_config()          # 自动读取 configs/base.yaml
    print(f"config = {os.path.join('configs', 'base.yaml')}")
    print(f"csv    = {cfg.csv_path}")
    print(f"scaler = {cfg.scaler_path}   seq_len={cfg.seq_len}  pred_len={cfg.pred_len}")
    print(f"rules  = {cfg.cleaning_summary()}")

    bundle = prepare_data(cfg, save=True, verbose=True)
    info, stats = bundle["clean_info"], bundle["stats"]

    # ---------------- 规则 1/2 段落清单 ----------------
    print("\n[规则 1 明细] 被标记为重复的段落（整段置缺失后插值）")
    for s in info["dup_segment_list"]:
        print(f"  rows[{s['start_idx']:>5d}..{s['end_idx']:>5d}] len={s['length']:>3d}  "
              f"{s['start_time']} -> {s['end_time']}")
    print(f"  合计 {info['dup_rows']} 行 / {info['dup_segments']} 段")

    print(f"\n[规则 2 明细] 行级全 0 异常段（行内 >={cfg.zero_min_cols} 列为 0）")
    for s in info["zero_segment_list"]:
        print(f"  rows[{s['start_idx']:>5d}..{s['end_idx']:>5d}] len={s['length']:>3d}  "
              f"{s['start_time']} -> {s['end_time']}")
    print(f"  合计 {info['zero_rows']} 行（其中整行 7 列全 0 的 {info['all_zero_rows']} 行），"
          f"置缺失单元格 {info['zero_cells']} 个")

    check_zero_run_rule(info, cfg)

    # ---------------- 清洗前后对比与自检 ----------------
    after = info["after"]
    print(f"\n[清洗前后 {after['target']} 对比]")
    print(f"  行数: {info['n_rows']} -> {after['rows']}  (未删任何行，小时网格连续)")
    print(f"  {after['target']} min/max: {after['target_min']:.4f} / {after['target_max']:.4f}"
          f"  mean={after['target_mean']:.4f}  std={after['target_std']:.4f}")

    raw = load_raw(cfg)
    check_cleaning(raw, bundle["clean_frame"], info)
    check_residual_zeros(bundle["clean_frame"], cfg)
    show_block_demo(raw, bundle["clean_frame"])

    # ---------------- 切分与窗口 ----------------
    print(f"\n{BAR}\n[三个 split 的窗口数与被 3σ 裁剪的单元格]\n{BAR}")
    for name in ("train", "val", "test"):
        s = stats[name]
        print(f"  {name:>5s}: rows={s['rows']:>6d}  windows={s['windows']:>6d}  "
              f"clipped={s['cells_clipped']:>5d}  {s['start_time']} -> {s['end_time']}")
    print("  train 每列被裁剪数:", stats["train"]["clipped_per_col"])

    # ---------------- 取一个 batch ----------------
    loaders = make_dataloaders(cfg, batch_size=32, num_workers=0, bundle=bundle)
    xb, yb = next(iter(loaders["train"]))

    print(f"\n{BAR}\n[SMOKE TEST] 一个 batch\n{BAR}")
    print(f"  x.shape = {tuple(xb.shape)}   dtype = {xb.dtype}")
    print(f"  y.shape = {tuple(yb.shape)}   dtype = {yb.dtype}")
    print(f"  batch_size={xb.shape[0]}  seq_len={xb.shape[1]}  n_features={xb.shape[2]}"
          f"  pred_len={yb.shape[1]}")

    x1, y1 = bundle["datasets"]["train"][0]
    print(f"  单样本: x.shape={tuple(x1.shape)}  y.shape={tuple(y1.shape)}  "
          f"dtypes=({x1.dtype}, {y1.dtype})")

    # ---------------- 断言 ----------------
    assert tuple(xb.shape[1:]) == (cfg.seq_len, len(FEATURES)), xb.shape
    assert tuple(yb.shape[1:]) == (cfg.pred_len,), yb.shape
    assert xb.dtype == torch.float32 and yb.dtype == torch.float32
    assert torch.isfinite(xb).all() and torch.isfinite(yb).all()
    assert float(xb.max()) <= cfg.clip_sigma + 1e-6, "标准化后应落在 +3σ 内"
    assert float(xb.min()) >= -cfg.clip_sigma - 1e-6, "标准化后应落在 -3σ 内"
    assert all(torch.isfinite(b[0]).all() and torch.isfinite(b[1]).all()
               for b in (next(iter(loaders["val"])), next(iter(loaders["test"]))))

    # ---------------- 反归一化自检 ----------------
    scaler = load_scaler(cfg.scaler_path)
    y_raw = denormalize_ot(yb.numpy(), scaler)
    print("\n[反归一化自检] outputs/scaler.npz 已保存并可复用")
    print(f"  scaler['OT']: mean={scaler['mean'][TARGET_IDX]:.4f} "
          f"std={scaler['std'][TARGET_IDX]:.4f}  "
          f"zero_run={int(scaler['zero_run_clean'])}"
          f"({int(scaler['zero_run_input_hours'])}h/{int(scaler['zero_run_target_hours'])}h)")
    print(f"  y(归一化) 第 0 行前 4 个:   {np.round(yb.numpy()[0, :4], 4).tolist()}")
    print(f"  y(原始量纲) 第 0 行前 4 个: {np.round(y_raw[0, :4], 4).tolist()}")
    print(f"  本 batch 原始量纲范围: [{y_raw.min():.4f}, {y_raw.max():.4f}]")
    assert -60.0 < y_raw.mean() < 60.0, "反归一化结果不像 OT 的物理量纲"

    # ---------------- 逐时对齐自检 ----------------
    for name in ("train", "val", "test"):
        s = stats[name]
        assert s["windows"] == s["rows"] - cfg.seq_len - cfg.pred_len + 1, name
    assert stats["train"]["end_time"] < stats["val"]["start_time"] < stats["test"]["start_time"]
    assert stats["val"]["end_time"] < stats["test"]["start_time"]
    print("\n  ✓ 窗口数 = 行数 - 48 + 1；train/val/test 时间区间严格先后无重叠")

    print("\n全部断言通过 [OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
