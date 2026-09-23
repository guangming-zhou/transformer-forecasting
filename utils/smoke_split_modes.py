# -*- coding: utf-8 -*-
"""验证两种切分模式（time_sequential / season_stratified）都能正确加载。

运行：
    python -m utils.smoke_split_modes

核对的关键性质
1. 两种模式都能产出 (N, 24, 7) / (N, 24) 的滑窗
2. season 模式**不重拟合 scaler**：两种模式的 mean/std/low/high 必须逐位相同
3. season 模式的 test 与主切分的 test **完全相同**（test 没被动过）
4. 滑窗不跨区间边界：每个区间的首个窗口起点 = 区间起点，末个窗口终点 = 区间终点
5. 每个季节在 season_train / season_val 里都出现，且 val 约占该季的 25%
6. season_val 的 OT 分布偏移远小于主 val 的 -1.2024σ
"""

from __future__ import annotations

import os
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from utils.dataset import (  # noqa: E402
    FEATURES, SEASONS, TARGET_IDX, load_data_config, prepare_data,
)

BAR = "=" * 78


def ot_offset(bundle, split: str, scaler) -> float:
    """该 split 的 OT 均值相对 train-only scaler 的偏移（单位：train-σ）。"""
    frames = bundle["split_frames"][split]
    ot = np.concatenate([f["OT"].to_numpy(dtype="float64") for f in frames])
    return float((ot.mean() - scaler["mean"][TARGET_IDX]) / scaler["std"][TARGET_IDX])


def check_windows_align(bundle, name: str, scaled: np.ndarray) -> int:
    """核对窗口与区间边界严格对齐（证明滑窗没跨区间）。返回窗口数。"""
    ranges = bundle["ranges"][name]
    ds = bundle["datasets"][name]
    seq_len = bundle["config"].seq_len
    pred_len = bundle["config"].pred_len
    span = seq_len + pred_len

    total, offset = 0, 0
    for start, end in ranges:
        n_win = max(0, end - start - span + 1)
        if n_win:
            x_first = ds.x[offset].numpy()
            assert np.allclose(x_first, scaled[start:start + seq_len].astype("float32")), \
                f"{name} 区间 {(start, end)} 首个窗口起点未对齐区间起点"
            y_last = ds.y[offset + n_win - 1].numpy()
            assert np.allclose(y_last, scaled[end - pred_len:end, TARGET_IDX].astype("float32")), \
                f"{name} 区间 {(start, end)} 末个窗口终点未对齐区间终点"
        total += n_win
        offset += n_win

    assert total == len(ds), f"{name} 窗口数 {len(ds)} != 各区间可切窗口之和 {total}"
    return total


def check_window_stride() -> None:
    """验证 window_stride：1 = 与原行为逐位一致；k>1 时只降采样 train。"""
    print("\n[3/3] window_stride（滑窗降采样）")
    b_base = prepare_data(load_data_config(), save=False, verbose=False)
    b_one = prepare_data(load_data_config(window_stride=1), save=False, verbose=False)

    for name in ("train", "val", "test"):
        assert np.array_equal(b_base["datasets"][name].x.numpy(),
                              b_one["datasets"][name].x.numpy()), f"{name} x 不一致"
        assert np.array_equal(b_base["datasets"][name].y.numpy(),
                              b_one["datasets"][name].y.numpy()), f"{name} y 不一致"
    print("  ✓ window_stride=1 时 train/val/test 窗口与默认（不传该参数）逐位一致")

    full = b_base["stats"]["train"]["windows"]
    print(f"    {'stride':>7s}{'train_windows':>15s}{'ceil(N/k)':>11s}{'val':>8s}{'test':>8s}")
    for k in (1, 4, 8, 12):
        bk = prepare_data(load_data_config(window_stride=k), save=False, verbose=False)
        n_tr = bk["stats"]["train"]["windows"]
        expect = -(-full // k)                                   # ceil(N/k)
        assert n_tr == expect, f"stride={k}: train 窗口 {n_tr} != {expect}"
        for name in ("val", "test"):
            assert np.array_equal(bk["datasets"][name].x.numpy(),
                                  b_base["datasets"][name].x.numpy()), f"stride={k} 改动了 {name}"
        # 必须是原 train 窗口的等间隔子序列
        assert np.array_equal(bk["datasets"]["train"].x.numpy(),
                              b_base["datasets"]["train"].x.numpy()[::k])
        print(f"    {k:>7d}{n_tr:>15d}{expect:>11d}{bk['stats']['val']['windows']:>8d}"
              f"{bk['stats']['test']['windows']:>8d}")
    print("  ✓ stride>1 只降采样 train（ceil(N/k) 个窗口），val/test 逐位不变")


def main() -> int:
    cfg_seq = load_data_config()                                  # 默认 time_sequential
    cfg_sea = load_data_config(split_mode="season_stratified")

    print(BAR)
    print("[1/2] time_sequential（主切分）")
    print(BAR)
    b_seq = prepare_data(cfg_seq, save=False, verbose=True)

    print()
    print(BAR)
    print("[2/2] season_stratified（对照）")
    print(BAR)
    b_sea = prepare_data(cfg_sea, save=False, verbose=True)

    scaled = b_seq["clean_frame"].to_numpy(dtype="float64")
    # 复算 apply_scaler 的结果，用于逐窗口对齐核对
    sc = b_seq["scaler"]
    scaled = (np.clip(scaled, sc["low"], sc["high"]) - sc["mean"]) / sc["std"]

    print()
    print(BAR)
    print("核对")
    print(BAR)

    # ---- 1) 形状 ----
    for tag, b, cfg in (("time_sequential", b_seq, cfg_seq), ("season_stratified", b_sea, cfg_sea)):
        for name in ("train", "val", "test"):
            x, y = b["datasets"][name].x, b["datasets"][name].y
            assert x.ndim == 3 and x.shape[1:] == (cfg.seq_len, len(FEATURES)), (tag, name, x.shape)
            assert y.shape[1:] == (cfg.pred_len,), (tag, name, y.shape)
    print("  ✓ 两种模式都产出 (N, 24, 7) / (N, 24)")

    # ---- 2) scaler 未被重拟合 ----
    for key in ("mean", "std", "low", "high"):
        assert np.array_equal(b_seq["scaler"][key], b_sea["scaler"][key]), f"scaler[{key}] 被重拟合了"
    print("  ✓ 两种模式的 scaler mean/std/low/high 逐位相同 —— season 模式没有重拟合")

    # ---- 3) test 完全一致 ----
    assert b_seq["stats"]["test"]["rows"] == b_sea["stats"]["test"]["rows"]
    assert np.array_equal(b_seq["datasets"]["test"].x.numpy(), b_sea["datasets"]["test"].x.numpy())
    assert np.array_equal(b_seq["datasets"]["test"].y.numpy(), b_sea["datasets"]["test"].y.numpy())
    print(f"  ✓ test 完全相同（{b_seq['stats']['test']['rows']} 行 / "
          f"{b_seq['stats']['test']['windows']} 窗口），两种模式都没动 test")

    # ---- 4) 滑窗不跨区间 ----
    for tag, b in (("time_sequential", b_seq), ("season_stratified", b_sea)):
        for name in ("train", "val", "test"):
            n = check_windows_align(b, name, scaled)
        print(f"  ✓ {tag}: 滑窗与区间边界严格对齐"
              f"（train {b['stats']['train']['windows']} / val {b['stats']['val']['windows']} "
              f"/ test {b['stats']['test']['windows']}）")

    # ---- 5) 季节分层 ----
    print("\n  季节分层核对（season_stratified）")
    st = b_sea["stats"]
    print(f"    {'季节':<8s}{'train 行数':>11s}{'val 行数':>10s}{'val 占比':>10s}")
    for s in SEASONS:
        n_tr = st["train"]["seasons"][s]
        n_va = st["val"]["seasons"][s]
        tot = n_tr + n_va
        ratio = n_va / tot if tot else 0.0
        print(f"    {s:<8s}{n_tr:>11d}{n_va:>10d}{ratio:>9.1%}")
        assert tot > 0, f"季节 {s} 在 season 切分里没出现"
        assert n_tr > 0 and n_va > 0, f"季节 {s} 在 train/val 中有空侧"
        assert abs(ratio - (1 - cfg_sea.season_train_ratio)) <= 0.01 + 1.0 / tot, \
            f"季节 {s} 的 val 占比 {ratio:.3f} 偏离 {1 - cfg_sea.season_train_ratio:.2f}"
    print("  ✓ 4 个季节都出现在 season_train 与 season_val 中，且 val 约占该季 25%")

    # ---- 6) 分布对齐 ----
    off_seq_val = ot_offset(b_seq, "val", b_seq["scaler"])
    off_sea_tr = ot_offset(b_sea, "train", b_sea["scaler"])
    off_sea_va = ot_offset(b_sea, "val", b_sea["scaler"])
    print("\n  OT 均值相对 train-only scaler 的偏移（单位 train-σ）")
    print(f"    主切分 val（跨季节）        : {off_seq_val:+.4f}σ")
    print(f"    season_train                : {off_sea_tr:+.4f}σ")
    print(f"    season_val（季节对齐）      : {off_sea_va:+.4f}σ")
    assert abs(off_sea_va) < abs(off_seq_val), "season_val 的偏移没有变小，分层没起到对齐作用"
    print(f"  ✓ season_val 的偏移比主 val 小 "
          f"{abs(off_seq_val) / max(abs(off_sea_va), 1e-9):.1f} 倍")

    check_window_stride()

    print("\n全部断言通过 [OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
