#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Head-to-head 汇总：统一模型指标并报告完整 seed 的配对置信区间。

这是本项目**唯一**的汇总入口 —— 之前 `summarize_baselines.py` / `summarize_residual.py`
各写一份聚合与判定逻辑，口径容易分叉；现在两者都是本脚本的薄封装。

评测对象永远是「同一份 test 滑窗 + 同一份磁盘 scaler + `utils.metrics.compute_metrics`」。

用法（24 -> 24 主设定）
    python eval/head_to_head.py --config configs/base.yaml \
        --ckpt "Transformer A=outputs/ckpt_A_ms" \
        --ckpt "Transformer+残差=outputs/ckpt_res" \
        --seed-json "LSTM=outputs/figs/metrics_lstm_seed{s}.json" \
        --seed-json "DLinear=outputs/figs/metrics_dlinear_seed{s}.json" \
        --fixed-json "ARIMA(1,1,0)=outputs/figs/metrics_arima.json" \
        --tag "ETTh1 24->24"

参数说明
    --ckpt      "名字=ckpt 目录"  目录下按 seed 取 seed{s}.pt，现场推理
    --seed-json "名字=模板"       模板里的 {s} 会被 seed 替换；用于 baselines 的 metrics_*.json
    --fixed-json"名字=文件"       无随机性的单次结果（如 ARIMA），不参与 2σ 判定
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from typing import Dict, List, Optional, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.common import (  # noqa: E402
    BAR, SEEDS, PROJECT_ROOT, agg, collect_ckpt_metrics, collect_seed_metrics,
    compare_pair_ci, compare_runs_to_fixed_ci, fmt, load_test_set, metrics_from_json,
    persistence_preds, rel_to_project,
    render_table, require_complete_seeds, resolve_device, table_columns, write_json,
)
from utils.dataset import load_data_config, load_scaler, prepare_data  # noqa: E402
from utils.metrics import compute_metrics  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="多预测器 head-to-head 汇总（完整 seed + 配对 CI）")
    p.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "base.yaml"))
    p.add_argument("--ckpt", action="append", default=[], metavar="名称=目录")
    p.add_argument("--seed-json", action="append", default=[], metavar="名称=模板")
    p.add_argument("--fixed-json", action="append", default=[], metavar="名称=文件")
    p.add_argument("--out", default=None, help="把结果写成 JSON（默认不写）")
    p.add_argument("--tag", default="", help="这次汇总的标签，写进 JSON 与标题")
    p.add_argument("--per-seed", action="store_true",
                   help="额外打印每个多 seed 预测器的逐 seed 明细（便于检查种子波动）")
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


def _split_pair(spec: str) -> Tuple[str, str]:
    name, sep, value = spec.partition("=")
    if not sep or not name.strip() or not value.strip():
        raise SystemExit(f"参数需为 名称=值 形式，收到 {spec!r}")
    return name.strip(), value.strip()


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    cfg = load_data_config(args.config)
    bundle = prepare_data(cfg, save=False, verbose=False)
    scaler = load_scaler(cfg.scaler_path)
    for key in ("mean", "std"):
        if not (scaler[key] == bundle["scaler"][key]).all():
            raise SystemExit(f"磁盘 scaler 的 {key} 与按 --config 重新拟合的不一致，先排查再评估")

    ts = load_test_set(cfg, scaler)
    x, gts = ts["x"], ts["gts"]
    print(BAR)
    print(f"Head-to-head  {args.tag or cfg.data_path}")
    print(BAR)
    print(f"  config   = {os.path.abspath(args.config)}")
    print(f"  test     = {x.shape[0]} 个滑窗 × {cfg.pred_len} 步  "
          f"({ts['target_times'][0, 0]} -> {ts['target_times'][-1, -1]})")
    print(f"  scaler   = {os.path.abspath(cfg.scaler_path)}")
    print(f"  seeds    = {list(SEEDS)}")
    print(BAR)

    # ---------------- 逐预测器收集 ----------------
    entries: List[Dict] = []          # 顺序即表格顺序
    raw: Dict[str, List[Dict]] = {}   # 名字 -> 每个 seed 的 metrics

    pers_preds = persistence_preds(ts["last_input_ot"], cfg.pred_len)
    pers_m = compute_metrics(pers_preds, gts)
    raw["persistence"] = [pers_m]
    entries.append({"name": "persistence", "kind": "deterministic", "metrics": [pers_m],
                    "seeds": []})

    for spec in args.fixed_json:
        name, path = _split_pair(spec)
        if not os.path.isfile(path):
            print(f"  i 跳过 {name}：{path} 不存在")
            continue
        m = metrics_from_json(path)
        raw[name] = [m]
        entries.append({"name": name, "kind": "deterministic", "metrics": [m], "seeds": []})

    for spec in args.seed_json:
        name, pattern = _split_pair(spec)
        seeds, runs = collect_seed_metrics(
            pattern, expected_seq_len=cfg.seq_len, expected_pred_len=cfg.pred_len,
            expected_split_mode=cfg.split_mode)
        if not runs:
            print(f"  i 跳过 {name}：没有任何 seed 文件匹配 {pattern}")
            continue
        try:
            require_complete_seeds(seeds, label=name)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        raw[name] = runs
        entries.append({"name": name, "kind": "multi_seed", "metrics": runs, "seeds": seeds})

    for spec in args.ckpt:
        name, ckpt_dir = _split_pair(spec)
        seeds, runs = collect_ckpt_metrics(ckpt_dir, x, gts, scaler, device=device)
        if not runs:
            print(f"  i 跳过 {name}：{ckpt_dir} 下没有 seed*.pt")
            continue
        try:
            require_complete_seeds(seeds, label=name)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        raw[name] = runs
        entries.append({"name": name, "kind": "multi_seed", "metrics": runs, "seeds": seeds})
        print(f"  [OK] {name}: {len(runs)} seed {seeds}")

    if not any(e["kind"] == "multi_seed" for e in entries):
        print("\n  警告：没有任何多 seed 预测器，2σ 判定无法进行。")

    # ---------------- 逐 seed 明细 ----------------
    if args.per_seed:
        print()
        print("逐 seed 明细（test MAE / RMSE，摄氏度）")
        for e in entries:
            if e["kind"] != "multi_seed":
                continue
            print(f"  {e['name']}:")
            for s, m in zip(e["seeds"], e["metrics"]):
                print(f"    seed {s:<5d} MAE={m['overall']['MAE']:.4f}  "
                      f"RMSE={m['overall']['RMSE']:.4f}")

    # ---------------- 表格 ----------------
    columns = table_columns(pers_m)
    rows = []
    for e in entries:
        runs = e["metrics"]
        if e["kind"] == "multi_seed" and len(runs) > 1:
            mae, rmse = agg(runs, ["overall", "MAE"]), agg(runs, ["overall", "RMSE"])
            hcols = {c: agg(runs, [c, "MAE"]) for c in columns}
            label = f"{e['name']} ({len(runs)} seed)"
        else:
            m = runs[0]
            mae = (m["overall"]["MAE"], 0.0)
            rmse = (m["overall"]["RMSE"], 0.0)
            hcols = {c: (m[c]["MAE"], 0.0) for c in columns}
            label = e["name"]
        rows.append({"name": label, "MAE": mae, "RMSE": rmse, "h": hcols,
                     "note": "(确定性)" if e["kind"] == "deterministic" else ""})

    print()
    print(f"汇总表（MAE / RMSE，摄氏度；多 seed 行为 mean ± std；分列为分 horizon MAE）")
    print(render_table(rows, columns))

    # ---------------- 结论句 ----------------
    ref_name = "persistence"
    ref_m = raw[ref_name][0]
    ref_mae = ref_m["overall"]["MAE"]
    print()
    print(BAR)
    print("结论句（固定 5 seed；相同 seed 配对差值的 95% Student-t 置信区间）")
    print(BAR)
    print(f"  参照基准 {ref_name}: MAE = {ref_mae:.4f} °C（确定性，无种子方差）")
    for e in entries:
        if e["name"] == ref_name:
            continue
        if e["kind"] == "deterministic":
            delta = e["metrics"][0]["overall"]["MAE"] - ref_mae
            print(f"  · {e['name']} vs {ref_name}: delta = {delta:+.4f} —— "
                  f"确定性算法，单次运行，**不参与 2σ 判定**")
        else:
            print(compare_runs_to_fixed_ci(
                f"{e['name']}({len(e['metrics'])} seed)", e["metrics"], ref_name, ref_mae))

    multi = [e for e in entries if e["kind"] == "multi_seed"]
    for i in range(len(multi)):
        for j in range(i + 1, len(multi)):
            print(compare_pair_ci(f"{multi[i]['name']}({len(multi[i]['metrics'])} seed)",
                                  multi[i]["metrics"],
                                  f"{multi[j]['name']}({len(multi[j]['metrics'])} seed)",
                                  multi[j]["metrics"]))

    print(f"\n  可靠性小结: persistence 无方差；确定性算法的 delta 仅供量级参考。"
          f"95% CI 包含 0 的对比不得表述为稳定方向；该区间只反映 seed 波动，"
          f"不替代时间块不确定性分析。")
    print(BAR)

    if args.out:
        write_json(args.out, {
            "tag": args.tag or os.path.basename(args.config),
            "config": rel_to_project(args.config),
            "split_mode": cfg.split_mode,
            "data_path": rel_to_project(cfg.data_path),
            "data_sha256": _sha256(cfg.data_path),
            "scaler_path": rel_to_project(cfg.scaler_path),
            "scaler_sha256": _sha256(cfg.scaler_path),
            "features": list(cfg.features),
            "target": cfg.target,
            "seq_len": cfg.seq_len,
            "pred_len": cfg.pred_len,
            "n_windows": int(x.shape[0]),
            "seeds": list(SEEDS),
            "horizons": columns,
            "predictors": [
                {"name": e["name"], "kind": e["kind"], "seeds": e["seeds"],
                 "metrics": e["metrics"]}
                for e in entries
            ],
            "summary": [
                {"name": row["name"],
                 "MAE": {"mean": row["MAE"][0], "std": row["MAE"][1]},
                 "RMSE": {"mean": row["RMSE"][0], "std": row["RMSE"][1]},
                 "horizon_MAE": {
                     key: {"mean": row["h"][key][0], "std": row["h"][key][1]}
                     for key in columns}}
                for row in rows
            ],
        })
        print(f"  已写出: {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
