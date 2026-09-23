#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""串行多变体 / 多 seed 训练驱动（严格串行，不并行）。

为什么需要它
    本项目所有结论都建立在「同一协议、同一 seed 集合、严格串行」之上：
    并行跑训练会让不同 run 争抢 CPU 线程，逐 epoch 的耗时与显存/内存峰值不可比，
    也会让一次运行的时间戳日志失去意义。这个脚本把「变体 × seed」展开成一串
    顺序执行的 `train/train.py` 调用，并把每次调用的完整 stdout 落盘。

用法
    python scripts/sweep.py \
        --config configs/lstf96.yaml \
        --out-root outputs/_runs/sweep96 \
        --seeds 42 0 1 2 9999 \
        --global-override output.scaler_path=outputs/scaler96.npz \
        --run "lstf96|model.residual=false" \
        --run "lstf96_res|model.residual=true"

    # 每个变体产出 <out-root>/<name>/{ckpt,logs,stdout.log}
    # --run 的格式是 "输出子目录名|k=v,k=v"；不带 | 时视为无覆盖。
    # --global-override 作用于所有变体（可重复给，也可一次给多个）。

退出码
    0  全部成功
    1  有任意一个 (变体, seed) 失败；失败的组合与已完成的组合都会打印出来
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "train", "train.py")
BAR = "=" * 88


def status_marker(ok: bool) -> str:
    """返回仅含 ASCII 的状态标记，兼容 Windows GBK 控制台。"""
    return "[OK]" if ok else "[FAIL]"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="串行多变体多 seed 训练驱动")
    p.add_argument("--config", required=True, help="基础 yaml 配置")
    p.add_argument("--out-root", required=True,
                   help="输出根目录，每个变体落在 <out-root>/<name>/ 下")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 9999])
    p.add_argument("--run", action="append", default=[], metavar="NAME|k=v,k=v",
                   help="一个训练变体，可重复给")
    p.add_argument("--global-override", action="extend", nargs="*", default=[],
                   metavar="KEY=VALUE", help="作用于所有变体的点路径覆盖")
    p.add_argument("--script", default=TRAIN_SCRIPT,
                   help="被驱动的脚本（默认 train/train.py；baseline 用 eval/baseline_*.py）")
    p.add_argument("--out-style", choices=("ckpt-log", "out"), default="ckpt-log",
                   help="ckpt-log: 传 output.ckpt_dir/output.log_dir（train.py）；"
                        "out: 传 --out <变体目录>（baseline 脚本）")
    p.add_argument("--flat-out", action="store_true",
                   help="不建 <out-root>/<name>/ 子目录，直接把产物写进 out-root"
                        "（baseline 指标要与既有 outputs/figs 放一起时用）")
    p.add_argument("--stdout-dir", default=None,
                   help="stdout.log 的存放目录（默认放在变体目录里）")
    p.add_argument("--dry-run", action="store_true", help="只打印命令，不真的训练")
    return p.parse_args(argv)


def parse_run(spec: str) -> Tuple[str, List[str]]:
    """把 "NAME|k=v,k=v" 解析成 (name, [k=v, ...])。"""
    name, _, body = spec.partition("|")
    name = name.strip()
    if not name:
        raise SystemExit(f"--run 缺少名字: {spec!r}")
    items = [kv.strip() for kv in body.split(",") if kv.strip()] if body else []
    for kv in items:
        if "=" not in kv:
            raise SystemExit(f"--run {name}: {kv!r} 不是 KEY=VALUE 形式")
    return name, items


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    runs = [parse_run(s) for s in args.run]
    if not runs:
        raise SystemExit("至少要给一个 --run")

    out_root = args.out_root if os.path.isabs(args.out_root) \
        else os.path.join(PROJECT_ROOT, args.out_root)
    config = args.config if os.path.isabs(args.config) \
        else os.path.join(PROJECT_ROOT, args.config)

    plan: List[Tuple[str, int, str, List[str], List[str]]] = []
    for name, items in runs:
        var_dir = out_root if args.flat_out else os.path.join(out_root, name)
        stdout_dir = args.stdout_dir or var_dir
        os.makedirs(var_dir, exist_ok=True)
        os.makedirs(stdout_dir, exist_ok=True)
        overrides = list(args.global_override) + items
        if args.out_style == "ckpt-log":
            if args.flat_out:
                raise SystemExit("--flat-out 只适用于 --out-style out")
            ckpt_dir = os.path.join(var_dir, "ckpt")
            log_dir = os.path.join(var_dir, "logs")
            os.makedirs(ckpt_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)
            extra_args = [
                "--override",
                f"output.ckpt_dir={os.path.relpath(ckpt_dir, PROJECT_ROOT).replace(os.sep, '/')}",
                f"output.log_dir={os.path.relpath(log_dir, PROJECT_ROOT).replace(os.sep, '/')}",
            ]
        else:                                     # baseline 脚本：只认一个 --out 目录
            if overrides:
                raise SystemExit("--out-style out 的脚本不支持 --override / 变体键值对，"
                                 "请改用 --script train/train.py + --out-style ckpt-log")
            extra_args = ["--out", os.path.relpath(var_dir, PROJECT_ROOT).replace(os.sep, "/")]

        for seed in args.seeds:
            plan.append((name, seed, os.path.join(stdout_dir, f"stdout_{name}.log"),
                         overrides, extra_args))

    if not os.path.isabs(args.script):
        script = os.path.join(PROJECT_ROOT, args.script)
    else:
        script = args.script
    if not os.path.isfile(script):
        raise SystemExit(f"找不到被驱动的脚本: {script}")

    print(BAR)
    print(f"串行 sweep：{len(runs)} 个变体 × {len(args.seeds)} 个 seed = {len(plan)} 次训练")
    print(f"  script   = {script}")
    print(f"  config   = {config}")
    print(f"  out-root = {out_root}   flat={args.flat_out}")
    print(f"  out-style= {args.out_style}")
    print(f"  seeds    = {args.seeds}")
    for name, items in runs:
        print(f"  variant  {name:<18s} {items or '(无变体覆盖)'}")
    print(BAR)
    sys.stdout.flush()

    failed: List[str] = []
    done: List[str] = []
    t_all = time.time()
    for i, (name, seed, stdout_path, overrides, extra_args) in enumerate(plan, 1):
        cmd = [sys.executable, script, "--config", config, "--seed", str(seed)]
        if overrides:
            cmd += ["--override", *overrides]
        cmd += extra_args
        tag = f"{name}/seed{seed}"
        print(f"\n[{i}/{len(plan)}] {tag}  -> {stdout_path}")
        print("  " + " ".join(cmd))
        sys.stdout.flush()
        if args.dry_run:
            continue

        t0 = time.time()
        # stdout/stderr 直接写文件：既留全量日志，也避免管道被提前关闭产生孤儿进程。
        # PYTHONUNBUFFERED 让日志可以实时 tail，而不是等 8 KB 缓冲写满。
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        with open(stdout_path, "a", encoding="utf-8") as fh:
            fh.write(f"\n\n{'#' * 78}\n# {tag}  {' '.join(cmd)}\n{'#' * 78}\n")
            fh.flush()
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=fh,
                                  stderr=subprocess.STDOUT, check=False, env=env)
        dt = time.time() - t0
        if proc.returncode == 0:
            done.append(tag)
            print(f"  {status_marker(True)} 完成，用时 {dt:.1f}s")
        else:
            failed.append(tag)
            print(f"  {status_marker(False)} 失败，returncode={proc.returncode}，"
                  f"用时 {dt:.1f}s（见 {stdout_path}）")
        sys.stdout.flush()

    print()
    print(BAR)
    print(f"sweep 结束：成功 {len(done)} / 失败 {len(failed)} / 总计 {len(plan)}；"
          f"总用时 {(time.time() - t_all) / 60:.1f} 分钟")
    if failed:
        print("失败组合：" + ", ".join(failed))
    print(BAR)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
