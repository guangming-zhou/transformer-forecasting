# -*- coding: utf-8 -*-
"""评估与基线脚本包。

`common.py` 放共用组件（test 集构造、ckpt 推理、多 seed 聚合、2σ 判定、表格渲染），
其余脚本只管各自的 CLI 与产物。所有脚本都支持直接执行，例如
`python eval/head_to_head.py --help`。
"""
