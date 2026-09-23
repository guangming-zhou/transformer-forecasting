# 来源、贡献与第三方许可

本项目研究已有预测方法的表现，不主张发明 ETTh1 数据集、Transformer、LSTM、DLinear 或 ARIMA。
本项目的工作包括任务配置、训练与评测流程、基线集成、多 seed 实验、测试、可视化和 Demo。
这些工作使用第三方数据和软件，不能将整个仓库笼统称为“全部原创”。

## 数据

`data/ETTh1.csv` 来自 [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset)，
属于上游数据，采用 CC BY-ND 4.0，不适用本项目代码的 MIT 许可。
来源、原文件校验值、许可全文链接、免责声明和 Informer 引用见 [data/README.md](data/README.md)。
归一化、插值与滑窗生成的文件仅用于本地实验，不作为重新授权的数据集发布。

## DLinear 与分解组件

- 方法：Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu，
  [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504)，AAAI 2023。
- 参考实现：[cure-lab/LTSF-Linear/models/DLinear.py](https://github.com/cure-lab/LTSF-Linear/blob/main/models/DLinear.py)。
- 上游声明：Copyright 2022 DLinear Authors. All rights reserved.
- 上游许可：Apache License 2.0；[随附许可原文](LICENSES/LTSF-Linear-Apache-2.0.txt)。
- 本地对应：`eval/baseline_dlinear.py` 的移动平均、序列分解和双线性预测结构。
- 本地差异：配置接口、维度检查、目标通道选择、训练循环、指标与产物输出。

对与上游相应的组件保留上游归属和 Apache-2.0 条款，不以根目录 MIT 覆盖其权利。
当前没有可用于逐行追溯早期编写过程的 Git 历史，因此这里明确记录参考关系，
不声称其为完全独立、无参考的原创算法或代码。上游未为本项目背书。

## 其他方法与软件依赖

- Transformer：Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)，2017。
  本项目通过 PyTorch 的 TransformerEncoder 构建预测器。
- LSTM：Hochreiter and Schmidhuber, [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735)，1997。
  本项目使用 PyTorch LSTM。
- ARIMA：使用 statsmodels 实现；不是本项目原创统计方法。
- NumPy、Pandas、PyTorch、SciPy、statsmodels、Matplotlib、FastAPI、Streamlit 等是安装依赖，
  权利及许可归各自作者。虚拟环境和依赖源码不随本仓库发布。

## 实验结果、图表与权重

本地训练得到的指标、日志和权重是本项目实验产物，不是论文作者报告的数值或官方预训练模型。
官方切分表示采用固定数据边界，不表示本项目获得上游认证或完整复现了论文。
结果图由本地绘图脚本生成；数据来源仍应注明 ETTh1。预测曲线包含原始或清洗后序列内容，
不能仅因图是本地生成就断言其中第三方数据自动适用 MIT。
发布改动后的数据、序列导出或图表前应根据上游许可核对其内容；本项目许可不扩大这些权利。

## 项目许可范围

根目录 [LICENSE](LICENSE) 的 MIT 条款适用于本项目有权授权的原创代码与文档贡献。
第三方数据、参考或改编组件及安装依赖遵循各自许可。已有版权人姓名保持不变；
维护者应在发布前确认该姓名确为希望公开的署名，并确认未遗漏其他来源或共同作者。
