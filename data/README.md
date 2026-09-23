# ETTh1 数据来源与许可

本目录的 `ETTh1.csv` 来自 Haoyi Zhou 等作者维护的
[ETDataset](https://github.com/zhouhaoyi/ETDataset)。上游介绍说明，数据由研究团队与
Beijing Guowang Fuda Science & Technology Development Company 合作采集。
本项目使用公开的小时级 ETTh1 子集，不主张数据所有权。

- 上游文件：[ETT-small/ETTh1.csv](https://github.com/zhouhaoyi/ETDataset/blob/main/ETT-small/ETTh1.csv)
- 许可：[CC BY-ND 4.0](https://creativecommons.org/licenses/by-nd/4.0/)
- 上游许可全文及免责声明：[ETDataset/LICENSE](https://github.com/zhouhaoyi/ETDataset/blob/main/LICENSE)
- 文件大小：2,589,657 字节
- 数据行：17,420（不含表头）
- 列：`date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT`
- SHA256：`f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`

随附文件未修改；清洗、归一化和切窗由代码在运行时进行，不覆盖该 CSV。
本项目根目录的 MIT 许可证不覆盖此数据。数据按原样提供，无担保。
再分发应保留上游署名、来源链接、许可及免责声明。CC BY-ND 4.0 不授权分享演绎材料，
因此不要把修改后的数据副本作为本项目的可自由分发数据发布。上游未为本项目提供背书。

## 引用

上游要求使用数据时引用 Informer（AAAI 2021）：

```bibtex
@inproceedings{haoyietal-informer-2021,
  author = {Haoyi Zhou and Shanghang Zhang and Jieqi Peng and Shuai Zhang and
            Jianxin Li and Hui Xiong and Wancai Zhang},
  title = {Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume = {35},
  number = {12},
  pages = {11106--11115},
  year = {2021}
}
```

## 完整性检查（PowerShell）

在项目根目录运行：

```powershell
Get-FileHash data/ETTh1.csv -Algorithm SHA256
```

输出应与上述 SHA256 一致。若需重新获取文件，请使用上游文件链接并重新核对哈希。
