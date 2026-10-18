# HyperMap 比较配置

本入口对应正文 Figure 2b 的 HyperMap D0 与 D20，采用作者网络在有符号 LINCS 响应、零对照向量和药物 one-hot 身份输入上的任务适配。它们来自同一次源背景拟合；D20 再使用固定的 20 个目标药物锚点。

```bash
python experiments/comparisons/hypermap/run.py --target A375 --seed 52 --device cuda:0
python experiments/comparisons/hypermap/run.py --target A375 --seed 52 --smoke
```

默认读取本项目 `data/processed/sources/`，结果保存在 `results/comparisons/HyperMap/<target>/<seed>/D0` 和 `D20`。

`fixed_protocol.json` 仅记录五目标 × 三种子的最终源端选择结果、查询及锚点身份。拟合直接使用固定的内层学习率和 epoch 数，在全部允许源药物上重新初始化并训练；本入口不重新进行超参数搜索。目标药物响应在拟合完成后才读取，用于锚点适配和最终评分。

网络文件 `author_model.py` 与核验过的作者源码逐字一致，MIT 许可见 `LICENSE`。训练保留 `higher` 的完整二阶 MAML、四任务、每任务五批、每批五次内层更新以及原损失。D20 保留三次全网络梯度更新，通过仅依赖锚点损失的回溯决定步长。

作者网络的 softmax 跨批内样本计算，故推理批次会影响数值。入口保留固定查询顺序、每批最多 512 个样本，并对单个尾样本重复后裁回。不能任意改成逐个药物推理。

| 依赖 | 用途 |
|---|---|
| PyTorch | 作者网络及优化器 |
| higher | 完整二阶 MAML |
| NumPy、SciPy | 数据与统一评分 |

`--smoke` 将任务数、批次和 epoch 缩小，仅验证 CPU forward、二阶元梯度、D0/D20 及锚点回溯的可执行性，不输出论文结果。实际验证记录见 `verification.json`。完整训练未在项目整理期间重跑。
