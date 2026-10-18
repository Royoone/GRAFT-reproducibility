# G2CP 比较配置

本目录生成正文 Figure 2b 中的 **G2CP（D0）**：采用作者 GEARS–UniPert 骨干、允许训练数据构建的共表达图，以及遗传预训练后进行源药物微调的两阶段训练。源码许可见 `LICENSE`。

```bash
python experiments/comparisons/g2cp/run.py --target A375 --seed 52 --device cuda:0
python experiments/comparisons/g2cp/verify_forward.py
python experiments/comparisons/g2cp/run.py --target A375 --seed 52 --smoke
```

默认读取 `data/processed/sources/{A375,A549,HT29,MCF7,PC3}.npz`，输出到 `results/comparisons/G2CP/<target>/<seed>/`。每个文件的八个约定字段由项目数据准备步骤提供。遍历五个目标与种子 52、229、564 即得到采用的 15 次运行。

- 第一阶段合池允许源背景和目标背景的遗传响应；验证只取源遗传条件。共表达图仅从遗传训练条件和源药物训练条件构建。
- 第二阶段仅使用源药物响应，并以源药物验证 MSE 选择检查点。目标查询药物不用于图构建、拟合或选择。
- 两阶段分别运行 20/10 个 epoch；微调从最后一个遗传预训练 epoch 开始。保留原 Adam、逐 epoch 学习率减半、四次误差与符号项、梯度值裁剪、BatchNorm 和批处理逻辑。
- `author/model.py` 为未改动的作者骨干。`author/utils.py` 只抽取原工具模块的三个必要函数，避免引入不使用的下载与统计依赖。`fast_forward.py` 是采用运行使用的代数等价向量化实现；独立检查比较作者 forward、参数梯度和 BatchNorm 状态，见 `verification.json`。
- `--smoke` 仅在小规模真实源/目标遗传条件及源药物上执行两阶段各一批，用于检查代码路径；它明确标记为诊断，不是论文评估结果。

| 依赖 | 用途 |
|---|---|
| PyTorch | 网络和训练 |
| torch-geometric | 作者骨干 SGConv |
| networkx | 作者共表达图的边顺序与权重 |
| NumPy、pandas、SciPy | 数据、图和统一评分 |

当前 CPU 小步验证使用 torch-geometric 2.8.1。项目构建时未重跑完整 15 次训练。

## 同图 G2CP-adapter 的药物预算

同图另有 G2CP-adapter，它是无共表达图、将允许响应合池后从头训练的独立比较配置，使用上级入口：

```bash
python -m experiments.comparisons.run --cohort internal --method G2CP-adapter --target A375 --seed 52 --budget D0
python -m experiments.comparisons.run --cohort internal --method G2CP-adapter --target A375 --seed 52 --budget D20
python -m experiments.comparisons.run --cohort internal --method G2CP-adapter --target A375 --seed 52 --budget D10pct
```

D20 读取 `data/processed/comparisons/anchors20.json`，与 HyperMap 使用同一份固定锚点。D10pct 严格沿用原采样函数：预算为目标全部药物条件数的 10% 四舍五入，从排除 644 个查询的池中按固定种子无放回抽取。两个预算不构成本目录作者骨干的递增适配系列。`budget_verification.json` 检查了全部 15 份列表；添加 `--smoke` 时仅验证真实允许样本的 forward、MSE 和一次更新，不改动完整训练函数的数据采样。
