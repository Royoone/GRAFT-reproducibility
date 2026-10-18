# iPSC 分析复现

本目录用于正文 iPSC 计算：用 iPSC 遗传扰动响应进行 GRAFT-D0 预测，并复算药物预测与遗传响应之间的探索性对应。不存在 iPSC 药物训练或药物锚点校准。

从项目根目录运行：

```bash
python experiments/ipsc/predict.py
python experiments/ipsc/analyze_correspondence.py \
  --input data/processed/ipsc/graft_input_976.npz \
  --bundle data/model_bundle \
  --catalog data/processed/ipsc/record_catalog.csv
python figures/figure7/run.py
```

`predict.py` 读取本项目的 `data/model_bundle`，默认按三个种子预测全部 19 株、7,755 个化学实体。结果保存在 `results/ipsc`。可用 `--lines eipl_3` 做单株检查，`--device cuda:0` 指定 GPU。每条记录的上下文修正先按完整 InChIKey 对应到实体，再加实体源参考；最后对有符号基因预测求种子平均，再计算 RMS。

`data/processed/ipsc/graft_input_976.npz` 只保留最终使用的 `effects__lognorm_nt` 和质量信息。它包含 118 个源参考覆盖的模型靶基因，以及六个仅供遗传评分使用的实测靶基因，共 124 个条件身份。后六个不进入模型背景编码。15 株质量筛选后的模型输入为每株 114–118 个遗传扰动；固定十株关联分析使用另一套明确的细胞株集合，包含测量质量较低但保留分析的细胞株。模型输入要求至少 20 个靶向细胞、两个 guides 和 50 个同株对照；遗传评分的靶向细胞阈值为 10。

`analyze_correspondence.py` 使用固定的 235 个实体和十株，重算 Spearman 相关、精确置换概率、191 个保留实体，以及八类 MoA 的共同可评价细胞株统计。原分析只对非正相关实体应用配对和敲低支持排除条件，保留 158 个正相关和 33 个非正相关实体；这是基于结果的保留集合。完整 235 个实体及其标记始终保留，不能把这个比例解释为独立药物实验准确率。传入上述三个资源参数时，还会从实测遗传响应和源药物 Top20 方向重算公式 (10)。

## 从原始 RNA 计数开始

原始计数不进入模型。公开数据来源和校验信息见 `raw_data_manifest.json`。RNA 计数约 4.2 GB，未放入 Git；将计数和元数据恢复到 `data/raw/ipsc/raw/`，再执行：

```bash
c++ -O3 -std=c++17 experiments/ipsc/preprocess/stream_counts.cpp -lz -o /tmp/graft_stream_counts
/tmp/graft_stream_counts \
  data/raw/ipsc/raw/TargetedScreen_RNA-UMI-Counts.csv.gz \
  experiments/ipsc/preprocess/wanted_gene_symbols.txt \
  data/raw/ipsc/sparse_rows
python experiments/ipsc/preprocess/aggregate_counts.py --work data/raw/ipsc
/tmp/graft_stream_counts \
  data/raw/ipsc/raw/TargetedScreen_RNA-UMI-Counts.csv.gz \
  experiments/ipsc/preprocess/wanted_alias_symbols.txt \
  data/raw/ipsc/sparse_alias_rows
python experiments/ipsc/preprocess/aggregate_alias_targets.py --work data/raw/ipsc
python experiments/ipsc/prepare_input.py \
  --input data/raw/ipsc/processed/graft_input_976.npz \
  --reference data/model_bundle/reference.npz \
  --output data/processed/ipsc/graft_input_976.npz
```

计数扫描必须覆盖完整转录组，以获得每个细胞的总 UMI；重复基因符号先在计数空间合并，再计算 CP10K、log1p 和同株 NonTarget 差值。未分配 guide 的细胞不作为 NonTarget。第二次扫描使用四个经核实的现行转录本符号，`aggregate_alias_targets.py` 将其对应到原始扰动名称，仅补全敲低质量核验表 `clone_target_qc_with_verified_aliases.csv`，不改变 976 基因模型输入。`prepare_input.py` 的有效性使用细胞数、guide 数和 NonTarget 数，不读取靶基因敲低幅度；235 个实体的固定配对与敲低支持筛选标记保存在图件源表中。`preprocess` 仅生成最终 CP10K-log1p、同株 NonTarget 差值，以及相应的质量信息和细胞抽样标准误；模型和关联分析读取 `effects__lognorm_nt`。229 个细胞的小型稀疏输入核验覆盖重复基因、零表达细胞、guide 与对照数量阈值；最终响应、SEM、质量字段和压缩后的模型输入均与原最终计算逐值一致，见 `preprocess/verification.json`。作者公布的调整效应和 q 值来自 Figshare 26819743，与模型输入的计数推导响应分开。

已完成全部 19 株的 CPU 复算。相对冻结的 GPU 结果，四个案例的最大基因误差为 9.54×10⁻⁷，全部药物 RMS 的最大误差为 3.83×10⁻⁷；235 个实体的相关性和正相关数量不变。完整记录见 `verification.json`。
