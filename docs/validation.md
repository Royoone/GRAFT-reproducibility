# Validation and scope

One complete 200-step A375/52 CPU training run was also exported through `tools/export_checkpoint.py`; its 644-query predictions were identical to the direct post-training evaluation.

The following checks were performed while preparing the portable project:

| Area | Check |
|---|---|
| Backbone training | Source references, response scaling, episode sampling, loss, initialization and short optimizer trajectories compared with the author implementation |
| Frozen GRAFT benchmarks | All 30 context/seed prediction settings recomputed from 18 packaged checkpoints; macro Top20 Pearson differences below 1×10⁻⁹ |
| Adaptation | Adapter algebra/gradients and source training checked; all ten contexts, three seeds and 30 fixed anchor lists reevaluated, with bootstrap and Holm statistics preserved |
| Comparators | 79 implementation checks covering initialization, forward, loss, gradients and optimizer updates against the author code; selected real-data training paths executed |
| G2CP and HyperMap | Two-stage/second-order training paths exercised on small real inputs; G2CP vectorized forward, gradients and BatchNorm state checked against its author backbone |
| Frozen diagnostics and capacity | All 68 A375/52 diagnostic readings matched the original CPU loop exactly. Six capacity configurations matched initialization, forward, gradients and short trajectories; one complete width32 fit matched original weights and predictions exactly |
| Interface controls | All eighteen controls matched initialization, forward, gradients and one AdamW update; all eighteen real-data diagnostic paths were executed. One complete attn_set CPU fit matched the original training function exactly |
| Representation probes | All sixteen adopted A375/52 probe rows recomputed; maximum Top20 Pearson difference from the frozen table was 4.59×10⁻⁹ and maximum MSE difference was 5.32×10⁻⁸ |
| iPSC | All 19 lines × 7,755 entities × 976 genes recomputed; maximum checked gene difference 9.54×10⁻⁷ and maximum RMS difference 3.83×10⁻⁷ relative to frozen GPU outputs |
| iPSC correspondence | All 235 correlations and the 191/158 retained/positive counts checked; genetic score recomputed from measured responses and fixed source directions |
| Atlas13 | Full input RMS and class gene means matched exactly; Hallmark scores differed by at most 2.98×10⁻⁷. A 310-entity real-data pipeline test included PCA, k-means and UMAP; full 100,815-point UMAP was not refitted |
| Figures | Selected quantitative panel commands executed; panels inspected; scientific values preserved |

These checks establish the portability of the supplied computations within floating-point tolerance. Full three-seed comparator training and every diagnostic retraining grid were not repeated during project preparation.

`python run.py figures` starts from frozen evaluated tables. Diagnostic computations and capacity/interface retraining are provided separately in `experiments/diagnostics/`; the complete five-context, three-seed retraining grids were not rerun during packaging. The Atlas13 reanalysis starts from the fixed 13-context predicted/observed tensor. Its quick figures retain the paper's fitted coordinates. Raw LINCS preprocessing and the complete upstream generation of every external atlas context are outside the automated entry points. The iPSC raw-count aggregation code is included separately.

Recomputed outputs are written under `results/`. They do not automatically replace frozen figure sources. Use the fixed splits, full query sets and complete training settings when evaluating a reproduction run. A smoke test checks executability and cannot be interpreted as a paper performance result.
