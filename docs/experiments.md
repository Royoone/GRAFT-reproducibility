# Running the manuscript experiments

Run commands from the repository root after installing the dependencies and restoring the packaged data. `cuda:0` is recommended for training; use `cpu` for a small diagnostic or checkpoint inference. Seeds are **52, 229 and 564**. Use a fresh output directory when repeating a run.

## GRAFT training and benchmark inference

The root driver fixes the architecture, initialization and 200-step training recipe for each experiment:

```bash
python run.py train --experiment internal --target A375 --seed 52 --device cuda:0
python run.py train --experiment external --seed 52 --device cuda:0
python run.py train --experiment adaptation --target A375 --seed 52 --device cuda:0
python run.py train --experiment ipsc --seed 52 --device cuda:0
```

| Experiment | Training contexts | Purpose |
|---|---|---|
| `internal` | Four of A375, A549, HT29, MCF7 and PC3; exclude `--target` | Internal benchmark; also evaluates the trained fold |
| `external` | All five source contexts | External benchmark backbone |
| `adaptation` | Four source contexts; exclude `--target` | Internal adaptation backbone |
| `ipsc` | All five source contexts | iPSC and atlas backbone |

Repeat internal/adaptation training for each of the five targets and all three seeds. External/iPSC training needs one five-source fit per seed. Backbone training receives only the declared source tables; target genetic responses condition inference, and target query drug responses are read for scoring.

To recompute benchmark predictions with the **packaged manuscript checkpoints**:

```bash
python run.py benchmark --target A375 HepG2 --seed 52 --device cpu
# All ten contexts and three seeds:
python run.py benchmark --device cuda:0
```

Internal folds use the same **644 drug records**, fixed by the salted SHA-256 ordering of the 3,220 shared records. External matched-query counts are HepG2 **4,530**, MCF10A **1,484**, Jurkat **1,404**, K562_L1000 **96** and HAP1 **86**. These identities and their order are fixed; missing molecular encodings use the declared source-mean fallback in the comparison methods.

Training writes `model.pt`, `metadata.json` and `history.json` under `results/backbone_training/<experiment>/...`; benchmark inference writes predictions, per-drug scores and completion metadata under `results/benchmark_predictions/<target>/<seed>/`. Metadata records sources, seed, response scale, recipe, hashes and runtime information.

**Export new checkpoints before inference.** Training produces `model.pt`; the frozen loaders read `model.npz` and matching metadata. The export tool accepts a complete 200-step checkpoint, validates its tensors and training metadata, and refuses an existing output directory. For a new internal fit:

```bash
python run.py train --experiment internal --target A375 --seed 52 \
  --device cuda:0 --output results/retrained/training
python tools/export_checkpoint.py \
  --input results/retrained/training/internal/A375/cross_moment_residual/52/model.pt \
  --output results/retrained/benchmark/A375/52
python -m experiments.backbone.predict --target A375 --seed 52 \
  --checkpoints results/retrained/benchmark --output results/retrained/predictions
```

The exporter preserves the source/scale contract and records both file hashes. The final command reevaluates the exported model; it does not change the packaged checkpoints.

## Adaptation: features, learned adapters and fixed evaluation

Generate all three-seed features from the packaged backbones. This leaves the source tables and frozen weights unchanged:

```bash
for cell in A375 A549 HT29 MCF7 PC3 HepG2 MCF10A Jurkat K562_L1000 HAP1; do
  for seed in 52 229 564; do
    python -m experiments.adaptation.features --cell "$cell" --seed "$seed" \
      --output results/adaptation/features --device cuda:0
  done
done
```

For newly trained adaptation backbones, export each fold to `<root>/adaptation_backbone/<cell>/<seed>/` and the external backbone to `<root>/benchmark/external/<seed>/`, then supply `--checkpoint-root <root>` to feature extraction. Each export contains `model.npz` and `metadata.json`; use the corresponding `adaptation` and `external` training recipes above.

The following fits the main learned adapter and the ridge scientific control. Each adapter is learned across the three backbone seeds, so this command has no separate `--seed` argument:

```bash
for target in A375 A549 HT29 MCF7 PC3 external; do
  python -m experiments.adaptation.train --features results/adaptation/features \
    --output results/adaptation/adapters --target "$target" --device cuda:0
  python -m experiments.adaptation.train --features results/adaptation/features \
    --output results/adaptation/adapters --target "$target" --kind ridge --device cuda:0
done
for target in A375 A549 HT29 MCF7 PC3; do
  python -m experiments.adaptation.train --features results/adaptation/features \
    --output results/adaptation/adapters --target "$target" --kind kernel --device cuda:0
done
```

`ridge_remix` is the default main adapter. The UniPert kernel control is internal only. Adapter fitting and selection use the prescribed source contexts; held-out target anchors enter the subsequent adaptation evaluation.

Evaluate the fixed 30 anchor orderings, summarize the budgets and compute the displayed six-metric ranking together with the full eighteen-metric table:

```bash
python -m experiments.adaptation.evaluate_fixed \
  --features results/adaptation/features --adapters data/checkpoints/adaptation \
  --splits data/processed/adaptation/splits \
  --annotations data/annotations/compounds.csv \
  --out results/adaptation/fixed
python -m experiments.adaptation.summarize \
  --runs results/adaptation/fixed --out results/adaptation/summary
python -m experiments.adaptation.ranking \
  --summary results/adaptation/summary/primary_metric_summary.csv \
  --output results/adaptation/ranking
```

This evaluates the frozen adapters. To assess the newly fitted adapters, change `--adapters` to `results/adaptation/adapters` and use a separate `--out` directory. The primary budget is **10 anchors**; the evaluator retains the fixed query/anchor partitions and all budget calculations. Summarization and ranking require all ten contexts.

## Comparison methods

Use the method names below with the fixed comparison entry point:

```bash
python -m experiments.comparisons.run --cohort internal --method XPert \
  --target A375 --seed 52 --device cuda:0
python -m experiments.comparisons.run --cohort external --method PRnet \
  --target HepG2 --seed 52 --device cuda:0
```

| Cohort | Accepted method names |
|---|---|
| Internal, ten configurations | `DeepCE`, `CIGER-hinge`, `CIGER-RankNet`, `PRnet`, `TranSiGen`, `chemCPA`, `Morgan-256`, `Morgan-2048`, `UniPert-MLP`, `XPert` |
| External, eight configurations | `DeepCE`, `CIGER`, `TranSiGen`, `PRnet`, `chemCPA`, `Morgan-256`, `XPert`, `G2CP` |

Repeat each method over the relevant five contexts and three seeds. Internal G2CP-adapter additionally has the manuscript's three target-drug budgets:

```bash
for budget in D0 D20 D10pct; do
  python -m experiments.comparisons.run --cohort internal --method G2CP-adapter \
    --target A375 --seed 52 --budget "$budget" --device cuda:0
done
python -m experiments.comparisons.g2cp.run --target A375 --seed 52 --device cuda:0
python -m experiments.comparisons.hypermap.run --target A375 --seed 52 --device cuda:0
```

The separate G2CP command uses the author backbone with genetic pretraining followed by source-drug fine-tuning; its permitted target genetics enter pretraining. HyperMap produces D0 and D20 from the fixed source fit. D20 uses the frozen 20-anchor list; G2CP-adapter D10pct samples the prescribed 10% budget outside the sealed queries. See the [G2CP](../experiments/comparisons/g2cp/README.md) and [HyperMap](../experiments/comparisons/hypermap/README.md) READMEs for their exact protocols and dependencies. `--smoke` checks execution only and must not be treated as a manuscript result.

## Component and representation diagnostics

The [diagnostic entries](../experiments/diagnostics/README.md) recompute the adopted genetic-input/state interventions, six capacity controls, eighteen conditional-interface controls and frozen/ridge representation probes:

```bash
python -m experiments.diagnostics.evaluate --target A375 --seed 52
python -m experiments.diagnostics.train --configuration width32 --target A375 --seed 52
python -m experiments.diagnostics.interfaces --target A375 --seed 52 --output results/diagnostics/interfaces
python -m experiments.diagnostics.probes --cells A375 --seeds 52 --output results/probes
```

Capacity/interface controls retrain from allowed source responses. Representation ridge probes explicitly use non-query target drug responses as a scientific diagnostic; they are not a D0 prediction experiment.

## iPSC, atlas and figure inputs

Follow the [iPSC README](../experiments/ipsc/README.md) and [atlas README](../experiments/atlas/README.md) for those analyses. Recomputed training/evaluation results are written separately and **do not automatically replace frozen figure inputs**. Full retraining is a user-run workflow; the packaged checkpoints, source tables and verification records document the supplied manuscript results.
