# GRAFT manuscript reproduction

Code and fixed inputs for reproducing the GRAFT manuscript computations and selected key quantitative panels. For applying GRAFT to a new context, use the separate [GRAFT application project](https://github.com/Royoone/GRAFT).

## Quick start

Use Python 3.12. Clone the repository and download the prepared data and model resources from the [v0.1.0 release](https://github.com/Royoone/GRAFT-reproducibility/releases/tag/v0.1.0):

```bash
git clone https://github.com/Royoone/GRAFT-reproducibility.git
cd GRAFT-reproducibility
curl -fL https://github.com/Royoone/GRAFT-reproducibility/releases/download/v0.1.0/graft-reproduction-resources.tar.gz -o graft-reproduction-resources.tar.gz
tar -xzf graft-reproduction-resources.tar.gz
rm graft-reproduction-resources.tar.gz
```

The resource archive contains prepared experiment tensors and NPZ model arrays at their expected `data/` paths. The release also provides `SHA256SUMS`; the verification command below checks each packaged file. Install the environment and run commands from the repository directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py verify
python run.py figures
```

Figure outputs are written to `results/figures/`. To run one group:

```bash
python run.py figures --only figure2b figure3
```

The figure entry contains eighteen key quantitative panels, covering cross-context response differences, benchmarks, adaptation, atlas biology, iPSC and genetic-input diagnostics. They are regenerated from fixed manuscript arrays/tables. See [the panel map](docs/figures.md).

## Recompute experiments

Recompute one GRAFT benchmark fold using its packaged checkpoint:

```bash
python run.py benchmark --target A375 --seed 52
```

Train the source-only internal fold using the prescribed 200-step recipe:

```bash
python run.py train --experiment internal --target A375 --seed 52 --device cuda:0
```

The training, anchor adaptation, comparator, Atlas13 and iPSC commands are described in [experiments](docs/experiments.md). Each experiment has fixed data splits and settings. Training never runs as a side effect of drawing figures.

## Contents

```text
experiments/       Backbone, adaptation, comparators, diagnostics, atlas and iPSC analyses
figures/           Selected key quantitative panel code and source data
data/              Prepared responses, annotations and frozen model resources
docs/              Experiment, panel, data and validation notes
tools/             Resource checks
tests/             Axis, split and scoring contract checks
run.py             Short entry point
resources_manifest.json   File sizes and SHA-256 checksums
```

## Validation

The project has been checked against the author implementation at the level of model operations, training steps, fixed checkpoint predictions, anchor evaluation and figure calculations. All 19 iPSC lines were recomputed. The selected source-data plots were executed and visually inspected. Full retraining of every comparator and every diagnostic experiment was not repeated while preparing this package. See [validation and scope](docs/validation.md).

For the lightweight contract checks:

```bash
pip install -r requirements-test.txt
python -m pytest -q
```

GPU training is recommended for the complete experiments. Device and library differences can affect floating-point reductions and nearly tied ranks. Keep the supplied row/gene order, query IDs, seeds and evaluation conventions.

Third-party code notices and resource terms are summarized in [THIRD_PARTY.md](THIRD_PARTY.md).

## Versioned release

The manuscript-associated reproduction code is tagged [v0.1.0](https://github.com/Royoone/GRAFT-reproducibility/releases/tag/v0.1.0). The resource archive attached to that release complements the files tracked in Git. See [data and resources](docs/data.md) for provenance and layout.
