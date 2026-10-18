# Data and resource layout

All paths are relative to this repository. `resources_manifest.json` indexes packaged files by size and SHA-256. `python run.py verify` checks the complete repository after the release resources have been extracted. Do not reorder gene or perturbation axes.

## Prepared source and external responses

`data/processed/sources/{A375,A549,HT29,MCF7,PC3}.npz` contains:

| Field | Meaning |
|---|---|
| `genes` | Fixed 976-gene order |
| `genetic_names`, `chemical_names` | Perturbation identities in row order |
| `genetic_embeddings`, `chemical_embeddings` | Frozen 256-dimensional UniPert identity representations |
| `genetic_native`, `chemical_native` | Signed responses on the prepared response scale |
| `weights_rest` | Original source drug-vs-rest gene weights, aligned with chemical responses |

There are 3,220 chemical identities shared by all five source contexts. The internal 644-query panel is obtained using the frozen SHA-256 ordering rule implemented in `experiments/backbone/data.py`; it is not resampled per run. The remaining source records are retained for the prescribed experiments.

`data/processed/external/` contains the aligned HepG2, MCF10A, Jurkat, K562_L1000 and HAP1 tables, including the fixed `query_ids`. Query counts are 4,530; 1,484; 1,404; 96; and 86, respectively. External genetic responses and drug truth remain separate modalities. These prepared tensors are the main training/evaluation entry point; this package does not automate downloading and preprocessing every original LINCS or external raw release.

`data/annotations/compounds.csv` supplies chemical names, structures and annotations used in the comparator and figure analyses.

`data/processed/diagnostics/representations.npz` preserves the fixed drug identity, target-mean protein/ontology, random and shuffled representations used by FS2f/g. Its manifest records axes, shapes, finite-row counts and checksum. The diagnostic code reconstructs GRAFT conditional features from the corresponding benchmark checkpoints.

## Frozen models and adaptation

| Directory | Resource |
|---|---|
| `data/checkpoints/benchmark/` | Five internal folds and one shared external source-five fit, each at seeds 52, 229 and 564 |
| `data/checkpoints/adaptation_backbone/` | Three fixed feature-extraction backbones for each internal context |
| `data/checkpoints/adaptation/` | Ridge, remix and kernel adaptation parameters used by the corresponding paper experiments |
| `data/model_bundle/` | Self-contained source reference, identity catalogue and three GRAFT weights for iPSC inference |
| `data/processed/adaptation/splits/` | Ten query partitions and 30 nested anchor orders per context |
| `data/processed/comparisons/anchors20.json` | Fixed 20-anchor lists used by the comparator experiments |

Weights are exported as non-pickle NPZ arrays. Metadata records the original checkpoint checksum, source contexts, seed and response scale. The training entry point writes its own `model.pt`, metadata and history to `results/`; it does not overwrite these resources.

## Atlas13 and iPSC

`data/processed/atlas/atlas13_entities.npz` has separate prediction and observed layers, a measurement mask and the exact cell/entity/gene axes. It contains 13 contexts, 7,755 entities and 976 genes. Only predictions enter the atlas embedding. The accompanying catalogue and Hallmark GMT define the fixed biological annotations. See `experiments/atlas/README.md`.

`data/processed/ipsc/graft_input_976.npz` contains the final CP10K-log1p NonTarget-difference effects and quality fields for 19 lines. Of 124 retained measured genetic conditions, 118 have a source reference and six are needed only for the genetic correspondence score. The analysis selects the specified eligible subsets; the input is not drug-labelled training data. Original raw-data links/checksums and the count-processing entry are in `experiments/ipsc/`.

## Selected figure source data and distribution

Each `figures/<group>/source_data/` contains only the inputs needed by the selected quantitative panels. Frozen numerical checks for the full scientific computations are retained separately in `data/processed/atlas/reference/` and `data/processed/diagnostics/reference/`. Tables, coordinates and case selections are kept fixed. Intermediate derived tables and new plots go into `results/`.

Large prepared experiment tensors and NPZ model arrays are distributed in `graft-reproduction-resources.tar.gz`, attached to the [v0.1.0 release](https://github.com/Royoone/GRAFT-reproducibility/releases/tag/v0.1.0). The archive contains 68 NPZ files and preserves their `data/` paths. Download and extract it in the repository root as shown in the [quick start](../README.md). `SHA256SUMS` in the same release gives the archive checksum, and `python run.py verify` checks the individual resource hashes against `resources_manifest.json`. The remaining fixed inputs, selected figure source data and documentation are tracked in Git.
