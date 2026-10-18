# Atlas13 scientific analysis

This entry point recomputes the manuscript Atlas analysis from the final frozen prediction and observation tensors. The prediction layer contains one predicted 976-gene response for every chemical–context pair. Measured responses remain separate and are used only in explicitly measured comparisons.

```sh
python -m experiments.atlas \
  --input data/processed/atlas/atlas13_entities.npz \
  --catalog data/processed/atlas/chemical_catalog.csv \
  --hallmark data/processed/atlas/hallmark.gmt \
  --output results/atlas
```

`record_catalog.csv` and `coverage.csv` are read beside the chemical catalogue. They provide the original record/entity census. Their paths can be set with `--record-catalog` and `--coverage`.

The input archive requires `prediction` (or its identical `response` alias), `observed`, `observed_mask`, `chemical_keys`, `genes` and `cells`. Shapes are 13 × 7,755 × 976 for the response layers and 13 × 7,755 for the mask. Entity order must equal `chemical_catalog.csv`. Context order is A375, A549, HT29, MCF7, PC3, HepG2, MCF10A, Jurkat, K562_L1000, HAP1, RPE1, neurons and Luhmes. The exact Hallmark GMT supplied with the author analysis is included.

The analysis uses the fixed paper parameters: row-centred L2 response patterns; top 30 within-context cosine neighbours; randomized PCA50; MiniBatchKMeans with 8 clusters, 10 initializations and batch size 4,096; UMAP with 200 neighbours, minimum distance 0.50 and 200 epochs. The random seed is 20260926. MoA labels annotate the map after fitting and are not embedding inputs. Hallmarks retain at least 5 assayed genes; directional ORA uses the 976-gene background and 50 strongest up/down genes. Bootstrap counts, class overlap handling and multiplicity families follow the original scripts.

Outputs include the prediction/observation archive, response/programme metrics, observed and predicted class summaries, ORA/BH tables, top 30 neighbour indices/enrichment, PCA50, UMAP coordinates, eight-community summaries, disjoint class correlations, PI3K–mTOR fingerprint confidence intervals and 976 exported gene means. All outputs are written to the requested directory. These outputs can be checked against the frozen figure-source tables; the normal quick figure tasks read those frozen tables separately.

Dependencies are NumPy, pandas, SciPy, scikit-learn, PyTorch and umap-learn. `--device auto` selects CUDA if available; `--device cpu` uses the same tensor matmul/topk algorithm on CPU. The original GPU analysis disabled TF32. Hardware/library versions can affect floating-point ties, PCA/UMAP coordinates and community label numbering; the fixed statistical rules and parameters remain the same. Full 100,815-point UMAP fitting is computationally substantial.

Frozen scientific-kernel verification, without fitting UMAP:

```sh
python -m experiments.atlas.validate \
  --reference data/processed/atlas/reference \
  --output results/atlas_validation
```

A small operational test uses real profiles selected by fixed catalogue order, class labels and observation availability. It keeps all 13 contexts, all 976 genes and the same PCA50/k8/UMAP200/.50 parameters. Its estimates are only a software test:

```sh
python -m experiments.atlas.smoke \
  --input data/processed/atlas/atlas13_entities.npz \
  --catalog data/processed/atlas/chemical_catalog.csv \
  --hallmark data/processed/atlas/hallmark.gmt \
  --output results/atlas_smoke
```

The full 100,815-point analysis has not been rerun during repository packaging. Validation records identify exactly which checks were executed. During packaging, the complete input tensor was checked against the frozen RMS, 16 class gene profiles and 49 predicted Hallmark score arrays. RMS/class profiles match exactly; all programme scores agree within 3×10⁻⁷. The complete scientific pipeline also passed the real 310-chemical smoke test, including UMAP and extended statistics. Model inference that created the frozen tensor belongs to the backbone/adaptation experiment branches; this entry point begins with that adopted tensor.

To repeat the full-input numerical check without fitting the 100,815-point embedding:

```sh
python -m experiments.atlas.validate \
  --reference data/processed/atlas/reference \
  --input data/processed/atlas/atlas13_entities.npz \
  --catalog data/processed/atlas/chemical_catalog.csv \
  --hallmark data/processed/atlas/hallmark.gmt \
  --output results/atlas_validation
```
