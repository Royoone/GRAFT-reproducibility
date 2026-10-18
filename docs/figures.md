# Selected quantitative figures

`python run.py figures` generates the eighteen panels below. `--only` accepts one or more task IDs. Each entry generates only its listed panels, under `results/figures/<task>/`. Commands and required outputs are indexed in `figures/figure_tasks.json`.

| Task ID | Panels | Purpose |
|---|---|---|
| `figure1` | F1c, F1d | Measured genetic/chemical cross-context transfer |
| `figure1i` | F1i | Conditional-modulation component intervention |
| `figure2b` | F2b | Internal method comparison |
| `figure2` | F2d | Drug-level source performance versus prediction gain |
| `figure3` | F3b, F3c | External method comparison and ranks |
| `figure4` | F4b, F4c, F4f | Anchor adaptation ranking, gains and mechanism strata |
| `figure5` | F5b, F5k | Atlas map and PI3K–mTOR fingerprint |
| `figure6` | F6e | Context-dependent glucocorticoid ROS programme |
| `figure7` | F7f, F7g, F7h | iPSC genetic correspondence, MoA summary and fixed drug cases |
| `supplementary` | FS2a, FS2b | Genetic input budgets and retained state dimensions |

For example:

```bash
python run.py figures --only figure2b figure3
```

The scripts reproduce quantitative panels from the manuscript's fixed tables and arrays. The Atlas map uses the fixed fitted coordinates. Training, adaptation, representation probes, Atlas13 statistics and iPSC predictions retain their separate [computation entries](experiments.md). Recomputed experiment outputs do not automatically replace the fixed figure inputs.

F6e shows predicted signed ROS-gene-set scores, with 41 matched chemical connectivity groups. It is a paired comparison of A549 with the mean of four other LINCS contexts; the programme score is not a measured intracellular ROS concentration.

Each panel is exported as PDF, SVG and PNG. Full-page manuscript assembly is left to the author layout.
