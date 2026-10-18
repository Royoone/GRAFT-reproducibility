# F4b, F4c, F4f

```sh
python figures/figure4/code/render.py --output results/figures/figure4
```

Fixed five-cell adaptation ranking and anchor-budget gains. The displayed six-metric set was selected after inspecting the original results; it is exploratory and is not reselected here.

Only these three quantitative panels are generated, as PDF/SVG/PNG. Plotted values and display transformations are written under the output `source_data/`; immutable inputs are never modified.

F4f retains the original shared-MoA, no-shared-annotated-MoA and unavailable-annotation strata, paired drug-bootstrap intervals and Holm-adjusted paired Wilcoxon P values across 30 cell–stratum tests.
