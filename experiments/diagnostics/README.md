# Fixed component and representation diagnostics

These entries recompute the scientific controls used in FS2 and FS4. Run from the repository root. Repeat the relevant commands for the five internal contexts and seeds 52, 229 and 564. Output directories are separate from frozen figure inputs.

## Genetic inputs and state dimensions

```bash
python -m experiments.diagnostics.evaluate --target A375 --seed 52
```

The fixed checkpoint is evaluated under the paper's absolute genetic-input budgets, largest/random state-dimension masks, raw/residual shuffles, matched support, source-cell replacement and zero state. `--task budgets`, `dimensions` or `controls` runs only that analysis. The complete entry produces 68 rows per fold and uses the original query order and random-sampling rules. Target drug responses are sealed during inference.

## Capacity and conditional interfaces

```bash
python -m experiments.diagnostics.train \
  --configuration width32 --target A375 --seed 52 --device cuda:0
python -m experiments.diagnostics.interfaces \
  --target A375 --seed 52 --device cuda:0 --output results/diagnostics/interfaces
```

The six capacity controls are `state16`, `state64`, `width32`, `width128`, `hyper1` and `moment16`. Each uses the fixed 200-step source-only training recipe. The interface entry runs the eighteen paper controls by default; `--variants` selects named scientific controls from that fixed list. These include attention, token pooling, source-only gene graph, context branches, ECFP4 input, low-rank readout and a first-order inner update. Their initialization/parameter registration and original sampling are preserved.

## Representation probes

```bash
python -m experiments.diagnostics.probes --output results/probes
```

This uses `data/processed/diagnostics/representations.npz` and the packaged internal benchmark checkpoints. It regenerates the five frozen-identity replacements and eleven ridge-probe spaces for every context/seed, producing 240 probe rows and the FS2f/g mean tables. Features are reconstructed from the prescribed checkpoint rather than taken from adaptation feature files.

Frozen identity replacement fits an identity-space projection on the non-query rows without fitting target drug response labels. The target-trained ridge diagnostic uses the measured residuals of 2,576 non-query target drugs; 644 target drugs are held out for scoring. PCA is fitted only on those support rows. This diagnostic is distinct from the zero-target-drug benchmark.

For a smaller computation, use `--cells A375 --seeds 52`. `--tasks ridge` or `swap` restricts the calculation. The fixed random/shuffled representations, gene/record axes and protocol are retained. The quick figure command draws the selected FS2a/b panels; other diagnostic calculations return numerical tables. Frozen source inputs are not overwritten.
