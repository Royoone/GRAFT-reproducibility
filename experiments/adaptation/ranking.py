"""The displayed six-metric ranking and the complete eighteen-metric source table."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .evaluate_fixed import INTERNAL, EXTERNAL

SIX = ["r20", "r50", "mse20", "mse50", "mae20", "mae50"]
EIGHTEEN = [f"{p}{k}" for p in ("r", "rho", "recall", "mse", "rmse", "mae") for k in (20, 50, 100)]


def rank_summary(table, output):
    """Retain the paper's metric directions, equal weights and average ties."""
    raw = pd.read_csv(table)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for stem, metrics in (("03", SIX), ("08", EIGHTEEN)):
        normalized, ranked, overall = [], [], []
        for cohort, cells in (("LINCS", INTERNAL), ("External", EXTERNAL)):
            d = raw[(raw.cohort == cohort) & raw.metric.isin(metrics)]
            methods = ["D0", "ordinary", "ridge", "ridge_remix", "kernel" if cohort == "LINCS" else "remix_wp5"]
            if len(d) != 5 * 5 * len(metrics):
                raise ValueError("The paper ranking requires all five contexts and five methods per cohort")
            means = d.groupby(["method", "metric"])["mean"].mean().unstack().loc[methods, metrics]
            ranks = means.copy()
            for metric in metrics:
                ranks[metric] = means[metric].rank(ascending=metric.startswith(("mse", "rmse", "mae")), method="average")
            aggregate = pd.DataFrame({"mean_rank": ranks.mean(axis=1), "first_place_count": ranks.eq(1).sum(axis=1), "r20": means.r20})
            order = aggregate.sort_values(["mean_rank", "r20"], ascending=[True, False]).index.tolist()
            for method in order:
                for metric in metrics:
                    all_values = d.loc[d.metric == metric, "mean"]
                    lo, hi = all_values.min(), all_values.max()
                    values = d[(d.metric == metric) & (d.method == method)].set_index("cell").loc[cells, "mean"].to_numpy()
                    norm = (values - lo)/(hi - lo) if hi > lo else np.full(5, .5)
                    if metric.startswith(("mse", "rmse", "mae")):
                        norm = 1 - norm
                    ranked.append(dict(cohort=cohort, method=method, metric=metric, raw_macro=means.loc[method, metric], rank=float(ranks.loc[method, metric])))
                    normalized.extend(dict(cohort=cohort, cell=cell, method=method, metric=metric, raw=float(v), normalized=float(n)) for cell, v, n in zip(cells, values, norm))
                overall.append(dict(cohort=cohort, method=method, n_metrics=len(metrics), mean_rank=float(aggregate.loc[method, "mean_rank"]), first_place_count=int(aggregate.loc[method, "first_place_count"])))
        pd.DataFrame(normalized).to_csv(output/f"{stem}_normalized_cell_metrics.csv", index=False)
        pd.DataFrame(ranked).to_csv(output/f"{stem}_metric_values_and_ranks.csv", index=False)
        pd.DataFrame(overall).to_csv(output/f"{stem}_overall_ranking.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank_summary(args.summary, args.output)


if __name__ == "__main__":
    main()
