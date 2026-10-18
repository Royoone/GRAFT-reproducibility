#!/usr/bin/env python3
"""Render the manuscript internal-comparison panel from complete, frozen scored records."""
from pathlib import Path
import hashlib
import json
import sys
import argparse

import numpy as np
import pandas as pd

from draw_main_grids import draw

BASE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract = json.loads((BASE / 'figure_contract.json').read_text())
    data = pd.read_csv(BASE / 'source_data/input_scored_runs.csv')
    if len(data) != 255 or data.duplicated(['cohort', 'method', 'cell', 'seed']).any():
        raise ValueError('The manuscript panel requires 255 unique run records')
    audits = []
    for name in ['F2b']:
        config = contract[name]
        note = 'GRAFT: RMS-bounded state model. G2CP-adapter: pooled model; G2CP: two-stage backbone.'
        audits.append(draw(data, config['cohort'], output, name, config['methods'],
                           tuple(config['seeds']), note))
        ranks = pd.read_csv(output / f'source_data/{name}_overall_ranking.csv')
        reference = 'GRAFT'
        expected_first = 16
        row = ranks[ranks.method.eq(reference)].iloc[0]
        if ranks.iloc[0].method != reference or int(row.first_place_count) != expected_first:
            raise ValueError('Unexpected ranking; inspect source data rather than changing the panel')
        values = pd.read_csv(output / f'source_data/{name}_metric_values.csv').set_index('method')
        baseline = json.loads((BASE / 'source_data/frozen_reference_metrics.json').read_text())[name]
        for method, metrics in baseline.items():
            for metric, expected in metrics.items():
                np.testing.assert_allclose(values.loc[method, metric], expected, rtol=0, atol=1e-12)
    record = {'input_sha256': hashlib.sha256((BASE / 'source_data/input_scored_runs.csv').read_bytes()).hexdigest(),
              'frozen_source_sha256': contract['source_sha256'], 'input_run_records': len(data),
              'figure_contract_sha256': hashlib.sha256((BASE / 'figure_contract.json').read_bytes()).hexdigest(),
              'figures': audits, 'model_training_or_selection_performed': False}
    (output / 'export_checks.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
