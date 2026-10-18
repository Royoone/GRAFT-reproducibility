"""Retain the final normalized iPSC response field and source-covered targets.

This compacts the original count-aggregation output; it does not normalize raw
counts or change thresholds. Count aggregation is supplied in ``preprocess``.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd


def compact(source: Path, reference: Path, output: Path, score_audit: Path | None = None) -> dict:
    with np.load(source, allow_pickle=False) as data, np.load(reference, allow_pickle=False) as refs:
        genes = data['genes'].astype(str)
        if not np.array_equal(genes, refs['genes'].astype(str)):
            raise ValueError('The prepared gene order must match the frozen 976-gene panel')
        catalog = set(refs['genetic_ids'].astype(str))
        names = data['perturbations'].astype(str)
        score_targets = set()
        if score_audit is not None:
            scores = pd.read_csv(score_audit)
            score_targets = set('|'.join(scores.exact_measured_targets).split('|'))
        # Six measured score targets have no source genetic embedding. Preserve
        # them for equation (10), while prediction still uses only covered IDs.
        keep = np.flatnonzero(np.isin(names, list(catalog | score_targets)))
        arrays = {key: data[key] for key in ['genes', 'contexts', 'donors', 'n_controls', 'measured_gene_mask']}
        arrays['perturbations'] = names[keep]
        for key in ['n_cells', 'n_guides', 'valid', 'effects__lognorm_nt']:
            arrays[key] = data[key][:, keep]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    record = {'screened_targets': len(names), 'source_covered_targets': int(np.isin(names, list(catalog)).sum()),
              'stored_targets': len(keep), 'association_score_targets': len(score_targets),
              'contexts': len(arrays['contexts']), 'genes': len(genes),
              'response_field': 'effects__lognorm_nt',
              'response_definition': 'CP10K using full-transcriptome UMI, log1p, same-line NonTarget subtraction',
              'stored_validity': 'at least 10 targeted cells, two guides and 50 controls; prediction further requires 20 targeted cells',
              'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--score-audit', type=Path,
                        default=Path(__file__).resolve().parents[2]/'figures/figure7/source_data/all235_evaluation_and_retention.csv')
    args = parser.parse_args()
    print(json.dumps(compact(args.input, args.reference, args.output, args.score_audit), indent=2))
