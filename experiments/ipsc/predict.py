"""Final GRAFT iPSC inference, following the original record-to-entity workflow.

Uses all eligible genetic inputs, frozen five-source checkpoints and no iPSC
drug measurements. Record corrections are averaged within chemical entities
before adding the entity source reference, as in the original inference code.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.backbone.model import GRAFT


def entity_mean(records, record_to_entity, n_entities):
    output = np.zeros((n_entities, records.shape[1]), dtype=np.float64)
    np.add.at(output, record_to_entity, records)
    return output / np.bincount(record_to_entity, minlength=n_entities)[:, None]


def query_catalog(reference, catalog):
    records = reference['drug_ids'].astype(str)
    table = pd.read_csv(catalog).drop_duplicates('Compound').set_index('Compound')
    if not set(records) <= set(table.index):
        raise ValueError('The record catalog is missing frozen source drug identifiers')
    keys = table.loc[records, 'chemical_key'].astype(str).to_numpy(dtype=str)
    entities, inverse = np.unique(keys, return_inverse=True)
    base = entity_mean(reference['drug_reference'], inverse, len(entities)).astype(np.float32)
    return records, entities, inverse, base


@torch.no_grad()
def infer(args):
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads((args.bundle / 'manifest.json').read_text())
    with np.load(args.bundle / manifest['reference']['file'], allow_pickle=False) as archive:
        reference = {key: archive[key] for key in archive.files}
    with np.load(args.input, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ['genes', 'contexts', 'donors', 'perturbations',
                                             'effects__lognorm_nt', 'valid', 'n_cells', 'n_guides', 'n_controls']}
    if not np.array_equal(data['genes'].astype(str), reference['genes'].astype(str)):
        raise ValueError('Gene order differs from the frozen model panel')
    records, entities, rte, base = query_catalog(reference, args.catalog)
    positions = {name: i for i, name in enumerate(reference['genetic_ids'].astype(str))}
    perts = data['perturbations'].astype(str)
    covered = np.array([name in positions for name in perts])
    gi = np.array([positions.get(name, 0) for name in perts])
    y = data['effects__lognorm_nt'].astype(np.float32)
    valid = data['valid'].astype(bool) & np.isfinite(y).all(2) & (data['n_cells'] >= 20)
    valid &= (data['n_guides'] >= 2) & (data['n_controls'] >= 50)[:, None] & covered[None, :]
    valid &= np.isfinite(reference['genetic_embeddings'][gi]).all(1)[None, :]
    lines = data['contexts'].astype(str)
    selected = np.arange(len(lines)) if not args.lines else np.array([list(lines).index(line) for line in args.lines])
    models = []
    for entry in manifest['models']:
        path = args.bundle / entry['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'Checkpoint checksum mismatch: {path.name}')
        model = GRAFT(len(data['genes']), entry['seed'], args.device)
        with np.load(path, allow_pickle=False) as archive:
            model.load_state_dict({key: torch.from_numpy(archive[key].copy()) for key in archive.files}, strict=True)
        models.append((model.eval(), float(entry['scale']), int(entry['seed'])))
    args.output.mkdir(parents=True, exist_ok=True)
    predicted = np.lib.format.open_memmap(args.output / 'prediction_mean.npy', mode='w+', dtype=np.float32,
                                         shape=(len(selected), len(entities), len(data['genes'])))
    np.save(args.output / 'source_mean.npy', base)
    np.savez_compressed(args.output / 'axes.npz', lines=lines[selected], donors=data['donors'][selected],
                        genes=data['genes'], chemical_keys=entities)
    ledger = []
    rms = np.zeros((len(entities), len(selected)), dtype=np.float64)
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=args.device)
    for ii, line_index in enumerate(selected):
        mask = valid[line_index]
        if mask.sum() < 5:
            raise ValueError(f'Too few eligible genetic conditions in {lines[line_index]}')
        ref = reference['genetic_reference'][gi[mask]]
        ensemble = np.zeros((len(entities), len(data['genes'])), dtype=np.float64)
        for model, scale, seed in models:
            support = {'genetic': {'x': tensor(reference['genetic_embeddings'][gi[mask]]),
                                   'mean': tensor(ref / scale),
                                   'delta': tensor((y[line_index, mask] - ref) / scale)}}
            state = model.encode(support)
            residual = np.zeros_like(reference['drug_reference'])
            for start in range(0, len(records), args.chunk):
                stop = start + args.chunk
                drug_ref = reference['drug_reference'][start:stop]
                query = {'x': tensor(reference['drug_embeddings'][start:stop]), 'mean': tensor(drug_ref / scale)}
                output = model(query, state).float().cpu().numpy() * scale
                residual[start:stop] = (output - drug_ref).astype(np.float32)
            ensemble += (entity_mean(residual, rte, len(entities)) + base) / len(models)
        predicted[ii] = ensemble.astype(np.float32)
        # The original analysis promotes the saved gene-level ensemble before RMS.
        rms[:, ii] = np.sqrt(np.mean(predicted[ii].astype(np.float64) ** 2, axis=1))
        for jj, target in enumerate(perts):
            ledger.append({'line': lines[line_index], 'target': target, 'included': bool(mask[jj]),
                           'n_cells': int(data['n_cells'][line_index, jj]),
                           'n_guides': int(data['n_guides'][line_index, jj]),
                           'n_controls': int(data['n_controls'][line_index])})
        predicted.flush()
        print(f'{lines[line_index]}: {int(mask.sum())} genetic inputs; {len(entities)} predicted entities', flush=True)
    frame = pd.DataFrame(rms, columns=lines[selected]); frame.insert(0, 'chemical_key', entities)
    frame.to_csv(args.output / 'per_line_response_rms.csv', index=False)
    pd.DataFrame(ledger).to_csv(args.output / 'genetic_input_ledger.csv', index=False)
    metadata = {'model': 'GRAFT', 'seeds': [seed for _, _, seed in models],
                'n_drug_records': len(records), 'n_entities': len(entities),
                'shape': list(predicted.shape), 'target_drug_measurements_used': False,
                'response': 'original processed scale', 'ensemble': 'signed gene-level arithmetic mean before RMS',
                'record_to_entity': 'equal mean of record corrections plus equal entity source reference',
                'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest()}
    (args.output / 'prediction_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=REPO/'data/processed/ipsc/graft_input_976.npz')
    parser.add_argument('--catalog', type=Path, default=REPO/'data/processed/ipsc/record_catalog.csv')
    parser.add_argument('--bundle', type=Path, default=REPO/'data/model_bundle')
    parser.add_argument('--output', type=Path, default=REPO/'results/ipsc')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--lines', nargs='+')
    parser.add_argument('--chunk', type=int, default=512)
    parser.add_argument('--threads', type=int, default=4)
    infer(parser.parse_args())
