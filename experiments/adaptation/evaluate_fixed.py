#!/usr/bin/env python3
"""Evaluate the manuscript's frozen 30 anchor lists and ten-anchor comparison."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from scipy.stats import rankdata

INTERNAL = ['A375', 'A549', 'HT29', 'MCF7', 'PC3']
EXTERNAL = ['HepG2', 'MCF10A', 'Jurkat', 'K562_L1000', 'HAP1']
SEEDS = [52, 229, 564]
BUDGETS = [0, 1, 2, 5, 10, 15, 20, 30, 50]
PRIMARY_K = 10
EXCLUDED = ['BRD-K43389675', 'BRD-K49371609', 'BRD-K17516382']
METHODS = ['D0', 'ordinary', 'ridge', 'ridge_remix']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def draw_order(pool, cell, draw):
    """Independent of model seed, outcomes, annotation and input row ordering."""
    salt = int(hashlib.sha256(cell.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(np.random.SeedSequence([20260924, salt, draw]))
    return rng.permutation(sorted(pool)).tolist()


def pearson(a, b):
    aa, bb = a-a.mean(1, keepdims=True), b-b.mean(1, keepdims=True)
    an, bn = np.linalg.norm(aa, axis=1), np.linalg.norm(bb, axis=1)
    r = (aa*bb).sum(1)/np.maximum(an*bn, 1e-12)
    r[an == 0] = np.nan
    r[(an > 0) & (bn == 0)] = 0
    return r


def metric_arrays(y, p, names, full=True):
    out = {}
    rows = np.arange(len(y))[:, None]
    for k in ([20, 50, 100] if full else [20]):
        ix = np.argpartition(np.abs(y), -k, axis=1)[:, -k:]
        t, v = y[rows, ix], p[rows, ix]
        r = pearson(t, v)
        if k == 20:
            r[np.isin(names, EXCLUDED)] = np.nan
        out[f'r{k}'] = r
        if not full:
            continue
        rho = pearson(rankdata(t, axis=1), rankdata(v, axis=1))
        if k == 20:
            rho[np.isin(names, EXCLUDED)] = np.nan
        out[f'rho{k}'] = rho
        err = v-t
        out[f'mse{k}'] = np.mean(err**2, axis=1)
        out[f'rmse{k}'] = np.sqrt(out[f'mse{k}'])
        out[f'mae{k}'] = np.mean(np.abs(err), axis=1)
        pix = np.argpartition(np.abs(p), -k, axis=1)[:, -k:]
        chosen = np.zeros(y.shape, bool)
        chosen[rows, pix] = True
        out[f'recall{k}'] = chosen[rows, ix].mean(1)
    return out


def load_parameters(path):
    with np.load(path, allow_pickle=False) as archive:
        if any(archive[k].dtype != np.float32 or not np.isfinite(archive[k]).all() for k in archive.files):
            raise ValueError("Invalid frozen adapter tensors")
        return {k: archive[k].astype(float) for k in archive.files}


def predict(p0, y, features, a, q, parameters=None):
    if not len(a):
        return p0[q].copy()
    f = features/np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    phi = np.column_stack([f, np.ones(len(f))])
    lam = 1.0 if parameters is None else float(np.logaddexp(0, parameters['log_lam'])+.05)
    if parameters is not None and '_kernel' in parameters:
        tau = float(np.logaddexp(0, parameters['log_tau'])+.05)
        delta = softmax((f[q]@f[a].T)/tau, axis=1)@(y[a]-p0[a])
    else:
        coeff = np.linalg.solve(phi[a]@phi[a].T + lam*np.eye(len(a)), y[a]-p0[a])
        delta = (phi[q]@phi[a].T)@coeff
    if parameters is not None:
        if '_remix' in parameters:
            delta += expit(parameters['raw_mix'])*((delta@parameters['V'].T)@parameters['U'].T)
        delta *= expit(parameters['raw_alpha'])*expit(parameters['raw_gate'])
    return p0[q]+delta


def load_fold(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(names=z['names'].astype(str), y=z['true'].astype(float),
                    p0=z['pred'].astype(float), post=z['post64'].astype(float),
                    uni=z['unipert'].astype(float))


def verify_checkpoint_prediction(fold, a, q, checkpoint, expected, root=None):
    import torch
    from .model import GatedAdapter
    metadata = json.loads((checkpoint.parent / "metadata.json").read_text())
    model = GatedAdapter(kind=metadata["kind"])
    with np.load(checkpoint, allow_pickle=False) as archive:
        model.load_state_dict({k:torch.tensor(archive[k]) for k in archive.files}, strict=True)
    model.eval()
    with torch.no_grad():
        got = model(torch.tensor(fold['p0'], dtype=torch.float32),
                    torch.tensor(fold['y'], dtype=torch.float32),
                    torch.tensor(fold['post'], dtype=torch.float32),
                    torch.tensor(a, dtype=torch.long), torch.tensor(q, dtype=torch.long)).numpy()
    error = float(np.max(np.abs(got-expected)))
    assert error < 1e-3, ('Frozen-module equivalence', error)
    return error


def annotations(path):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    ann = pd.read_csv(path).fillna('').drop_duplicates('Compound').set_index('Compound')
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    moa, fps = {}, {}
    for name, row in ann.iterrows():
        moa[name] = {t.strip().lower() for t in str(row.moa).split('|') if t.strip()}
        mol = Chem.MolFromSmiles(str(row.SMILES))
        if mol is not None:
            fps[name] = gen.GetFingerprint(mol)
    return moa, fps


def mechanism(anchors, queries, moa, fps):
    from rdkit import DataStructs
    am = set().union(*(moa.get(n, set()) for n in anchors))
    af = [fps[n] for n in anchors if n in fps]
    groups, sims = [], []
    for n in queries:
        qm = moa.get(n, set())
        groups.append(('Shared MoA' if qm & am else 'No shared annotated MoA')
                      if qm and am else 'Annotation unavailable')
        sims.append(max(DataStructs.BulkTanimotoSimilarity(fps[n], af)) if n in fps and af else np.nan)
    return groups, sims


def evaluate_method_budgets(cell, output, feature_dir, adapter_root):
    folder=Path(output)/cell; info=json.loads((folder/'COMPLETE.json').read_text())
    internal=cell in INTERNAL
    fd=Path(feature_dir)
    ad=Path(adapter_root)/(cell if internal else 'external')
    extra='kernel' if internal else 'remix_wp5'
    paths={'ridge':ad/'ridge/adapter.npz',extra:ad/'kernel/adapter.npz' if internal else Path(adapter_root)/'PC3/ridge_remix/adapter.npz'}
    params={m:load_parameters(p) for m,p in paths.items()};params[extra]['_kernel' if internal else '_remix']=True
    split=pd.read_csv(folder/'query_split.csv');queries=split[split.partition=='query'].drug.tolist();order=pd.read_csv(folder/'anchor_order.csv')
    known=pd.read_csv(folder/'run_metrics.csv');records=[];errors=[]
    for seed in SEEDS:
        f=load_fold(fd/f'{cell}_{seed}.npz');lookup={n:i for i,n in enumerate(f['names'])};q=np.array([lookup[n] for n in queries])
        y=f['y'][q];ix=np.argpartition(np.abs(y),-20,axis=1)[:,-20:];rows=np.arange(len(y))[:,None];t=y[rows,ix]
        mask=np.isin(queries,EXCLUDED)
        base=pearson(t,f['p0'][q][rows,ix]);base[mask]=np.nan
        norm={key:f[key]/np.maximum(np.linalg.norm(f[key],axis=1,keepdims=True),1e-12) for key in ['post','uni']}
        phi=np.column_stack([norm['post'],np.ones(len(f['names']))])
        for draw in range(30):
            ns=order[order.draw==draw].sort_values('rank').drug.tolist()
            for k in info['budgets']:
                ai=np.array([lookup[n] for n in ns[:k]],int)
                for method in ['ordinary','ridge',extra]:
                    par=params.get(method)
                    if k==0:p=f['p0'][q]
                    else:
                        if method=='kernel':
                            tau=float(np.logaddexp(0,par['log_tau'])+.05)
                            delta=softmax(norm['uni'][q]@norm['uni'][ai].T/tau,axis=1)@(f['y'][ai]-f['p0'][ai])
                        else:
                            lam=1. if par is None else float(np.logaddexp(0,par['log_lam'])+.05)
                            coeff=np.linalg.solve(phi[ai]@phi[ai].T+lam*np.eye(k),f['y'][ai]-f['p0'][ai])
                            delta=(phi[q]@phi[ai].T)@coeff
                            if method=='remix_wp5':delta+=expit(par['raw_mix'])*((delta@par['V'].T)@par['U'].T)
                        if par is not None:delta*=expit(par['raw_alpha'])*expit(par['raw_gate'])
                        p=f['p0'][q]+delta
                    r=pearson(t,p[rows,ix]);r[mask]=np.nan
                    value=float(np.nanmean(r))
                    if k==10:
                        v=known[(known.seed==seed)&(known.draw==draw)&(known.k==10)&(known.method==method)].r20.iloc[0]
                        errors.append(abs(value-v));assert abs(value-v)<1e-10
                    records.append(dict(cell=cell,cohort=info['cohort'],seed=seed,draw=draw,k=k,method=method,r20=value,gain_r20=float(np.nanmean(r-base))))
    pd.DataFrame(records).to_csv(folder/'method_budget_runs.csv',index=False)
    (folder/'method_budget_audit.json').write_text(json.dumps(dict(cell=cell,rows=len(records),primary_agreement_max_error=max(errors),script_sha256=sha(Path(__file__))),indent=2)+'\n')
    print('COMPLETE',cell,flush=True)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--features', type=Path, required=True)
    ap.add_argument('--adapters', type=Path, required=True)
    ap.add_argument('--splits', type=Path, required=True)
    ap.add_argument('--annotations', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cells', nargs='+', default=INTERNAL+EXTERNAL)
    args = ap.parse_args()
    out = args.out
    args.draws = 30
    out.mkdir(parents=True, exist_ok=True)
    paper_contract = json.loads((args.splits/"paper_contract.json").read_text())
    ann_file = args.annotations
    import torch
    torch.set_num_threads(2)
    moa, fps = annotations(ann_file)
    for cell in args.cells:
        cohort = 'LINCS' if cell in INTERNAL else 'External'
        for name, checksum in paper_contract['contexts'][cell]['files'].items():
            if sha(args.splits/cell/name) != checksum:
                raise ValueError(f'Frozen paper list checksum mismatch: {cell}/{name}')
        dest = out/cell
        dest.mkdir(exist_ok=True)
        marker = dest/'COMPLETE.json'
        if marker.exists():
            raise RuntimeError(f'Completed outputs already exist: {dest}; use a new output directory.')
        feature_dir = args.features
        adapter_dir = args.adapters/(cell if cohort == 'LINCS' else 'external')
        checkpoints = {m: adapter_dir/m/'adapter.npz' for m in ['ridge', 'ridge_remix']}
        extra_method = 'kernel' if cohort == 'LINCS' else 'remix_wp5'
        checkpoints[extra_method] = (adapter_dir/'kernel/adapter.npz' if cohort == 'LINCS'
                                     else args.adapters/'PC3/ridge_remix/adapter.npz')
        methods = METHODS+[extra_method]
        for method, checkpoint in checkpoints.items():
            metadata = json.loads((checkpoint.parent/'metadata.json').read_text())
            if metadata.get('sha256') and sha(checkpoint) != metadata['sha256']:
                raise ValueError(f'Adapter tensor checksum mismatch: {checkpoint}')
            origin = metadata.get('original_checkpoint_sha256')
            expected = paper_contract['contexts'][cell]['source_checkpoint_hashes'][method]
            if origin is not None and origin != expected:
                raise ValueError(f'Adapter source does not match the fixed paper experiment: {cell}/{method}')
        params = {m: load_parameters(p) for m, p in checkpoints.items()}
        params['ridge_remix']['_remix'] = True
        params[extra_method]['_kernel' if extra_method == 'kernel' else '_remix'] = True
        first = load_fold(feature_dir/f'{cell}_{SEEDS[0]}.npz')
        frozen_split = pd.read_csv(args.splits/cell/'query_split.csv')
        queries = frozen_split.loc[frozen_split.partition == 'query', 'drug'].tolist()
        pool = frozen_split.loc[frozen_split.partition == 'anchor_pool', 'drug'].tolist()
        if cohort == 'LINCS':
            assert len(queries) == 644 and len(pool) == 2576
        assert len(queries) and len(pool)
        assert len(set(queries)) == len(queries) and not set(queries) & set(pool)
        budgets = [k for k in BUDGETS if k <= len(pool)]
        assert PRIMARY_K in budgets and 15 in budgets
        pd.DataFrame([dict(drug=n, partition='query') for n in queries]+[dict(drug=n, partition='anchor_pool') for n in pool]).to_csv(dest/'query_split.csv', index=False)
        frozen_order = pd.read_csv(args.splits/cell/'anchor_order.csv')
        orders = [frozen_order[frozen_order.draw == d].sort_values('rank').drug.tolist() for d in range(args.draws)]
        for d, order in enumerate(orders):
            assert order == draw_order(pool, cell, d)[:max(budgets)]
            assert not set(order) & set(queries)
        pd.DataFrame([dict(draw=d, rank=i+1, drug=n) for d, order in enumerate(orders)
                      for i, n in enumerate(order[:max(budgets)])]).to_csv(dest/'anchor_order.csv', index=False)
        mech = [mechanism(order[:PRIMARY_K], queries, moa, fps) for order in orders]
        n = len(queries)
        gains = np.full((len(SEEDS), args.draws, len(budgets), n), np.nan, np.float32)
        d0 = np.full((len(SEEDS), n), np.nan)
        summary, per_drug, hashes, equivalence = [], [], [], []
        mechanism_rows = []
        for si, seed in enumerate(SEEDS):
            fp = feature_dir/f'{cell}_{seed}.npz'
            fold = load_fold(fp)
            names = fold['names']
            assert len(set(names)) == len(names)
            lookup = {n: i for i, n in enumerate(names)}
            qi = np.array([lookup[n] for n in queries])
            pi = np.array([lookup[n] for n in pool])
            used = np.r_[qi, pi]
            for key in ['y', 'p0', 'post', 'uni']:
                assert np.isfinite(fold[key][used]).all(), (cell, seed, key)
            if cohort == 'External':
                assert (np.linalg.norm(fold['uni'][used], axis=1) > 0).all()
            truth, base = fold['y'][qi], fold['p0'][qi]
            bm = metric_arrays(truth, base, queries)
            d0[si] = bm['r20']
            for met, values in bm.items():
                per_drug.extend(dict(seed=seed, method='D0', drug=n, metric=met, value=float(v)) for n, v in zip(queries, values))
            sums = {m: {met: np.zeros(n) for met in bm} for m in methods[1:]}
            counts = {m: {met: np.zeros(n, int) for met in bm} for m in methods[1:]}
            for draw, order in enumerate(orders):
                for bi, k in enumerate(budgets):
                    anchors = order[:k]
                    ai = np.array([lookup[n] for n in anchors], int)
                    assert len(ai) == k and not set(ai) & set(qi)
                    pr = predict(fold['p0'], fold['y'], fold['post'], ai, qi, params['ridge_remix'])
                    if draw == 0 and k == PRIMARY_K:
                        error = verify_checkpoint_prediction(fold, ai, qi[:30], checkpoints['ridge_remix'], pr[:30])
                        equivalence.append(dict(seed=seed, max_abs_error=error))
                    am = metric_arrays(truth, pr, queries, full=(k == PRIMARY_K))
                    gains[si, draw, bi] = am['r20']-bm['r20']
                    summary.append(dict(seed=seed, draw=draw, k=k, method='ridge_remix', n_query=n,
                                        n_pool=len(pool), n_r20=int(np.isfinite(am['r20']).sum()),
                                        **{met: float(np.nanmean(v)) for met, v in am.items()},
                                        d0_r20=float(np.nanmean(bm['r20'])), gain_r20=float(np.nanmean(am['r20']-bm['r20']))))
                    if k != PRIMARY_K:
                        continue
                    for method in methods[1:]:
                        if method == 'ridge_remix':
                            mm = am
                        else:
                            pred = predict(fold['p0'], fold['y'], fold['uni' if method == 'kernel' else 'post'], ai, qi, params.get(method))
                            mm = metric_arrays(truth, pred, queries)
                            summary.append(dict(seed=seed, draw=draw, k=k, method=method, n_query=n,
                                                n_pool=len(pool), n_r20=int(np.isfinite(mm['r20']).sum()),
                                                **{met: float(np.nanmean(v)) for met, v in mm.items()},
                                                d0_r20=float(np.nanmean(bm['r20'])), gain_r20=float(np.nanmean(mm['r20']-bm['r20']))))
                        for met, values in mm.items():
                            valid = np.isfinite(values)
                            sums[method][met][valid] += values[valid]
                            counts[method][met] += valid
                    groups, sims = mech[draw]
                    f = fold['post']/np.maximum(np.linalg.norm(fold['post'], axis=1, keepdims=True), 1e-12)
                    cos = np.max(f[qi]@f[ai].T, axis=1)
                    mechanism_rows.extend(dict(seed=seed, draw=draw, drug=n, moa_relation=g,
                                               max_tanimoto=float(t), max_post_cosine=float(co),
                                               d0_r20=float(b), adapt_r20=float(a), gain_r20=float(a-b))
                                          for n, g, t, co, b, a in zip(queries, groups, sims, cos, bm['r20'], am['r20']))
                print(cell, seed, 'draw', draw+1, '/', args.draws, flush=True)
            for method in methods[1:]:
                for met in bm:
                    values = np.divide(sums[method][met], counts[method][met],
                                       out=np.full(n, np.nan), where=counts[method][met] > 0)
                    per_drug.extend(dict(seed=seed, method=method, drug=n, metric=met, value=float(v)) for n, v in zip(queries, values))
            hashes.append(dict(file=str(fp.relative_to(args.features)), sha256=sha(fp)))
        pd.DataFrame(summary).to_csv(dest/'run_metrics.csv', index=False)
        pd.DataFrame(per_drug).to_csv(dest/'primary_per_drug.csv.gz', index=False, compression='gzip')
        pd.DataFrame(mechanism_rows).to_csv(dest/'primary_mechanism.csv.gz', index=False, compression='gzip')
        np.savez_compressed(dest/'paired_gain_draws.npz', gain=gains, d0=d0, budgets=budgets,
                            seeds=SEEDS, draws=np.arange(args.draws), names=queries)
        info = dict(cell=cell, cohort=cohort, n_query=n, n_pool=len(pool), budgets=budgets,
                    primary_k=PRIMARY_K, seeds=SEEDS, draws=args.draws, methods=methods, feature_hashes=hashes,
                    checkpoint_hashes={m: sha(p) for m, p in checkpoints.items()},
                    pytorch_module_equivalence=equivalence,
                    frozen_split_sha256=sha(args.splits/cell/'query_split.csv'), frozen_order_sha256=sha(args.splits/cell/'anchor_order.csv'), annotation_sha256=sha(ann_file),
                    evaluation_script_sha256=sha(Path(__file__)),
                    sampling='same 30 nested random lists across three model seeds; frozen query split',
                    rmse_definition='mean of per-drug sqrt(MSE), not sqrt of pooled MSE')
        marker.write_text(json.dumps(info, indent=2)+'\n')
        evaluate_method_budgets(cell, args.out, args.features, args.adapters)
        print('COMPLETE', cell, flush=True)


if __name__ == '__main__':
    main()
