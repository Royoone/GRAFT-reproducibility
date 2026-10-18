#!/usr/bin/env python3
"""Summarize actual fixed-count runs, with explicit resampling units."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr, norm
from .evaluate_fixed import INTERNAL, EXTERNAL, SEEDS, PRIMARY_K, draw_order


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    out = np.empty(len(p))
    out[order] = np.minimum(1, np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return out


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    out = np.empty(len(p))
    out[order] = np.minimum(1, np.minimum.accumulate((p[order]*len(p)/np.arange(1, len(p)+1))[::-1])[::-1])
    return out


def paired_stats(values):
    z = np.asarray(values, float)
    z = z[np.isfinite(z)]
    if not len(z):
        return dict(n_drugs=0, mean=np.nan, lo=np.nan, hi=np.nan, p=np.nan, positive_fraction=np.nan, log10_p=np.nan, p_underflow=False)
    rng = np.random.default_rng(20260924)
    boots = np.concatenate([z[rng.integers(len(z), size=(100, len(z)))].mean(1) for _ in range(40)])
    lo, hi = np.quantile(boots, [.025, .975])
    p = 1.0 if np.all(z == 0) else float(wilcoxon(z, alternative='two-sided', method='auto').pvalue)
    logp = float(np.log10(p)) if p > 0 else float((np.log(2)+norm.logsf(abs(wilcoxon(z, method='approx').zstatistic)))/np.log(10))
    return dict(n_drugs=len(z), mean=z.mean(), lo=lo, hi=hi, p=p, positive_fraction=(z > 0).mean(), log10_p=logp, p_underflow=(p == 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    curves, primary, effects, mechanisms, infos, audits = [], [], [], [], [], []
    for cell in INTERNAL+EXTERNAL:
        folder = args.runs/cell
        info = json.loads((folder/'COMPLETE.json').read_text())
        assert info['draws'] == 30 and info['seeds'] == SEEDS and info['primary_k'] == PRIMARY_K
        assert info['cell'] == cell
        infos.append(info)
        split = pd.read_csv(folder/'query_split.csv')
        pool = split.loc[split.partition == 'anchor_pool', 'drug'].tolist()
        query = split.loc[split.partition == 'query', 'drug'].tolist()
        orders = pd.read_csv(folder/'anchor_order.csv')
        for draw in range(30):
            actual = orders[orders.draw == draw].sort_values('rank').drug.tolist()
            assert actual == draw_order(pool, cell, draw)[:max(info['budgets'])]
            assert not set(actual) & set(query)
        with np.load(folder/'paired_gain_draws.npz') as z:
            assert z['gain'].shape == (3, 30, len(info['budgets']), len(query))
            assert z['names'].tolist() == query
            assert np.nanmax(np.abs(z['gain'][:, :, 0])) == 0
            arr = z['gain'][:, :, info['budgets'].index(PRIMARY_K)].astype(float)
            count = np.isfinite(arr).sum(axis=(0, 1))
            effect = np.divide(np.nansum(arr, axis=(0, 1)), count, out=np.full(len(query), np.nan), where=count > 0)
        effects.extend(dict(cell=cell, cohort=info['cohort'], drug=n, gain_r20=float(v)) for n, v in zip(query, effect))
        run = pd.read_csv(folder/'run_metrics.csv')
        assert not run.duplicated(['seed', 'draw', 'k', 'method']).any()
        for k in info['budgets']:
            assert len(run[(run.k == k) & (run.method == 'ridge_remix')]) == 90
        run['cell'], run['cohort'] = cell, info['cohort']
        curves.append(run)
        pdg = pd.read_csv(folder/'primary_per_drug.csv.gz')
        assert not pdg.duplicated(['seed', 'method', 'drug', 'metric']).any()
        pdg['cell'], pdg['cohort'] = cell, info['cohort']
        primary.append(pdg)
        mech = pd.read_csv(folder/'primary_mechanism.csv.gz')
        assert len(mech) == len(query)*90
        assert not mech.duplicated(['seed', 'draw', 'drug']).any()
        mech['cell'], mech['cohort'] = cell, info['cohort']
        mechanisms.append(mech)
        audits.append(dict(cell=cell, expected_runs=90, checked_budgets=info['budgets'], anchor_query_overlap=0))
    runs = pd.concat(curves, ignore_index=True)
    runs.to_csv(out/'run_metrics.csv', index=False)
    per = pd.concat(primary, ignore_index=True).groupby(['cohort', 'cell', 'method', 'drug', 'metric'], as_index=False).value.mean()
    per.to_csv(out/'primary_drug_means.csv.gz', index=False, compression='gzip')
    means = per.groupby(['cohort', 'cell', 'method', 'metric']).value.agg(['mean', 'count']).reset_index()
    means.to_csv(out/'primary_metric_summary.csv', index=False)
    gains = pd.DataFrame(effects)
    gains.to_csv(out/'primary_drug_gain.csv', index=False)
    stats = pd.DataFrame([dict(cell=c, cohort=g.cohort.iloc[0], **paired_stats(g.gain_r20)) for c, g in gains.groupby('cell')])
    stats['p_holm'] = holm(stats.p)
    stats.to_csv(out/'primary_paired_tests.csv', index=False)
    metric_r20 = means[(means.metric == 'r20') & (means.method == 'ridge_remix')].set_index('cell')['mean']
    metric_d0 = means[(means.metric == 'r20') & (means.method == 'D0')].set_index('cell')['mean']
    assert np.allclose(stats.set_index('cell')['mean'], (metric_r20-metric_d0).reindex(stats.cell), atol=2e-7)
    seed_summary = runs[runs.method == 'ridge_remix'].groupby(['cohort', 'cell', 'k', 'seed'], as_index=False)[['r20', 'gain_r20']].mean()
    seed_summary.to_csv(out/'seed_means.csv', index=False)
    # Seeds are averaged within each anchor list; never treated as new drug samples.
    draw = runs[runs.method == 'ridge_remix'].groupby(['cohort', 'cell', 'k', 'draw'], as_index=False)[['r20', 'd0_r20', 'gain_r20']].mean()
    draw.to_csv(out/'draw_means.csv', index=False)
    rows = []
    for (cohort, cell, k), g in draw.groupby(['cohort', 'cell', 'k']):
        rows.append(dict(cohort=cohort, cell=cell, k=k, n_lists=len(g),
                         r20=g.r20.mean(), d0_r20=g.d0_r20.mean(), gain=g.gain_r20.mean(),
                         draw_sd=g.gain_r20.std(), draw_lo=g.gain_r20.quantile(.025),
                         draw_hi=g.gain_r20.quantile(.975), positive_lists=int((g.gain_r20 > 0).sum())))
    pd.DataFrame(rows).to_csv(out/'budget_summary.csv', index=False)
    # Keep the set of cells fixed over every macro curve, including higher budgets.
    macro = []
    groups = [('LINCS (5)', INTERNAL, [0, 1, 2, 5, 10, 15, 20, 30, 50]),
              ('External (5)', EXTERNAL, [0, 1, 2, 5, 10, 15]),
              ('External larger pools (3)', EXTERNAL[:3], [0, 1, 2, 5, 10, 15, 20, 30, 50])]
    for label, cells, budgets in groups:
        d = draw[draw.cell.isin(cells) & draw.k.isin(budgets)]
        assert d.groupby(['k', 'draw']).cell.nunique().eq(len(cells)).all()
        d = d.groupby(['k', 'draw'], as_index=False)[['r20', 'd0_r20', 'gain_r20']].mean()
        for k, g in d.groupby('k'):
            macro.append(dict(cohort=label, n_cells=len(cells), k=k, r20=g.r20.mean(),
                              d0_r20=g.d0_r20.mean(), gain=g.gain_r20.mean(),
                              draw_sd=g.gain_r20.std(), draw_lo=g.gain_r20.quantile(.025), draw_hi=g.gain_r20.quantile(.975)))
    pd.DataFrame(macro).to_csv(out/'macro_budgets.csv', index=False)
    # Internal macro: shared drug identity is jointly resampled across all five cells.
    internal = gains[gains.cohort == 'LINCS'].pivot(index='drug', columns='cell', values='gain_r20').reindex(columns=INTERNAL)
    internal = internal.dropna()
    pd.DataFrame([dict(cohort='LINCS', **paired_stats(internal.mean(axis=1)))]).to_csv(out/'internal_paired_macro.csv', index=False)
    rng = np.random.default_rng(20260924)
    ext = [gains.loc[gains.cell == c, 'gain_r20'].dropna().to_numpy() for c in EXTERNAL]
    bootstrap = np.zeros(4000)
    for values in ext:
        bootstrap += np.concatenate([values[rng.integers(len(values), size=(100, len(values)))].mean(1) for _ in range(40)])/5
    pd.DataFrame([dict(cohort='External', mean=np.mean([v.mean() for v in ext]),
                       lo=np.quantile(bootstrap, .025), hi=np.quantile(bootstrap, .975),
                       n_cells=5, n_cell_drug_records=sum(map(len, ext)),
                       unit='within-cell drug bootstrap, fixed equal cell weights')]).to_csv(out/'external_paired_macro.csv', index=False)
    mech = pd.concat(mechanisms, ignore_index=True)
    grouped = mech.groupby(['cohort', 'cell', 'drug', 'moa_relation'], as_index=False).agg(
        gain_r20=('gain_r20', 'mean'), max_tanimoto=('max_tanimoto', 'mean'), n_memberships=('gain_r20', 'count'))
    grouped.to_csv(out/'mechanism_drug_strata.csv', index=False)
    ms = pd.DataFrame([dict(cohort=co, cell=c, moa_relation=mo, **paired_stats(g.gain_r20))
                       for (co, c, mo), g in grouped.groupby(['cohort', 'cell', 'moa_relation'])])
    ok = ms.p.notna()
    ms.loc[ok, 'p_holm'] = holm(ms.loc[ok, 'p'])
    ms.to_csv(out/'mechanism_paired_tests.csv', index=False)
    similarity, bins = [], []
    md = mech.groupby(['cohort', 'cell', 'drug'], as_index=False)[['gain_r20', 'max_tanimoto', 'max_post_cosine']].mean()
    md.to_csv(out/'similarity_drug_means.csv', index=False)
    for (co, c), g in md.groupby(['cohort', 'cell']):
        for feature in ['max_tanimoto', 'max_post_cosine']:
            s = g.dropna(subset=['gain_r20', feature])
            rho, p = spearmanr(s[feature], s.gain_r20)
            similarity.append(dict(cohort=co, cell=c, feature=feature, n_drugs=len(s), rho=rho, p=p))
            if s[feature].nunique() < 4:
                continue
            quartile = pd.qcut(s[feature], 4, labels=False, duplicates='drop')
            for q, sg in s.groupby(quartile):
                bins.append(dict(cohort=co, cell=c, feature=feature, quartile=int(q)+1,
                                 similarity=sg[feature].mean(), **paired_stats(sg.gain_r20)))
    sim = pd.DataFrame(similarity)
    sim['q_bh'] = bh(sim.p.fillna(1))
    sim.to_csv(out/'similarity_associations.csv', index=False)
    pd.DataFrame(bins).to_csv(out/'similarity_quartiles.csv', index=False)
    (out/'input_manifest.json').write_text(json.dumps(infos, indent=2)+'\n')
    (out/'analysis_audit.json').write_text(json.dumps(dict(status='complete', cells=audits,
        primary_k=10, primary_holm_family=10, mechanism_holm_family=int(ok.sum()),
        bootstrap_draws=4000, resampling_unit='drug after averaging model seeds and anchor draws',
        budget_band='empirical anchor-list interval, not confidence interval of the mean'), indent=2)+'\n')
    print(stats.to_string(index=False))


if __name__ == '__main__':
    main()
