"""Recompute the adopted FS2f/g frozen-identity and target-trained ridge probes.

The 644 query drugs are scored only after prediction. Ridge probes fit measured
responses for the other 2,576 target drugs; frozen identity replacements use no
measured target drug response for fitting or prediction. No backbone is trained.
"""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
from ..backbone.data import FIVE,SEEDS,load_sources,references,pack,query_names,sha256
from ..backbone.predict import load_paper_model
from ..backbone.evaluate import score

MT_KEYS=('esm_mean','onto_mean','protbert_mean','pseaac_mean')
RIDGE_SPACES=('post64','unipert','ecfp256','ecfp2048','unipert_ecfp256',*MT_KEYS,'random256','shuffle_unipert')
SWAP_SPACES=('ecfp2048','ecfp256','random256','shuffle_unipert','esm_mean')

def adapt(p0,y,features,anchor_idx,query_idx,lam=1.0):
    # Original dual ridge: row L2 normalization followed by a constant feature.
    # Lambda1 is added to the complete dual Gram, as in the author implementation.
    p0=np.asarray(p0,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    feat=np.asarray(features,dtype=np.float64)
    feat=feat/np.maximum(np.sqrt((feat*feat).sum(1,keepdims=True)),1e-12)
    phi=np.concatenate([feat,np.ones((len(feat),1),dtype=np.float64)],axis=1)
    a=np.asarray(anchor_idx,dtype=int);q=np.asarray(query_idx,dtype=int)
    if not a.size:return p0[q].copy()
    pa=phi[a];residual=y[a]-p0[a];system=pa@pa.T+float(lam)*np.eye(len(a))
    try:alpha=np.linalg.solve(system,residual)
    except np.linalg.LinAlgError:alpha=np.linalg.lstsq(system,residual,rcond=None)[0]
    return p0[q]+phi[q]@(pa.T@alpha)

def pca_fit_transform(train, test, dim, seed=0):
    ok = np.isfinite(train).all(1)
    if ok.sum() < dim + 2:
        return None, None
    pca = PCA(n_components=min(dim, int(ok.sum() - 1)), random_state=seed)
    pca.fit(np.nan_to_num(train[ok], nan=0.0))
    return pca.transform(np.nan_to_num(train, nan=0.0)), pca.transform(np.nan_to_num(test, nan=0.0))

def finite_rows(*mats):
    keep = np.ones(len(mats[0]), dtype=bool)
    for m in mats:
        keep &= np.isfinite(m).all(axis=1) if m.ndim == 2 else np.isfinite(m)
    return keep

def load_base(path):
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    return out

def fold_features(base, names, support_idx, query_idx, seed):
    """Fold-safe derived spaces. PCA is fit on support rows only."""
    spaces = {}
    spaces["unipert"] = np.asarray(base["unipert"], dtype=np.float64)
    spaces["random256"] = np.asarray(base["random256"], dtype=np.float64)
    spaces["shuffle_unipert"] = np.asarray(base["shuffle_unipert"], dtype=np.float64)
    if "ecfp" in base:
        ecfp = np.asarray(base["ecfp"], dtype=np.float64)
        spaces["ecfp2048"] = ecfp
        tr, te = pca_fit_transform(ecfp[support_idx], ecfp[query_idx], 256, seed)
        if tr is not None:
            full = np.zeros((len(names), tr.shape[1]), dtype=np.float64)
            full[support_idx] = tr
            full[query_idx] = te
            spaces["ecfp256"] = full
            uni = spaces["unipert"]
            cat_tr = np.concatenate([uni[support_idx], np.nan_to_num(ecfp[support_idx], nan=0.0)], 1)
            cat_te = np.concatenate([uni[query_idx], np.nan_to_num(ecfp[query_idx], nan=0.0)], 1)
            ctr, cte = pca_fit_transform(cat_tr, cat_te, 256, seed)
            if ctr is not None:
                full = np.zeros((len(names), ctr.shape[1]))
                full[support_idx] = ctr
                full[query_idx] = cte
                spaces["unipert_ecfp256"] = full
    for key in MT_KEYS:
        if key in base:
            spaces[key] = np.asarray(base[key], dtype=np.float64)
    return spaces

def source_ridge(p0, y, feat, support_idx, query_idx, lam=1.0):
    ok_a = finite_rows(feat[support_idx], y[support_idx], p0[support_idx])
    ok_q = finite_rows(feat[query_idx], p0[query_idx])
    if ok_a.sum() < 8 or ok_q.sum() < 8:
        return None
    a = support_idx[ok_a]
    q = query_idx[ok_q]
    pred_q = adapt(p0, y, np.nan_to_num(feat, nan=0.0), a, q, lam=lam)
    return pred_q, q, int(ok_a.sum()), int(ok_q.sum())

def project_x(src_x, src_y, q_x):
    ok = finite_rows(src_x, src_y)
    if ok.sum() < src_x.shape[1] + 2:
        return None
    xa = np.concatenate([src_x[ok], np.ones((int(ok.sum()), 1))], 1)
    w, *_ = np.linalg.lstsq(xa, src_y[ok], rcond=None)
    xq = np.concatenate([np.nan_to_num(q_x, nan=0.0), np.ones((len(q_x), 1))], 1)
    return xq @ w


def validate_identity(base,data):
    names=np.asarray(base['names'],dtype=str);queries=np.asarray(base['queries'],dtype=str)
    shared=sorted(set.intersection(*(set(data[c]['chemical']['names']) for c in FIVE)))
    if len(names)!=3220 or len(queries)!=644 or list(names)!=shared:
        raise ValueError('Representation identities must match the 3220 shared paper drugs')
    np.testing.assert_array_equal(queries,np.asarray(query_names(data),dtype=str))
    if len(set(names))!=3220 or len(set(queries))!=644:raise ValueError('Duplicate representation/query identities')
    required={'unipert','ecfp','random256','shuffle_unipert',*MT_KEYS}
    if required-set(base):raise ValueError(f'Missing adopted representations: {sorted(required-set(base))}')
    for key in required:
        if np.asarray(base[key]).ndim!=2 or len(base[key])!=3220:raise ValueError(f'Invalid representation axes: {key}')
    if base['unipert'].shape[1]!=256 or base['ecfp'].shape[1]!=2048:raise ValueError('Wrong frozen UniPert/ECFP dimension')
    pos={str(n):i for i,n in enumerate(data[FIVE[0]]['chemical']['names'])}
    actual=data[FIVE[0]]['chemical']['x'][[pos[n] for n in names]].cpu().numpy()
    np.testing.assert_allclose(base['unipert'],actual,rtol=0,atol=0)
    qset=set(queries);support=np.array([i for i,n in enumerate(names) if n not in qset],dtype=int)
    lookup={n:i for i,n in enumerate(names)};query=np.array([lookup[n] for n in queries],dtype=int)
    return names,queries,support,query


@torch.no_grad()
def extract_fold(model,data,cell,seed,names,scale,sources):
    """Rebuild the exact probe post64 path, including full-support RNG order."""
    refs=references(data,sources,data[cell],('genetic','chemical'))
    lookup={str(n):i for i,n in enumerate(data[cell]['chemical']['names'])}
    qi=np.asarray([lookup[str(n)] for n in names],dtype=int)
    q=pack(data[cell],refs,'chemical',qi,scale,labels=False)
    saved=data[cell]['chemical']['y'];truth=saved[qi].cpu().numpy().astype(np.float64)
    # native_indices(n, seed, draw=0, label='full') in the original config.
    gi=np.random.default_rng(np.random.SeedSequence([seed,17])).permutation(len(data[cell]['genetic']['names']))
    data[cell]['chemical']['y']=torch.full_like(saved,float('nan'))
    try:
        support={'genetic':pack(data[cell],refs,'genetic',gi,scale)}
        state=model.encode(support);pred=[];post=[];source=[]
        for lo in range(0,len(names),128):
            qq={k:v[lo:lo+128] for k,v in q.items()}
            p,feature=model.decode_with_features(qq,state,gain=1.)
            pred.append((p.detach().float().cpu().numpy()*scale).astype(np.float64))
            post.append(feature.detach().float().cpu().numpy().astype(np.float64))
            source.append((qq['mean'].detach().float().cpu().numpy()*scale).astype(np.float64))
    finally:data[cell]['chemical']['y']=saved
    arrays=dict(names=names,genes=data[cell]['genes'],true=truth,pred=np.concatenate(pred),src=np.concatenate(source),post64=np.concatenate(post),z=state['z'].cpu().numpy().astype(np.float64),scale=np.asarray(scale),sources=np.asarray(sources))
    return arrays,q,state,gi


@torch.no_grad()
def predict_swapped(model,q,state,query_idx,new_x,scale):
    query={k:v[query_idx] for k,v in q.items()}
    query['x']=torch.as_tensor(new_x,device=query['x'].device,dtype=query['x'].dtype)
    parts=[]
    for lo in range(0,len(query_idx),128):
        qq={k:v[lo:lo+128] for k,v in query.items()}
        parts.append((model(qq,state,gain=1.).detach().float().cpu().numpy()*scale).astype(np.float64))
    return np.concatenate(parts)


def row_metrics(task,space,p0,cell,seed,truth,pred,names,n_anchor,n_query):
    values=score(truth,pred,names)
    return dict(task=task,space=space,p0=p0,cell=cell,seed=int(seed),n_anchor=int(n_anchor),n_query=int(n_query),**{k:values[k] for k in ('r20','recall20','mse20','mae20','r50','r100')})


def compare_reference_features(actual,path):
    with np.load(path,allow_pickle=False) as z:
        np.testing.assert_array_equal(actual['genes'],z['genes'].astype(str))
        lookup={str(n):i for i,n in enumerate(z['names'])};take=np.array([lookup[str(n)] for n in actual['names']])
        for field in ('n_anchor','n_query'):
            if not np.array_equal(joined[field+'_actual'],joined[field+'_reference']):raise ValueError(f'Finite-row counts differ for {field}')
        errors={k:float(np.max(np.abs(actual[k]-z[k][take]))) for k in ('true','src','pred','post64')}
    if errors['true']!=0 or errors['src']>1e-5 or errors['post64']>1e-4 or errors['pred']>1e-3:
        raise ValueError(f'Frozen feature path differs from the author reference: {errors}')
    return errors


def run(args):
    data=load_sources(args.data,args.device);base=load_base(args.representations)
    names,queries,support_idx,query_idx=validate_identity(base,data)
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[];baselines=[];checks=[]
    for cell in args.cells:
        for seed in args.seeds:
            model,scale,sources,checkpoint,metadata=load_paper_model(args.checkpoints,cell,seed,len(data[cell]['genes']),args.device)
            feat,q,state,genetic_order=extract_fold(model,data,cell,seed,names,scale,sources)
            spaces=fold_features(base,names,support_idx,query_idx,seed);spaces['post64']=feat['post64']
            check=dict(cell=cell,seed=int(seed),checkpoint_sha256=sha256(checkpoint),mapping=metadata['mapping'],genetic_order_sha256=hashlib.sha256(genetic_order.tobytes()).hexdigest())
            if args.reference_features is not None:
                if len(args.cells)!=1 or len(args.seeds)!=1:raise ValueError('--reference-features requires one cell and seed')
                check['reference_feature_max_abs_error']=compare_reference_features(feat,args.reference_features)
            baseline=row_metrics('frozen_baseline','unipert','original_x',cell,seed,feat['true'][query_idx],feat['pred'][query_idx],queries,0,len(query_idx));baselines.append(baseline)
            if args.tasks in ('both','ridge'):
                for space in RIDGE_SPACES:
                    if args.spaces and space not in args.spaces:continue
                    if space not in spaces:raise ValueError(f'Adopted representation cannot be constructed: {space}')
                    got=source_ridge(feat['src'],feat['true'],spaces[space],support_idx,query_idx,lam=1.)
                    if got is None:raise ValueError(f'Insufficient finite rows for adopted ridge probe: {space}')
                    pred,qi,n_a,n_q=got
                    rows.append(row_metrics('source_ridge',space,'source_mean',cell,seed,feat['true'][qi],pred,names[qi],n_a,n_q))
            if args.tasks in ('both','swap'):
                uni=spaces['unipert']
                for space in SWAP_SPACES:
                    if args.spaces and space not in args.spaces:continue
                    mat=spaces[space];new_x=project_x(mat[support_idx],uni[support_idx],mat[query_idx])
                    if new_x is None:raise ValueError(f'Insufficient finite rows for adopted frozen replacement: {space}')
                    keep=finite_rows(mat[query_idx]);mixed=uni[query_idx].copy();mixed[keep]=new_x[keep]
                    pred=predict_swapped(model,q,state,query_idx,mixed,scale)
                    rows.append(row_metrics('frozen_b_swap',space,'projected_x',cell,seed,feat['true'][query_idx],pred,queries,finite_rows(mat[support_idx]).sum(),keep.sum()))
            if args.save_features:
                folder=args.output/'features';folder.mkdir(exist_ok=True);np.savez_compressed(folder/f'{cell}_{seed}.npz',**feat)
            checks.append(check);print(f'probes {cell} seed{seed}: {len(rows)} adopted rows',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(args.output/'probe_long.csv',index=False)
    pd.DataFrame(baselines).to_csv(args.output/'frozen_baseline.csv',index=False)
    frozen=pd.concat([pd.DataFrame(baselines),frame[frame.task=='frozen_b_swap']],ignore_index=True)
    frozen.groupby(['space','cell']).r20.mean().unstack().reindex(index=('unipert',*SWAP_SPACES),columns=args.cells).to_csv(args.output/'FS2f_probe_means.csv')
    frame[frame.task=='source_ridge'].groupby(['space','cell']).r20.mean().unstack().reindex(index=RIDGE_SPACES,columns=args.cells).to_csv(args.output/'FS2g_probe_means.csv')
    if args.reference is not None:
        reference=pd.read_csv(args.reference);metrics=['r20','recall20','mse20','mae20','r50','r100'];keys=['task','space','p0','cell','seed']
        joined=frame.merge(reference,on=keys,suffixes=('_actual','_reference'),validate='one_to_one')
        if len(joined)!=len(frame):raise ValueError('Some adopted rows have no frozen comparison')
        for field in ('n_anchor','n_query'):
            if not np.array_equal(joined[field+'_actual'],joined[field+'_reference']):raise ValueError(f'Finite-row counts differ for {field}')
        errors={k:float(np.max(np.abs(joined[k+'_actual']-joined[k+'_reference']))) for k in metrics}
        checks.append({'frozen_metric_max_abs_error':errors,'reference_rows_compared':len(joined)})
        joined.to_csv(args.output/'reference_comparison.csv',index=False)
        if max(errors.values())>1e-5:raise ValueError(f'Probe metrics differ from the frozen author rows: {errors}')
    result=dict(n_rows=len(frame),n_cells=len(args.cells),seeds=args.seeds,n_shared=3220,n_ridge_support=2576,n_queries=644,ridge_lambda=1.,ridge_uses_measured_target_support_drugs=True,query_responses_used_only_for_scoring=True,PCA_fitted_only_on_support_rows=True,projection='ordinary least squares with a constant column; project replacement identities into frozen UniPert space',frozen_model_training=False,representations_sha256=sha256(args.representations),checks=checks)
    result['metric_reference_absolute_tolerance']=1e-5
    result['numpy_version']=np.__version__;result['torch_version']=torch.__version__
    (args.output/'verification.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/processed/sources'))
    p.add_argument('--checkpoints',type=Path,default=Path('data/checkpoints/benchmark'))
    p.add_argument('--representations',type=Path,default=Path('data/processed/diagnostics/representations.npz'))
    p.add_argument('--output',type=Path,default=Path('results/probes'))
    p.add_argument('--cells',nargs='+',choices=FIVE,default=list(FIVE));p.add_argument('--seeds',nargs='+',type=int,choices=SEEDS,default=list(SEEDS))
    p.add_argument('--tasks',choices=('both','ridge','swap'),default='both')
    p.add_argument('--spaces',nargs='+',choices=RIDGE_SPACES)
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--save-features',action='store_true')
    p.add_argument('--reference',type=Path,help='Optional frozen adopted probe_long.csv for numerical checks')
    p.add_argument('--reference-features',type=Path,help='Optional original feature NPZ for one-fold checks')
    a=p.parse_args()
    if len(set(a.cells))!=len(a.cells) or len(set(a.seeds))!=len(a.seeds):p.error('Cells and seeds must be unique')
    active=set(RIDGE_SPACES if a.tasks=='ridge' else SWAP_SPACES if a.tasks=='swap' else RIDGE_SPACES)
    if a.spaces and not (set(a.spaces)&active):p.error('No selected representation belongs to the requested task')
    torch.set_num_threads(a.threads);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    print(json.dumps(run(a),indent=2),flush=True)
if __name__=='__main__':main()
