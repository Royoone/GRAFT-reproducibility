"""Train one fixed manuscript chemical comparator configuration."""
from pathlib import Path
from types import SimpleNamespace
import argparse,json
import numpy as np
import torch
from . import data as D
from . import internal as I
from . import models as W
from . import external as E
from . import external_extra as X
from experiments.metrics import score

INTERNAL={'DeepCE':'deepce_attention','CIGER-hinge':'ciger_attention','CIGER-RankNet':'ciger_ranknet',
    'PRnet':'prnet_repro','TranSiGen':'transigen_repro','chemCPA':'chemcpa_repro','Morgan-256':'morgan256',
    'Morgan-2048':'morgan2048','UniPert-MLP':'mlp_unipert','XPert':'xpert_dual_drug_residual',
    'G2CP-adapter':'g2cp_adapter'}
EXTERNAL={'DeepCE':'deepce','CIGER':'ciger','TranSiGen':'transigen','PRnet':'prnet','chemCPA':'chemcpa',
    'Morgan-256':'mlp_morgan','XPert':'xpert_dual','G2CP':'g2cp_d0'}


def sample_d10pct_names(data, target, queries, seed):
    """Official G2CP-style 10% of target chemicals, queries excluded."""
    qset = set(map(str, queries))
    pool = [str(n) for n in data[target]["names"] if str(n) not in qset]
    k = max(1, int(round(0.1 * len(data[target]["names"]))))
    k = min(k, len(pool))
    rng = np.random.default_rng(int(seed))
    return [str(x) for x in rng.choice(np.asarray(pool, dtype=object), size=k, replace=False)]


def budget_anchors(data, target, queries, seed, budget, data_root):
    if budget == 'D0':
        return []
    if budget == 'D10pct':
        return sample_d10pct_names(data, target, queries, seed)
    path = Path(data_root) / 'comparisons/anchors20.json'
    records = json.loads(path.read_text())['anchors20']
    matching = [row['anchors'] for row in records
                if row['target'] == target and int(row['seed']) == int(seed)]
    if len(matching) != 1:
        raise ValueError(f'No unique frozen 20-anchor list for {target}, seed {seed}')
    anchors = matching[0]
    if len(anchors) != 20 or set(anchors) & set(queries):
        raise ValueError('Invalid frozen 20-anchor list')
    return list(map(str, anchors))


def smoke_g2cp_adapter(data, target, names, extra_names, device):
    """Only a forward/loss check on allowed real rows; not a reduced paper fit."""
    source = next(cell for cell in D.FIVE if cell != target)
    context = {cell: data[cell]['gen_y'].mean(0).astype(np.float32) for cell in D.FIVE}
    x = [data[source]['chem_emb'][:4]]
    y = [data[source]['y'][:4]]
    ctx = [np.repeat(context[source][None], 4, 0)]
    if extra_names:
        lookup = {str(name): i for i, name in enumerate(data[target]['names'])}
        indices = [lookup[name] for name in extra_names[:4]]
        x.append(data[target]['chem_emb'][indices]); y.append(data[target]['y'][indices])
        ctx.append(np.repeat(context[target][None], len(indices), 0))
    x, y, ctx = [torch.tensor(np.concatenate(values), device=device) for values in (x, y, ctx)]
    model = W.G2CPD0Net(y.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=W.LR)
    model.train(); optimizer.zero_grad(set_to_none=True)
    loss = (model(x, ctx)-y).square().mean()
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite G2CP-adapter smoke loss')
    # Query labels are read only after this single diagnostic update.
    names = np.asarray(names[:8]); lookup = {str(name): i for i, name in enumerate(data[target]['names'])}
    indices = [lookup[name] for name in names]
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(data[target]['chem_emb'][indices], device=device),
                     torch.tensor(np.repeat(context[target][None], len(indices), 0), device=device)).cpu().numpy()
    return pred, data[target]['y'][indices], names, float(loss.detach()), sum(p.numel() for p in model.parameters())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/processed'))
    p.add_argument('--annotations',type=Path,default=Path('data/annotations/compounds.csv'))
    p.add_argument('--cohort',choices=['internal','external'],required=True)
    p.add_argument('--method',required=True)
    p.add_argument('--target',required=True)
    p.add_argument('--seed',type=int,default=52)
    p.add_argument('--output',type=Path,default=Path('results/comparisons'))
    p.add_argument('--device',default='cpu')
    p.add_argument('--budget',choices=['D0','D20','D10pct'],default='D0',
                   help='Target drug budget for internal G2CP-adapter only')
    p.add_argument('--smoke',action='store_true',help='One epoch; explicitly not a paper result')
    a=p.parse_args()
    methods=INTERNAL if a.cohort=='internal' else EXTERNAL
    if a.method not in methods:p.error('Use one of: '+', '.join(methods))
    allowed=D.FIVE if a.cohort=='internal' else D.EXTERNAL
    if a.target not in allowed:p.error('Target is not in this benchmark')
    if a.budget != 'D0' and not (a.cohort == 'internal' and a.method == 'G2CP-adapter'):
        p.error('--budget applies only to internal G2CP-adapter')
    torch.set_num_threads(2);I.seed_all(a.seed)
    rows,genes,queries=D.load(a.data,a.cohort)
    sources=[c for c in D.FIVE if c!=a.target]
    recs={} if a.method=='G2CP-adapter' else D.smiles(a.annotations); method=methods[a.method]
    dest=a.output/a.cohort/a.method/a.target/str(a.seed)
    if a.method=='G2CP-adapter':dest=dest/a.budget
    if (dest/'predictions.npz').exists():raise FileExistsError(dest)
    if a.cohort=='internal':
        data=D.array_view(rows)
        if a.method=='G2CP-adapter':
            names=queries[a.target]
            anchors=budget_anchors(data,a.target,names,a.seed,a.budget,a.data)
            if set(anchors)&set(names):raise ValueError('Anchor/query overlap')
            if a.smoke:
                pred,truth,names,loss,nparams=smoke_g2cp_adapter(data,a.target,names,anchors,a.device)
                used=anchors
            else:
                tag={'D0':'g2cp_d0','D20':'g2cp_d20','D10pct':'g2cp_d10pct'}[a.budget]
                pred,truth,coverage,seconds,nparams,used=W.train_g2cp(data,a.target,names,a.device,a.seed,
                    extra_names=anchors,tag=tag)
            fallback=[]
            for name in names:
                values=[data[cell]['y'][np.flatnonzero(data[cell]['names']==name)[0]] for cell in sources
                        if name in data[cell]['names']]
                fallback.append(np.mean(np.stack(values),axis=0))
            dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/'predictions.npz',names=names,genes=genes,pred=pred,truth=truth,
                                source_mean=np.asarray(fallback),anchor_names=np.asarray(used,dtype=str))
            metrics,_=score(truth,pred,names)
            record={'method':a.method,'cohort':a.cohort,'target':a.target,'seed':a.seed,'budget':a.budget,
                'smoke':a.smoke,'smoke_is_forward_loss_diagnostic':a.smoke,'metrics':metrics,
                'n_parameters':nparams,'anchor_names':used,'query_target_labels_used_for_training':False,
                'configuration':'graph-free pooled allowed responses, independently trained from scratch'}
            if a.smoke:record['diagnostic_training_loss']=loss
            (dest/'scores.json').write_text(json.dumps(record,indent=2)+'\n')
            print(dest)
            return
        args=SimpleNamespace(cohort='internal',max_epochs=1 if a.smoke else 80,patience=8,
            initial_epochs=40,batch_size=256,eval_batch_size=128,lr=.001,device=a.device,
            smoke_max_train_rows=256 if a.smoke else 0)
        feats=I.Features(recs)
        model,basal,status=I.fit(args,data,sources,method,a.seed,feats,W,dest/'training',genes)
        I.predict(args,model,basal,data,sources,a.target,queries[a.target],method,a.seed,
                  feats,genes,status,dest)
    else:
        names=queries[a.target]
        if method in ['ciger','transigen','chemcpa']:
            fitted=X.train_extra(method,rows,sources,recs,a.device,a.seed,max_epochs=1 if a.smoke else 40)
            model,src,mean,_,predict,_=fitted
            pred,truth,_,ok=E.infer_chem(model,predict,rows,a.target,names,recs,src,mean,a.device)
        elif method=='g2cp_d0':
            model,ctx,_=E.train_g2cp(rows,a.target,sources,names,a.device,a.seed,max_epochs=1 if a.smoke else 40,max_n=256 if a.smoke else None)
            pred,truth,_,ok=E.infer_g2cp(model,rows,a.target,names,ctx,a.device)
        else:
            fitted=E.train_chem(method,rows,a.target,sources,names,recs,a.device,a.seed,max_epochs=1 if a.smoke else 40,max_n=256 if a.smoke else None)
            model,src,mean,_,predict=fitted
            pred,truth,_,ok=E.infer_chem(model,predict,rows,a.target,names,recs,src,mean,a.device)
        # The source same-drug mean is the declared unencoded-query fallback.
        fallback=[]
        for n in names:
            y=[rows[c]['chemical']['y'][np.flatnonzero(rows[c]['chemical']['names']==n)[0]].numpy()
               for c in sources if n in rows[c]['chemical']['names']]
            fallback.append(np.mean(y,axis=0))
        pred=np.where(ok[:,None],pred,np.asarray(fallback))
        dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/'predictions.npz',names=names,genes=genes,pred=pred,truth=truth,source_mean=np.asarray(fallback))
        torch.save({'state_dict':model.state_dict(),'method':a.method,'seed':a.seed,'sources':sources},dest/'model.pt')
    with np.load(dest/'predictions.npz',allow_pickle=False) as z:
        metrics,_=score(z['truth'],z['pred'],z['names'].astype(str))
    (dest/'scores.json').write_text(json.dumps({'method':a.method,'cohort':a.cohort,'target':a.target,
        'seed':a.seed,'smoke':a.smoke,'metrics':metrics},indent=2)+'\n')
    print(dest)


if __name__=='__main__':main()
