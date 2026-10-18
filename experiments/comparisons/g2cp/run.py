"""Train the adopted pooled two-stage author-backbone G2CP D0 comparison.

Source and target genetic conditions are pooled for stage I, with source-only
validation. Stage II fine-tunes on source drugs only. No target query drug label
is used for fitting, graph construction or checkpoint selection.
"""
from pathlib import Path
import argparse,json,sys,time
from types import SimpleNamespace
import numpy as np
import torch
import pandas as pd
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))
from experiments.comparisons import data as D
from experiments.metrics import score
from experiments.comparisons.g2cp import helpers as H
from experiments.comparisons.g2cp.fast_forward import enable_fast_forward,fast_batch_for


def rows_for(data,cell,modality):
    pack=data[cell][modality];names=pack['names'].astype(str)
    x=pack['x'].cpu().numpy().astype(np.float32);y=pack['y'].cpu().numpy().astype(np.float32)
    good=np.isfinite(x).all(1)&(np.abs(x).sum(1)>0)&np.isfinite(y).all(1)
    ids=np.asarray([cell+'|'+modality+'|'+name for name in names[good]])
    return ids,x[good],y[good]


def join_rows(parts):
    return tuple(np.concatenate([part[i] for part in parts],axis=0) for i in range(3))


def pooled_split(n_sources,n_all,seed):
    rng=np.random.default_rng(seed);order=rng.permutation(n_sources)
    n=max(1,int(n_sources*.1));vi=order[:n]
    ti=np.concatenate([order[n:],np.arange(n_sources,n_all)])
    return ti,vi


def run(args):
    try:
        from experiments.comparisons.g2cp.author.model import GEARS_Model
        from experiments.comparisons.g2cp.author.utils import np_pearson_cor,GeneSimNetwork
    except ImportError as error:
        raise ImportError('G2CP requires torch-geometric and networkx; see this directory README.') from error
    torch.set_num_threads(args.threads);H.batch_for=fast_batch_for
    data,genes,queries=D.load(args.data,'internal');sources=[c for c in D.FIVE if c!=args.target]
    source_g=join_rows([rows_for(data,c,'genetic') for c in sources]);target_g=rows_for(data,args.target,'genetic')
    gn,gx,gy=join_rows([source_g,target_g]);dn,dx,dy=join_rows([rows_for(data,c,'chemical') for c in sources])
    names=queries[args.target];target=data[args.target]['chemical'];lookup={str(n):i for i,n in enumerate(target['names'])};qi=np.array([lookup[n] for n in names]);qx=target['x'][qi].numpy()
    qok=np.isfinite(qx).all(1)&(np.abs(qx).sum(1)>0)
    fallback=[]
    for name in names:
        available=[data[c]['chemical']['y'][np.flatnonzero(data[c]['chemical']['names']==name)[0]].numpy() for c in sources if name in data[c]['chemical']['names']]
        fallback.append(np.mean(np.stack(available),axis=0))
    fallback=np.stack(fallback)
    destination=args.output/args.target/str(args.seed);destination.mkdir(parents=True,exist_ok=True)
    if (destination/'COMPLETE.json').exists():raise FileExistsError(destination)
    H.seed_all(args.seed);gt,gv=pooled_split(len(source_g[0]),len(gn),args.seed);dt,dv=pooled_split(len(dn),len(dn),args.seed)
    if args.smoke:
        gt=np.concatenate([gt[gt<len(source_g[0])][:16],gt[gt>=len(source_g[0])][:16]])
        gv,dt,dv=gv[:8],dt[:32],dv[:8]
    assert not any(name.startswith(args.target+'|') for name in gn[gv])
    assert not any(name.startswith(args.target+'|') for name in dn)
    graph_y=np.concatenate([gy[gt],dy[dt]]);graph_names=np.concatenate([gn[gt],dn[dt]])
    edge,weight=H.graph_from_training(graph_y,genes,np_pearson_cor,GeneSimNetwork)
    config=SimpleNamespace(device=args.device,graph_batch_mode='corrected',batch_size=8 if args.smoke else 32,eval_batch_size=128,max_batches=1 if args.smoke else 0)
    model=H.make_model(config,GEARS_Model,edge,weight,genes,len(gn)+len(dn));enable_fast_forward(model)
    pre=H.train_stage(model,gx,gy,gt,gv,1 if args.smoke else 20,1e-3,5e-4,args.seed,config,'pooled_genetic_pretrain')
    model.load_state_dict(pre['last']);H.seed_all(args.seed)
    fine=H.train_stage(model,dx,dy,dt,dv,1 if args.smoke else 10,1e-4,5e-3,args.seed,config,'source_chemical_finetune')
    model.load_state_dict(fine['best']);torch.save(model.state_dict(),destination/'model.pt')
    evaluated=np.arange(min(8,len(names))) if args.smoke else np.arange(len(names))
    pred=np.zeros((len(evaluated),len(genes)),np.float32);ok=qok[evaluated]
    pred[ok]=H.predict(model,qx[evaluated][ok],128);pred[~ok]=fallback[evaluated][~ok]
    truth=target['y'][qi[evaluated]].numpy()
    metrics,_=score(truth,pred,names[evaluated])
    np.savez_compressed(destination/'predictions.npz',names=names[evaluated],genes=genes,pred=pred,truth=truth,source_mean=fallback[evaluated])
    np.savez_compressed(destination/'graph.npz',edge_index=edge.numpy(),edge_weight=weight.numpy())
    pd.DataFrame(pre['curves']+fine['curves']).to_csv(destination/'training_curves.csv',index=False)
    H.write_json(destination/'COMPLETE.json',{'complete':True,'method':'G2CP','configuration':'pooled_staged_d0','target':args.target,'seed':args.seed,'smoke':args.smoke,'sources':sources,'n_queries':len(evaluated),'target_genetic_used':True,'target_drug_training':False,'target_query_labels_used_for_selection':False,'checkpoint_selection':'source-only validation MSE','genetic_pretrain_epochs':1 if args.smoke else 20,'chemical_finetune_epochs':1 if args.smoke else 10,'finetune_starts_from':'last genetic epoch','graph_input_names':graph_names.tolist(),'graph_input_y_sha256':H.sha_array(graph_y),'genetic_training_ids':gn[gt].tolist(),'genetic_validation_ids':gn[gv].tolist(),'drug_training_ids':dn[dt].tolist(),'drug_validation_ids':dn[dv].tolist(),'parameters':sum(p.numel() for p in model.parameters()),'metrics':metrics})
    print(destination)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=REPO/'data/processed');p.add_argument('--target',choices=D.FIVE,required=True)
    p.add_argument('--seed',type=int,choices=[52,229,564],default=52);p.add_argument('--output',type=Path,default=REPO/'results/comparisons/G2CP')
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2);p.add_argument('--smoke',action='store_true')
    run(p.parse_args())
