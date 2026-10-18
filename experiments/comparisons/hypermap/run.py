"""Train the adopted fixed-source HyperMap model and evaluate D0/D20.

The per-fold inner learning rate and training duration are frozen in
fixed_protocol.json. Target responses are read only after fitting is complete.
No source-selection search is repeated by this entry point.
"""
from pathlib import Path
import argparse,hashlib,json,math,sys
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[3];sys.path.insert(0,str(REPO))
from experiments.comparisons.hypermap import helpers as H
from experiments.comparisons.hypermap.fit import fit,predict_target
from experiments.metrics import score


def source_data(args,protocol,dest):
    data={};vocabulary=set();genes=None;squares=[];sources=protocol['sources'];files={}
    for cell in H.CELLS:
        path=args.data/'sources'/f'{cell}.npz'
        with np.load(path,allow_pickle=False) as archive:
            panel=archive['genes'].astype(str)
            if genes is None:genes=panel
            np.testing.assert_array_equal(genes,panel)
            names=archive['chemical_names'].astype(str);vocabulary.update(names)
            data[cell]={'names':names}
            if cell in sources:
                # Target response arrays are deliberately never accessed here.
                y=archive['chemical_native'].astype(np.float32).copy()
                if not np.isfinite(y).all():raise ValueError('Nonfinite source response')
                data[cell]['truth']=y;squares.append(float(np.mean(y.astype(np.float64)**2)))
        files[cell]=H.sha(path)
    assert 'truth' not in data[args.target]
    vocab=np.asarray(sorted(vocabulary));lookup={n:i for i,n in enumerate(vocab)}
    if hashlib.sha256('\n'.join(vocab).encode()).hexdigest()!=protocol['vocabulary_sha256']:
        raise ValueError('The source drug-identity vocabulary differs from the frozen protocol')
    scale=float(np.sqrt(np.mean(squares)))
    for cell in sources:
        d=data[cell];d['ti']=np.arange(len(d['names']));d['hx']=torch.tensor([lookup[n] for n in d['names']],device=args.device);d['y']=torch.tensor(d['truth']/scale,device=args.device)
    query=np.asarray(protocol['query_names']);anchors=np.asarray(protocol['anchor_names'])
    assert len(query)==644 and len(anchors)==20 and not set(query)&set(anchors)
    train_union=set().union(*(set(data[cell]['names']) for cell in sources))
    assert set(query)<=train_union
    H.write(dest/'input_manifest.json',{'sources':sources,'source_data_sha256':files,'source_scale':scale,'target_responses_materialized_before_fit':False,'query_names':query.tolist(),'anchor_names':anchors.tolist(),'source_selection_sha256':protocol['selection_sha256']})
    return data,sources,genes,vocab,scale,query,anchors


def run(args):
    from experiments.comparisons.hypermap.author_model import Net
    from experiments.comparisons.hypermap import author_model
    torch.set_num_threads(args.threads)
    if str(args.device).startswith('cuda'):torch.cuda.set_device(torch.device(args.device))
    torch.backends.cudnn.benchmark=False
    record=json.loads((Path(__file__).parent/'fixed_protocol.json').read_text())['folds'][args.target+'_'+str(args.seed)]
    args.selection_sha256=record['selection_sha256']
    destination=args.output/args.target/str(args.seed);destination.mkdir(parents=True,exist_ok=True)
    if (destination/'FOLD_COMPLETE.json').exists():raise FileExistsError(destination)
    data,sources,genes,vocab,scale,query,anchors=source_data(args,record,destination)
    chosen={'inner_lr':record['inner_lr'],'best_epoch':record['fixed_epochs'],'best_source_score':record['source_selection_score']}
    if args.smoke:
        H.TRAIN_BATCH=4;H.META_TASKS=1;H.INNER_BATCHES=1;H.INNER_STEPS=1;chosen['best_epoch']=1
        query=query[:8]
    H.write(destination/'fixed_protocol.json',{**record,'smoke':args.smoke})
    model,result=fit(args,Net,Path(author_model.__file__),data,sources,len(vocab),scale,chosen,destination)
    if model is None:raise RuntimeError('Nonfinite fit; see FOLD_FAILED.json; no fallback or rescue is applied')
    predict_target(args,model,genes,vocab,scale,query,anchors,result,destination)
    for budget in [0,20]:
        with np.load(destination/f'D{budget}/predictions.npz',allow_pickle=False) as archive:
            metrics,_=score(archive['truth'],archive['pred'],archive['names'].astype(str))
        H.write(destination/f'D{budget}/scores.json',{'method':f'HyperMap-D{budget}','smoke':args.smoke,'metrics':metrics})
    print(destination)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=REPO/'data/processed');p.add_argument('--target',choices=H.CELLS,required=True)
    p.add_argument('--seed',type=int,choices=[52,229,564],default=52);p.add_argument('--output',type=Path,default=REPO/'results/comparisons/HyperMap')
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2);p.add_argument('--smoke',action='store_true')
    run(p.parse_args())
