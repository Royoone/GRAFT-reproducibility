"""Internal benchmark fitting, using the fixed manuscript configurations."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib,json,random,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
class XPertAdapter(nn.Module):
    """Attend from drug features to the shared source-response proxy.

    The readout combines the attention output, drug query and response proxy.
    """
    def __init__(self, genes=976):
        super().__init__()
        self.pre = nn.Sequential(nn.Linear(genes, 256), nn.GELU(), nn.LayerNorm(256))
        self.drug = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.LayerNorm(256))
        self.cross = nn.MultiheadAttention(256, 8, batch_first=True)
        self.out = nn.Linear(256, genes)

    def forward(self, fp, ctrl):
        q = self.drug(fp).unsqueeze(1)
        k = self.pre(ctrl).unsqueeze(1)
        h, _ = self.cross(q, k, k, need_weights=False)
        return self.out(h.squeeze(1) + q.squeeze(1) - k.squeeze(1))

class Features:
    def __init__(self, recs): self.recs, self.cache = recs, {}
    def get(self, row, kind):
        if kind == 'unipert':
            x = row['chem_emb'].astype(np.float32)
            return x, np.isfinite(x).all(1) & (np.abs(x).sum(1)>0)
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import rdFingerprintGenerator
        RDLogger.DisableLog('rdApp.warning')
        nbits = int(kind); generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=nbits)
        output, valid = [], []
        for name in row['names']:
            key = (str(name), nbits)
            if key not in self.cache:
                s = self.recs.get(str(name)); m = Chem.MolFromSmiles(s) if s else None
                a = np.zeros(nbits, dtype=np.float32)
                if m is not None: DataStructs.ConvertToNumpyArray(generator.GetFingerprint(m), a)
                self.cache[key] = (a, m is not None)
            a, ok = self.cache[key]; output.append(a); valid.append(ok)
        return np.stack(output), np.asarray(valid, dtype=bool)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(x):
    a = np.ascontiguousarray(x)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda v: v.item() if isinstance(v, np.generic) else v.tolist())+'\n')
    temp.replace(path)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class AttentionAdapter(nn.Module):
    def __init__(self, original):
        super().__init__(); self.drug=original.drug; self.cell=original.cell
        self.gene=original.gene; self.mix=original.mix
    def forward(self, x, cell):
        d, c = self.drug(x), self.cell(cell)
        weights = torch.softmax(d @ self.gene.T/8.0, dim=-1)
        context = weights @ self.gene
        d = d + context
        g = self.gene.unsqueeze(0).expand(len(x), -1, -1)
        return self.mix(torch.cat((d.unsqueeze(1).expand_as(g),
            c.unsqueeze(1).expand(-1, g.size(1), -1), g), dim=-1)).squeeze(-1)


class FingerprintMLP(nn.Module):
    def __init__(self, dim, genes, ncell):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim+ncell,512),nn.ReLU(),nn.Dropout(.1),
            nn.Linear(512,256),nn.ReLU(),nn.Linear(256,genes))
    def forward(self, x, cell): return self.net(torch.cat((x,cell),-1))


def feature_kind(method):
    return 'unipert' if method=='mlp_unipert' else '2048' if method=='morgan2048' else '256'


def make_model(method, genes, ncell, W, args=None, gene_names=None, model_dir=None):
    if method=='xpert_dual_drug_residual': return XPertAdapter(genes)
    if method in ('deepce_attention','ciger_attention','ciger_ranknet'):
        return AttentionAdapter(W.DeepCENet(genes,ncell))
    if method=='prnet_repro': return W.PRnetNet(genes)
    if method=='transigen_repro': return W.TranSiGenNet(genes)
    if method=='chemcpa_repro': return W.ChemCPANet(genes,ncell)
    if method in ('morgan256','morgan2048','mlp_unipert'):
        return FingerprintMLP(2048 if method=='morgan2048' else 256, genes, ncell)
    raise ValueError('Unknown manuscript comparison configuration: '+method)


def forward(model, method, x, cell, basal):
    if method=='xpert_dual_drug_residual': return model(x,basal),None,None
    if method=='transigen_repro': return model(x,basal)
    if method=='prnet_repro': return model(x,basal),None,None
    if method=='chemcpa_repro': return model(x,basal,cell),None,None
    return model(x,cell),None,None


def ranknet_mean(pred, truth, pair_chunk=16384):
    """Exact all-unordered-pair mean, mathematically matching official RankNet/N.

    No random pair sampling. Chunking reduces temporary tensor size; torch's
    autograd retains only the pairwise terms needed for this loss.
    """
    i,j = torch.triu_indices(pred.shape[1],pred.shape[1],offset=1,device=pred.device)
    total = pred.new_zeros(())
    for begin in range(0,len(i),pair_chunk):
        a,b=i[begin:begin+pair_chunk],j[begin:begin+pair_chunk]
        difference=pred[:,a]-pred[:,b]
        direction=torch.sign(truth[:,a]-truth[:,b])
        total=total+(.5*(1-direction)*difference+F.softplus(-difference)).sum()
    return total/(len(pred)*len(i))


def loss_fn(pred, truth, method, W, mu=None, lv=None):
    mse=(pred-truth).square().mean()
    if method=='ciger_attention': return mse+.5*W.rank_loss(pred,truth)
    if method=='ciger_ranknet': return mse+.5*ranknet_mean(pred,truth)
    if method=='transigen_repro':
        return mse+.1*(-.5*(1+lv-mu.square()-lv.exp()).mean())
    return mse


def source_pool(data, sources, method, feats):
    xx,yy,cc,labels=[],[],[],[]
    coverage={}
    for index,cell in enumerate(sources):
        row=data[cell];x,ok=feats.get(row,feature_kind(method))
        # Three representation-only MLP probes share the intersection of valid
        # ECFP and UniPert source rows, removing training-coverage confounding.
        if method in ('morgan256','morgan2048','mlp_unipert'):
            _,molok=feats.get(row,'256');_,uniok=feats.get(row,'unipert');ok &= molok & uniok
        xx.append(x[ok]);yy.append(row['y'][ok]);cc.append(np.full(int(ok.sum()),index,np.int64))
        labels.extend([cell+'::'+str(n) for n in row['names'][ok]])
        coverage[cell]={'available':len(ok),'used':int(ok.sum())}
    return np.concatenate(xx),np.concatenate(yy),np.concatenate(cc),np.asarray(labels),coverage


def fit(args, data, sources, method, seed, feats, W, model_dir, genes):
    seed_all(seed);rng=np.random.default_rng(seed)
    x,y,c,labels,coverage=source_pool(data,sources,method,feats)
    perm=rng.permutation(len(y))
    nval=max(256,len(y)//10)
    if nval>=len(y): raise ValueError('Insufficient source training rows')
    vi,ti=perm[:nval],perm[nval:]
    if args.smoke_max_train_rows:ti=ti[:args.smoke_max_train_rows]
    basal=y[ti].mean(0).astype(np.float32)
    model=make_model(method,y.shape[1],len(sources),W,args,genes,model_dir).to(args.device)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    contract={'cohort':args.cohort,'method':method,'seed':seed,'sources':list(sources),
        'input':feature_kind(method),'n_train':len(ti),'n_validation':len(vi),
        'max_epochs':args.max_epochs,'patience':args.patience,'batch_size':args.batch_size,
        'initial_epoch_cap':args.initial_epochs,
        'extension_rule':'Every run reaching 40 epochs continues to at most 80 with unchanged source-validation patience; never selected on target scores',
        'lr':args.lr,'gradient_clip':1.,'selection':'source-validation MSE',
        'basal':'source-training rows only; constant response-space proxy',
        'source_coverage':coverage,'train_row_ids':labels[ti].tolist(),'validation_row_ids':labels[vi].tolist(),
        'basal_sha256':array_sha(basal),'source_response_sha256':array_sha(y),
        'runner_sha256':sha(__file__),'wp2_source_sha256':sha(Path(__file__).with_name('models.py')),
        'smoke':bool(args.smoke_max_train_rows),'model_parameters':sum(p.numel() for p in model.parameters()),
        'torch_version':torch.__version__,'numpy_version':np.__version__,
        'deterministic_algorithms':True,'attention_intervention':'d + softmax(d @ gene.T / 8) @ gene' if method in ('deepce_attention','ciger_attention','ciger_ranknet') else None,
        'ranknet':'MSE + 0.5 * official pair expression averaged over every upper-triangle gene pair; stable softplus' if method=='ciger_ranknet' else None}
    if method=='xpert_dual_drug_residual': contract['xpert_module_sha256']=sha(__file__)
    model_dir.mkdir(parents=True,exist_ok=True)
    write_json(model_dir/'training_contract.json',contract)
    np.savez_compressed(model_dir/'split.npz',train_indices=ti,validation_indices=vi,row_ids=labels,basal=basal)
    basal_t=torch.as_tensor(basal,device=args.device)
    def tensors(indices):
        return (torch.as_tensor(x[indices],device=args.device),torch.as_tensor(y[indices],device=args.device),
            F.one_hot(torch.as_tensor(c[indices],device=args.device),num_classes=len(sources)).float(),
            basal_t.unsqueeze(0).expand(len(indices),-1))
    best=float('inf');best_epoch=0;wait=0;history=[];started=time.monotonic();steps=0
    # RankNet keeps optimizer batch size 256 but uses row microbatches; weighted
    # gradient accumulation is the exact batch mean, avoiding G^2 * 256 storage.
    microbatch=min(args.batch_size,16) if method=='ciger_ranknet' else args.batch_size
    for epoch in range(1,args.max_epochs+1):
        model.train();rng.shuffle(ti);train_total=0.;seen=0
        for begin in range(0,len(ti),args.batch_size):
            batch=ti[begin:begin+args.batch_size];opt.zero_grad(set_to_none=True)
            for mbegin in range(0,len(batch),microbatch):
                sl=batch[mbegin:mbegin+microbatch];xx,yy,cc,bb=tensors(sl)
                pred,mu,lv=forward(model,method,xx,cc,bb);loss=loss_fn(pred,yy,method,W,mu,lv)
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss {method} epoch={epoch}')
                (loss*(len(sl)/len(batch))).backward()
                train_total+=float(loss.detach())*len(sl);seen+=len(sl)
            nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();steps+=1
        model.eval();squared_error=0.;elements=0
        with torch.no_grad():
            evbatch=args.eval_batch_size
            for begin in range(0,len(vi),evbatch):
                xx,yy,cc,bb=tensors(vi[begin:begin+evbatch])
                pred,_,_=forward(model,method,xx,cc,bb)
                squared_error+=float((pred-yy).double().square().sum());elements+=yy.numel()
        val=squared_error/elements
        if not np.isfinite(val):raise FloatingPointError('Nonfinite validation')
        improved=val<best
        if improved:
            best=val;best_epoch=epoch;wait=0
            torch.save({'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                'method':method,'contract':contract,'epoch':epoch,'validation_mse':val},model_dir/'checkpoint.pt')
        else:wait+=1
        history.append(dict(epoch=epoch,train_objective=train_total/seen,validation_mse=val,
            improved=improved,optimizer_steps=steps,elapsed_seconds=time.monotonic()-started))
        pd.DataFrame(history).to_csv(model_dir/'history.csv',index=False)
        if epoch==args.initial_epochs:
            import shutil
            shutil.copy2(model_dir/'checkpoint.pt',model_dir/'checkpoint_at_initial_cap.pt')
        print(f'{method} sources={",".join(sources)} seed={seed} epoch={epoch} train={train_total/seen:.6g} val={val:.6g} best={best_epoch}',flush=True)
        if wait>=args.patience:break
    checkpoint=torch.load(model_dir/'checkpoint.pt',map_location=args.device,weights_only=False)
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    status={**contract,'best_epoch':best_epoch,'best_validation_mse':best,
        'epochs_completed':len(history),'optimizer_steps':steps,'hit_epoch_cap':len(history)==args.max_epochs,
        'reached_initial_cap':len(history)>=args.initial_epochs,'extension_used':len(history)>args.initial_epochs,
        'best_at_cap':best_epoch==args.max_epochs,'stop_reason':'patience' if wait>=args.patience else 'epoch_cap',
        'training_seconds':time.monotonic()-started,'checkpoint_sha256':sha(model_dir/'checkpoint.pt')}
    write_json(model_dir/'TRAIN_COMPLETE.json',status)
    return model,basal,status


def source_mean(data,sources,names):
    maps={c:{str(n):i for i,n in enumerate(data[c]['names'])} for c in sources}
    fallback=np.stack([data[c]['y'].mean(0) for c in sources]).mean(0)
    out=[]
    for n in names:
        rows=[data[c]['y'][maps[c][str(n)]] for c in sources if str(n) in maps[c]]
        out.append(np.stack(rows).mean(0) if rows else fallback)
    return np.stack(out).astype(np.float32)


def predict(args,model,basal,data,sources,target,names,method,seed,feats,genes,status,out):
    # Only after fitting do we materialize target query responses for scoring.
    lookup={str(n):i for i,n in enumerate(data[target]['names'])}
    qi=np.asarray([lookup[str(n)] for n in names],dtype=np.int64)
    row={'names':np.asarray(names,dtype=str),'chem_emb':data[target]['chem_emb'][qi]}
    x,encoded=feats.get(row,feature_kind(method));sm=source_mean(data,sources,names)
    pred=np.empty_like(sm);model.eval()
    with torch.no_grad():
        evbatch=args.eval_batch_size
        for begin in range(0,len(names),evbatch):
            xx=torch.as_tensor(np.nan_to_num(x[begin:begin+evbatch]),device=args.device)
            cell=torch.full((len(xx),len(sources)),1./len(sources),device=args.device)
            bb=torch.as_tensor(basal,device=args.device).unsqueeze(0).expand(len(xx),-1)
            p,_,_=forward(model,method,xx,cell,bb)
            pred[begin:begin+len(xx)]=p.float().cpu().numpy()
    pred=np.where(encoded[:,None],pred,sm)
    if not np.isfinite(pred).all():raise FloatingPointError('Nonfinite prediction')
    truth=data[target]['y'][qi]
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'predictions.npz',names=names,genes=genes,truth=truth,pred=pred,
        encoded_mask=encoded,source_mean=sm)
    record={'complete':True,'cohort':args.cohort,'variant':method,'target':target,'seed':seed,
        'n_queries':len(names),'query_names_sha256':array_sha(np.asarray(names,dtype=str)),
        'truth_sha256':array_sha(truth),'prediction_sha256':array_sha(pred),
        'encoded_count':int(encoded.sum()),'encoded_fraction':float(encoded.mean()),
        'sources':list(sources),'training':status,'target_drug_labels_used_in_training':0,
        'target_genetic_labels_used_in_training':0}
    write_json(out/'COMPLETE.json',record)

