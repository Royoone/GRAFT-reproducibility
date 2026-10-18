"""Adopted HyperMap loss, inference and support-only adaptation helpers."""
from pathlib import Path
import copy,hashlib,json,math,os
import numpy as np
import torch

CELLS=("A375","A549","HT29","MCF7","PC3")
OUTER_LR=5e-4
TRAIN_BATCH=512
PRED_BATCH=512
META_TASKS=4
INNER_BATCHES=5
INNER_STEPS=5
ADAPT_STEPS=3
ADAPT_INITIAL_LR=.01

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(tmp,path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def seed_all(seed):
    np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def features(d,ix,pdim):
    ids=d['hx'][ix]
    return torch.nn.functional.one_hot(ids,pdim).float()

def scalar_loss(model,x,target):
    pred=model(torch.zeros(len(x),target.shape[1],device=x.device),x)[0]
    return .5*(pred-target).square().sum(1).mean()

@torch.no_grad()
def predict(model,hx,pdim,genes=976):
    """Stable query order, max512 per batch; duplicate singleton tail then crop."""
    model.eval();parts=[]
    for start in range(0,len(hx),PRED_BATCH):
        ids=hx[start:start+PRED_BATCH];n=len(ids)
        if n==1:ids=ids.repeat(2)
        x=torch.nn.functional.one_hot(ids,pdim).float()
        p=model(torch.zeros(len(ids),genes,device=ids.device),x)[0][:n]
        if not torch.isfinite(p).all():raise FloatingPointError('nonfinite inference')
        parts.append(p.cpu().numpy())
    return np.concatenate(parts)

def safe_adapt(model,x,truth,seed):
    """Same three stochastic-gradient / deterministic-support-loss steps as31.

    Restore random state so source validation cannot alter later training draws.
    """
    model=copy.deepcopy(model)
    log=[];device=x.device
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        for step in range(ADAPT_STEPS):
            model.eval()
            with torch.no_grad():before=float(scalar_loss(model,x,truth))
            if not math.isfinite(before):raise FloatingPointError('nonfinite pre-adaptation support loss')
            model.train();model.zero_grad(set_to_none=True)
            loss=scalar_loss(model,x,truth)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite adaptation gradient loss')
            loss.backward()
            params=list(model.parameters())
            original=[p.detach().clone() for p in params]
            grads=[p.grad.detach().clone() for p in params]
            if not all(bool(torch.isfinite(g).all()) for g in grads):
                raise FloatingPointError('nonfinite adaptation gradient')
            model.eval();rate=ADAPT_INITIAL_LR;accepted=False
            for attempt in range(30):
                with torch.no_grad():
                    for p,v,g in zip(params,original,grads):p.copy_(v-rate*g)
                    after=float(scalar_loss(model,x,truth))
                if math.isfinite(after) and after<=before+1e-6:
                    accepted=True;break
                rate*=.5
            if not accepted:
                with torch.no_grad():
                    for p,v in zip(params,original):p.copy_(v)
                rate=0.;after=before
            log.append({'step':step+1,'before':before,'after':after,'rate':rate,
                        'accepted':accepted,'backtracks':attempt})
            del original,grads,params,loss
    model.eval()
    return model,log
