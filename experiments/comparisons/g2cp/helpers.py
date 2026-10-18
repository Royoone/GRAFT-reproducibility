"""Final pooled G2CP training helpers, extracted without changing computations."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,random,time
import numpy as np
import pandas as pd
import torch

def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def sha_array(x):
    a = np.ascontiguousarray(x)
    h = hashlib.sha256(str(a.dtype).encode() + repr(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()

def write_json(p, d):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    def default(x):
        if isinstance(x, (np.integer, np.floating)): return x.item()
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x))
    Path(p).write_text(json.dumps(d, indent=2, ensure_ascii=False, default=default) + '\n')

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def graph_from_training(y, genes, pearson, graph_class):
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.abs(pearson(y, y))
    corr = np.nan_to_num(corr, nan=0.0)
    ix = np.argsort(corr, axis=1)[:, -21:]
    rows = []
    for i in range(len(genes)):
        for j in ix[i]:
            if corr[i, j] > .4:
                rows.append((str(genes[j]), str(genes[i]), float(corr[i, j])))
    df = pd.DataFrame(rows, columns=['source', 'target', 'importance'])
    if len(df):
        graph = graph_class(df, list(genes), dict(zip(genes, range(len(genes)))))
        return graph.edge_index, graph.edge_weight
    return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32)

def fix_graph_batching(model, mode):
    if mode != 'corrected': raise ValueError('Use the fixed independent-graph batching protocol')
    n = model.num_genes
    def hook(_layer, inputs):
        x, ei, ew = inputs
        b = len(x) // n
        if b <= 1: return inputs
        # Independent graphs per perturbation; only indices change, not parameters.
        offsets = torch.arange(b, device=ei.device) * n
        edges = (ei[:, None, :] + offsets[None, :, None]).reshape(2, -1)
        return x, edges, ew.repeat(b)
    for layer in model.layers_emb_pos:
        layer.register_forward_pre_hook(hook)

def make_model(args, author_cls, edge, weight, genes, n_perts):
    config = dict(num_genes=len(genes), num_perts=n_perts, hidden_size=64,
        uncertainty=False, decoder_hidden_size=16, num_gene_gnn_layers=1,
        no_perturb=False, G_coexpress=edge, G_coexpress_weight=weight, device=args.device)
    model = author_cls(config).to(args.device)
    fix_graph_batching(model, args.graph_batch_mode)
    return model

def batch_for(x, genes, device):
    # Matches author's list of one UniPert perturbagen per sample.
    return SimpleNamespace(x=torch.zeros((len(x) * genes, 1), device=device),
        unipert_rep=[[v.tolist()] for v in x],
        batch=torch.arange(len(x), device=device).repeat_interleave(genes))

def predict(model, x, batch_size=128):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            out.append(model(batch_for(x[start:start + batch_size], model.num_genes,
                       model.args['device'])).detach().cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, model.num_genes), np.float32)

def state_cpu(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def train_stage(model, x, y, ti, vi, epochs, lr, weight_decay, seed, args, stage):
    if len(ti) < 2 or len(vi) < 1: raise ValueError(f'{stage} needs >=2 train and >=1 validation')
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=.5)
    rng = np.random.default_rng(seed)
    best, best_state, best_epoch = np.inf, None, None
    curves = []
    t0 = time.time()
    effective_batch = min(args.batch_size, len(ti))
    # Like the author's train loader, discard incomplete trailing batches.
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(ti)
        losses, steps = [], 0
        start_time = time.time()
        for start in range(0, len(order) - effective_batch + 1, effective_batch):
            ii = order[start:start + effective_batch]
            yy = torch.as_tensor(y[ii], dtype=torch.float32, device=args.device)
            opt.zero_grad(set_to_none=True)
            yp = model(batch_for(x[ii], model.num_genes, args.device))
            # Exactly equivalent to author loss_fct for our unique-condition rows;
            # author's GEARS.dict_filter is all genes, and effect-space ctrl is 0.
            loss = (yp - yy).pow(4).mean() + .1 * (torch.sign(yy) - torch.sign(yp)).square().mean()
            if not torch.isfinite(loss): raise FloatingPointError(f'nonfinite {stage} loss')
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.)
            opt.step()
            losses.append(float(loss.detach()))
            steps += 1
            if args.max_batches and steps >= args.max_batches: break
        scheduler.step()
        val_pred = predict(model, x[vi], args.eval_batch_size)
        val_mse = float(np.mean((val_pred.astype(np.float64) - y[vi]) ** 2))
        if not np.isfinite(val_mse): raise FloatingPointError(f'nonfinite {stage} validation')
        if val_mse < best:
            best, best_state, best_epoch = val_mse, state_cpu(model), epoch + 1
        rec = dict(stage=stage, epoch=epoch + 1, train_loss=float(np.mean(losses)),
            val_mse=val_mse, steps=steps, effective_batch_size=effective_batch,
            next_lr=opt.param_groups[0]['lr'], seconds=time.time() - start_time)
        curves.append(rec)
        print(json.dumps(rec), flush=True)
    return dict(best=best_state, last=state_cpu(model), best_val_mse=best,
        best_epoch=best_epoch, curves=curves, seconds=time.time() - t0,
        effective_batch_size=effective_batch)
