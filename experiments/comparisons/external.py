"""External benchmark comparator fitting under its fixed task protocol."""
from __future__ import annotations
import time
import re
import numpy as np
import torch
from torch import nn
def compact(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


class DeepCENet(nn.Module):
    def __init__(self, genes, n_cell=4):
        super().__init__()
        self.drug = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1))
        self.cell = nn.Linear(n_cell, 32)
        self.gene = nn.Parameter(torch.randn(genes, 128) * 0.02)
        self.mix = nn.Sequential(nn.Linear(128 + 32 + 128, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, fp, cell):
        d = self.drug(fp)
        c = self.cell(cell)
        g = self.gene.unsqueeze(0).expand(len(fp), -1, -1)
        return self.mix(torch.cat((d.unsqueeze(1).expand_as(g), c.unsqueeze(1).expand(-1, g.size(1), -1), g), -1)).squeeze(-1)


class PRnetNet(nn.Module):
    def __init__(self, genes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(256 + genes, 512), nn.GELU(), nn.LayerNorm(512), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, genes))

    def forward(self, fp, unpert):
        return self.net(torch.cat((fp, unpert), -1))


class MLPMorgan(nn.Module):
    def __init__(self, genes, n_cell=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(256 + n_cell, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, genes))

    def forward(self, fp, cell):
        return self.net(torch.cat((fp, cell), -1))


class XPertDual(nn.Module):
    def __init__(self, genes):
        super().__init__()
        self.pre = nn.Sequential(nn.Linear(genes, 256), nn.GELU(), nn.LayerNorm(256))
        self.drug = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.LayerNorm(256))
        self.cross = nn.MultiheadAttention(256, 8, batch_first=True)
        self.out = nn.Linear(256, genes)

    def forward(self, fp, ctrl):
        q = self.drug(fp).unsqueeze(1)
        k = self.pre(ctrl).unsqueeze(1)
        h, _ = self.cross(q, k, k, need_weights=False)
        return self.out(h.squeeze(1) - k.squeeze(1))


class G2CPD0Net(nn.Module):
    def __init__(self, genes, hidden=64):
        super().__init__()
        self.unipert_emb_trans = nn.Sequential(nn.Linear(256, hidden), nn.ReLU())
        self.gene_emb = nn.Embedding(genes, hidden)
        self.emb_trans = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.pert_fuse = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.ctx = nn.Linear(genes, hidden)
        self.recovery = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, hidden))
        self.indv_w1 = nn.Parameter(torch.randn(genes, hidden) * 0.02)
        self.indv_b1 = nn.Parameter(torch.zeros(genes))
        self.cross = nn.Sequential(nn.Linear(genes, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.indv_w2 = nn.Parameter(torch.randn(genes, hidden + 1) * 0.02)
        self.indv_b2 = nn.Parameter(torch.zeros(genes))

    def forward(self, pert, ctx):
        bsz = pert.size(0)
        n_g = self.gene_emb.num_embeddings
        base = self.emb_trans(self.gene_emb.weight).unsqueeze(0).expand(bsz, -1, -1)
        p = self.pert_fuse(torch.nn.functional.normalize(self.unipert_emb_trans(pert), dim=-1))
        h = self.recovery(base + p.unsqueeze(1) + 0.2 * self.ctx(ctx).unsqueeze(1))
        out = (h * self.indv_w1.unsqueeze(0)).sum(-1) + self.indv_b1
        cat = torch.cat((out.unsqueeze(-1), self.cross(out).unsqueeze(1).expand(-1, n_g, -1)), -1)
        return (cat * self.indv_w2.unsqueeze(0)).sum(-1) + self.indv_b2


def has_g(row):
    return "genetic" in row and len(row["genetic"]["names"]) >= 1


def to_np(y):
    return y.detach().float().cpu().numpy() if torch.is_tensor(y) else np.asarray(y, dtype=float)


def ecfp_matrix(names, recs):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.warning")
    from rdkit.Chem import AllChem, DataStructs

    mat = np.zeros((len(names), 256), dtype=np.float32)
    ok = np.zeros(len(names), dtype=bool)
    for i, name in enumerate(names):
        smi = recs.get(str(name)) or recs.get(str(name).lower()) or recs.get(compact(name))
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256)
        arr = np.zeros((256,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        mat[i] = arr
        ok[i] = True
    return mat, ok


def _emb_ok(mat):
    return np.isfinite(mat).all(1) & (np.abs(mat).sum(1) > 0)


def train_g2cp(data, target, sources, names, device, seed, max_epochs=40, max_n=None):
    rng = np.random.default_rng(seed)
    xs, ys, ctxs = [], [], []
    cell_ctx = {}
    for cell in list(sources) + [target]:
        gy = to_np(data[cell]["genetic"]["y"]) if has_g(data[cell]) else np.zeros(len(data[target]["genes"]), np.float32)
        cell_ctx[cell] = gy.mean(0).astype(np.float32) if len(gy) else np.zeros(len(data[target]["genes"]), np.float32)
    for cell in sources:
        emb, y = to_np(data[cell]["chemical"]["x"]), to_np(data[cell]["chemical"]["y"])
        if len(emb) == 0:
            continue
        ok = _emb_ok(emb)
        if not ok.any():
            continue
        xs.append(emb[ok].astype(np.float32))
        ys.append(y[ok].astype(np.float32))
        ctxs.append(np.repeat(cell_ctx[cell][None], int(ok.sum()), 0))
    for cell in list(sources) + [target]:
        if not has_g(data[cell]):
            continue
        emb, y = to_np(data[cell]["genetic"]["x"]), to_np(data[cell]["genetic"]["y"])
        ok = _emb_ok(emb)
        if not ok.any():
            continue
        xs.append(emb[ok].astype(np.float32))
        ys.append(y[ok].astype(np.float32))
        ctxs.append(np.repeat(cell_ctx[cell][None], int(ok.sum()), 0))
    if not xs:
        raise RuntimeError("g2cp has no training rows")
    x, y, ctx = np.concatenate(xs), np.concatenate(ys), np.concatenate(ctxs)
    if max_n and len(y) > max_n:
        take = rng.choice(len(y), size=max_n, replace=False)
        x, y, ctx = x[take], y[take], ctx[take]
    n = len(y)
    perm = rng.permutation(n)
    n_val = max(32, min(256, n // 10))
    if n > n_val:
        val_i, tr_i = perm[:n_val], perm[n_val:]
    else:
        val_i, tr_i = perm, perm
    m = G2CPD0Net(y.shape[1]).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    best, best_state, wait = 1e9, None, 0
    t0 = time.time()
    for epoch in range(max_epochs):
        m.train()
        rng.shuffle(tr_i)
        for s in range(0, len(tr_i), 256):
            sl = tr_i[s:s + 256]
            opt.zero_grad(set_to_none=True)
            pred = m(torch.tensor(x[sl], device=device), torch.tensor(ctx[sl], device=device))
            loss = (pred - torch.tensor(y[sl], device=device)).square().mean()
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        m.eval()
        with torch.no_grad():
            pred = m(torch.tensor(x[val_i], device=device), torch.tensor(ctx[val_i], device=device))
            vloss = float((pred - torch.tensor(y[val_i], device=device)).square().mean())
        if vloss < best:
            best, wait, best_state = vloss, 0, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= 8:
                break
    if best_state:
        m.load_state_dict(best_state)
    return m, cell_ctx, time.time() - t0


def infer_g2cp(m, data, target, names, cell_ctx, device):
    look = {str(n): i for i, n in enumerate(data[target]["chemical"]["names"])}
    qemb = np.stack([to_np(data[target]["chemical"]["x"])[look[str(n)]] for n in names])
    qok = _emb_ok(qemb)
    truth = np.stack([to_np(data[target]["chemical"]["y"])[look[str(n)]] for n in names])
    tctx = torch.tensor(np.repeat(cell_ctx[target][None], len(names), 0), device=device)
    pred = np.zeros_like(truth)
    m.eval()
    with torch.no_grad():
        for s in range(0, len(names), 128):
            sl = slice(s, s + 128)
            pred[sl] = m(torch.tensor(qemb[sl], device=device), tctx[sl]).float().cpu().numpy()
    return pred, truth, float(qok.mean()), qok


def train_chem(kind, data, target, sources, names, recs, device, seed, max_epochs=40, max_n=None):
    rng = np.random.default_rng(seed)
    xs, ys, cs = [], [], []
    src = [c for c in sources if len(data[c]["chemical"]["names"])]
    for si, cell in enumerate(src):
        fp, ok = ecfp_matrix(data[cell]["chemical"]["names"], recs)
        if not ok.any():
            continue
        xs.append(fp[ok])
        ys.append(to_np(data[cell]["chemical"]["y"])[ok])
        cs.append(np.full(int(ok.sum()), si, dtype=np.int64))
    if not xs:
        return None, None, None, 0.0, None
    x, y, c = np.concatenate(xs), np.concatenate(ys), np.concatenate(cs)
    if max_n and len(y) > max_n:
        take = rng.choice(len(y), size=max_n, replace=False)
        x, y, c = x[take], y[take], c[take]
    basal_src = y.mean(0, keepdims=True).repeat(len(y), 0)
    n = len(y)
    perm = rng.permutation(n)
    n_val = max(32, min(256, n // 10))
    if n > n_val:
        val_i, tr_i = perm[:n_val], perm[n_val:]
    else:
        val_i, tr_i = perm, perm
    genes = y.shape[1]
    if kind == "deepce":
        m = DeepCENet(genes, len(src)).to(device)
    elif kind == "prnet":
        m = PRnetNet(genes).to(device)
    elif kind == "mlp_morgan":
        m = MLPMorgan(genes, len(src)).to(device)
    else:
        m = XPertDual(genes).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    best, best_state, wait = 1e9, None, 0
    t0 = time.time()

    def one_hot(idx):
        o = torch.zeros((len(idx), len(src)), device=device)
        o[torch.arange(len(idx), device=device), idx] = 1.0
        return o

    def predict(fp, cc, bb):
        if kind == "deepce":
            return m(fp, cc)
        if kind == "prnet":
            return m(fp, bb)
        if kind == "mlp_morgan":
            return m(fp, cc)
        return m(fp, bb)

    for epoch in range(max_epochs):
        m.train()
        rng.shuffle(tr_i)
        for s in range(0, len(tr_i), 256):
            sl = tr_i[s:s + 256]
            fp = torch.tensor(x[sl], device=device)
            yy = torch.tensor(y[sl], device=device)
            cc = one_hot(torch.tensor(c[sl], device=device))
            bb = torch.tensor(basal_src[sl], device=device)
            opt.zero_grad(set_to_none=True)
            loss = (predict(fp, cc, bb) - yy).square().mean()
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        m.eval()
        with torch.no_grad():
            vloss = float((predict(torch.tensor(x[val_i], device=device), one_hot(torch.tensor(c[val_i], device=device)), torch.tensor(basal_src[val_i], device=device)) - torch.tensor(y[val_i], device=device)).square().mean())
        if vloss < best:
            best, wait, best_state = vloss, 0, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= 8:
                break
    if best_state:
        m.load_state_dict(best_state)
    return m, src, y.mean(0), time.time() - t0, predict


def infer_chem(m, predict, data, target, names, recs, src, ymean, device):
    qfp, qok = ecfp_matrix(names, recs)
    look = {str(n): i for i, n in enumerate(data[target]["chemical"]["names"])}
    truth = np.stack([to_np(data[target]["chemical"]["y"])[look[str(n)]] for n in names])
    basal = torch.tensor(np.repeat(ymean[None], len(names), 0), device=device)
    cell = torch.full((len(names), len(src)), 1.0 / len(src), device=device)
    pred = np.zeros_like(truth)
    m.eval()
    with torch.no_grad():
        for s in range(0, len(names), 128):
            sl = slice(s, s + 128)
            pred[sl] = predict(torch.tensor(qfp[sl], device=device), cell[sl], basal[sl]).float().cpu().numpy()
    return pred, truth, float(qok.mean()), qok

