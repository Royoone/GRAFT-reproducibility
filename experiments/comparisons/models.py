"""Task-adapted neural comparators; response-space inputs as in Methods."""
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
CELLS=('A375','A549','HT29','MCF7','PC3')
EPOCHS=40
BATCH=256
LR=1e-3
class DeepCENet(nn.Module):
    """DeepCE flow: drug fingerprint × gene table + cell id → profile (MSE)."""

    def __init__(self, genes, n_cell=4):
        super().__init__()
        self.drug = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1))
        self.cell = nn.Linear(n_cell, 32)
        self.gene = nn.Parameter(torch.randn(genes, 128) * 0.02)
        self.mix = nn.Sequential(nn.Linear(128 + 32 + 128, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, fp, cell):
        d = self.drug(fp)
        c = self.cell(cell)
        attn = torch.softmax(d @ self.gene.T / 8.0, dim=-1)
        ctx = attn @ self.gene
        h = torch.cat((d, c, ctx), -1)
        # gene-wise readout: broadcast drug/cell onto each gene
        g = self.gene.unsqueeze(0).expand(len(fp), -1, -1)
        d_b = d.unsqueeze(1).expand_as(g)
        c_b = c.unsqueeze(1).expand(-1, g.size(1), -1)
        return self.mix(torch.cat((d_b, c_b, g), -1)).squeeze(-1)


class CIGERNet(DeepCENet):
    """CIGER: same backbone, ranking-oriented head (shared weights, rank loss outside)."""


class TranSiGenNet(nn.Module):
    """TranSiGen-style VAE: basal + molecule → latent → profile."""

    def __init__(self, genes):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(256 + genes, 256), nn.ReLU(), nn.Linear(256, 128))
        self.mu = nn.Linear(128, 64)
        self.lv = nn.Linear(128, 64)
        self.dec = nn.Sequential(nn.Linear(64 + 256, 256), nn.ReLU(), nn.Linear(256, genes))

    def forward(self, fp, basal):
        h = self.enc(torch.cat((fp, basal), -1))
        mu, lv = self.mu(h), self.lv(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv.clamp(-8, 8)) if self.training else mu
        return self.dec(torch.cat((z, fp), -1)), mu, lv


class PRnetNet(nn.Module):
    """PRnet-style: unperturbed profile + molecule → perturbed profile."""

    def __init__(self, genes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256 + genes, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, 256), nn.GELU(), nn.Linear(256, genes),
        )

    def forward(self, fp, unpert):
        return self.net(torch.cat((fp, unpert), -1))


class ChemCPANet(nn.Module):
    """chemCPA-style: encode control, add drug and cell, decode."""

    def __init__(self, genes, n_cell=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(genes, 128), nn.ReLU())
        self.drug = nn.Linear(256, 128)
        self.cell = nn.Linear(n_cell, 128)
        self.dec = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, genes))

    def forward(self, fp, ctrl, cell):
        z = self.enc(ctrl) + self.drug(fp) + self.cell(cell)
        return self.dec(z)


class MLPMorgan(nn.Module):
    """XPert MLP_Morgan: Morgan/ECFP MLP, no gene attention."""

    def __init__(self, genes, n_cell=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256 + n_cell, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, genes),
        )

    def forward(self, fp, cell):
        return self.net(torch.cat((fp, cell), -1))


class G2CPD0Net(nn.Module):
    """G2CP-D0 adaptation: UniPert 256 → hidden, add to gene basis, gene-wise decode.

    Follows G2CP/GEARS information flow (unipert_emb_trans + gene table +
    recovery + cross-gene head) without the coexpression GNN. Effect-space
    output (no control add). Not official weights / not same-cell Dk.
    """

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
        p = torch.nn.functional.normalize(self.unipert_emb_trans(pert), dim=-1)
        p = self.pert_fuse(p)
        c = self.ctx(ctx)
        h = self.recovery(base + p.unsqueeze(1) + 0.2 * c.unsqueeze(1))
        out = (h * self.indv_w1.unsqueeze(0)).sum(-1) + self.indv_b1
        cross = self.cross(out).unsqueeze(1).expand(-1, n_g, -1)
        cat = torch.cat((out.unsqueeze(-1), cross), -1)
        return (cat * self.indv_w2.unsqueeze(0)).sum(-1) + self.indv_b2


class XPertDual(nn.Module):
    """XPert dual-branch idea without official binning/HG: control branch vs drug branch."""

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
        return self.out((h.squeeze(1) - k.squeeze(1)))


def rank_loss(pred, y):
    # CIGER-style: pairwise sign agreement on a random gene subset
    i = torch.randint(0, y.size(1), (y.size(0), 64), device=y.device)
    j = torch.randint(0, y.size(1), (y.size(0), 64), device=y.device)
    yi = y.gather(1, i)
    yj = y.gather(1, j)
    pi = pred.gather(1, i)
    pj = pred.gather(1, j)
    return torch.relu(1.0 - torch.sign(yi - yj) * (pi - pj)).mean()


def make_model(kind, genes, n_cell, device):
    if kind == "deepce":
        m = DeepCENet(genes, n_cell)
    elif kind == "ciger":
        m = CIGERNet(genes, n_cell)
    elif kind == "transigen":
        m = TranSiGenNet(genes)
    elif kind == "prnet":
        m = PRnetNet(genes)
    elif kind == "chemcpa":
        m = ChemCPANet(genes, n_cell)
    elif kind == "xpert_dual":
        m = XPertDual(genes)
    elif kind == "mlp_morgan":
        m = MLPMorgan(genes, n_cell)
    elif kind == "g2cp_d0":
        m = G2CPD0Net(genes)
    else:
        raise KeyError(kind)
    return m.to(device)


def predict_batch(m, kind, fp, cell, basal):
    if kind in ("deepce", "ciger"):
        return m(fp, cell)
    if kind == "transigen":
        return m(fp, basal)[0]
    if kind == "prnet":
        return m(fp, basal)
    if kind == "chemcpa":
        return m(fp, basal, cell)
    if kind == "xpert_dual":
        return m(fp, basal)
    if kind == "mlp_morgan":
        return m(fp, cell)
    raise KeyError(kind)


def one_hot(idx, n, device):
    x = torch.zeros((len(idx), n), device=device)
    x[torch.arange(len(idx), device=device), idx] = 1.0
    return x


def _emb_ok(mat):
    return np.isfinite(mat).all(1) & (np.abs(mat).sum(1) > 0)


def train_g2cp(data, target, names, device, seed, extra_names=None, tag="g2cp_d0"):
    """Source D + all-cell G. Optional non-query target chemicals (Dk). Queries sealed."""
    sources = [c for c in CELLS if c != target]
    qset = set(map(str, names))
    extra_names = [str(n) for n in (extra_names or []) if str(n) not in qset]
    rng = np.random.default_rng(seed)
    xs, ys, ctxs = [], [], []
    cell_ctx = {}
    for cell in CELLS:
        gy = data[cell]["gen_y"]
        cell_ctx[cell] = gy.mean(0).astype(np.float32)
    for cell in sources:
        emb, y = data[cell]["chem_emb"], data[cell]["y"]
        ok = _emb_ok(emb)
        xs.append(emb[ok])
        ys.append(y[ok])
        ctxs.append(np.repeat(cell_ctx[cell][None], int(ok.sum()), 0))
    if extra_names:
        look = {str(n): i for i, n in enumerate(data[target]["names"])}
        emb, y = data[target]["chem_emb"], data[target]["y"]
        picked = []
        for name in extra_names:
            if name not in look:
                continue
            i = look[name]
            if not bool(_emb_ok(emb[i:i + 1])[0]):
                continue
            xs.append(emb[i:i + 1])
            ys.append(y[i:i + 1])
            ctxs.append(cell_ctx[target][None])
            picked.append(name)
        extra_names = picked
    for cell in CELLS:
        emb, y = data[cell]["gen_emb"], data[cell]["gen_y"]
        ok = _emb_ok(emb)
        xs.append(emb[ok])
        ys.append(y[ok])
        ctxs.append(np.repeat(cell_ctx[cell][None], int(ok.sum()), 0))
    print(f"  {tag} {target} {seed} extra_target_D={len(extra_names)}", flush=True)
    x = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.float32)
    ctx = np.concatenate(ctxs).astype(np.float32)
    n = len(y)
    perm = rng.permutation(n)
    n_val = max(256, n // 10)
    val_i, tr_i = perm[:n_val], perm[n_val:]
    m = G2CPD0Net(y.shape[1]).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    best, best_state, wait = 1e9, None, 0
    t0 = time.time()
    for epoch in range(EPOCHS):
        m.train()
        rng.shuffle(tr_i)
        tot, steps = 0.0, 0
        for s in range(0, len(tr_i), BATCH):
            sl = tr_i[s:s + BATCH]
            opt.zero_grad(set_to_none=True)
            pred = m(torch.tensor(x[sl], device=device), torch.tensor(ctx[sl], device=device))
            loss = (pred - torch.tensor(y[sl], device=device)).square().mean()
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            steps += 1
        m.eval()
        with torch.no_grad():
            pred = m(torch.tensor(x[val_i], device=device), torch.tensor(ctx[val_i], device=device))
            vloss = float((pred - torch.tensor(y[val_i], device=device)).square().mean())
        if vloss < best:
            best, wait = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= 8:
                break
        print(f"  {tag} {target} {seed} epoch {epoch+1} train={tot/max(steps,1):.4f} val={vloss:.4f}", flush=True)
    if best_state is not None:
        m.load_state_dict(best_state)
    look = {str(n): i for i, n in enumerate(data[target]["names"])}
    qn = names
    qemb = np.stack([data[target]["chem_emb"][look[str(n)]] for n in qn])
    qok = _emb_ok(qemb)
    truth = np.stack([data[target]["y"][look[str(n)]] for n in qn])
    tctx = torch.tensor(np.repeat(cell_ctx[target][None], len(qn), 0), device=device)
    m.eval()
    pred = np.zeros_like(truth)
    with torch.no_grad():
        for s in range(0, len(qn), 128):
            sl = slice(s, s + 128)
            pred[sl] = m(torch.tensor(qemb[sl], device=device), tctx[sl]).float().cpu().numpy()
    src_mean = []
    for name in qn:
        acc = []
        for cell_n in sources:
            li = {str(n): i for i, n in enumerate(data[cell_n]["names"])}
            if str(name) in li:
                acc.append(data[cell_n]["y"][li[str(name)]])
        src_mean.append(np.mean(np.stack(acc), 0) if acc else np.zeros(truth.shape[1], dtype=np.float32))
    pred = np.where(qok[:, None], pred, np.stack(src_mean))
    return pred, truth, float(qok.mean()), time.time() - t0, int(sum(p.numel() for p in m.parameters())), extra_names

