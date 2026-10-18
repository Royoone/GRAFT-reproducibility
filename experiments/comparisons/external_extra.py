"""External CIGER, TranSiGen and chemCPA comparison training."""
from __future__ import annotations
import time
import numpy as np
import torch
from torch import nn
from . import external as C
class CIGERNet(C.DeepCENet):
    pass


class TranSiGenNet(nn.Module):
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


class ChemCPANet(nn.Module):
    def __init__(self, genes, n_cell=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(genes, 128), nn.ReLU())
        self.drug = nn.Linear(256, 128)
        self.cell = nn.Linear(n_cell, 128)
        self.dec = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, genes))

    def forward(self, fp, ctrl, cell):
        return self.dec(self.enc(ctrl) + self.drug(fp) + self.cell(cell))


def rank_loss(pred, y):
    i = torch.randint(0, y.size(1), (y.size(0), 64), device=y.device)
    j = torch.randint(0, y.size(1), (y.size(0), 64), device=y.device)
    yi, yj = y.gather(1, i), y.gather(1, j)
    pi, pj = pred.gather(1, i), pred.gather(1, j)
    return torch.relu(1.0 - torch.sign(yi - yj) * (pi - pj)).mean()


def train_extra(kind, data, sources, recs, device, seed, max_epochs=40):
    rng = np.random.default_rng(seed)
    xs, ys, cs = [], [], []
    src = [c for c in sources if len(data[c]["chemical"]["names"])]
    for si, cell in enumerate(src):
        fp, ok = C.ecfp_matrix(data[cell]["chemical"]["names"], recs)
        if not ok.any():
            continue
        xs.append(fp[ok])
        ys.append(C.to_np(data[cell]["chemical"]["y"])[ok])
        cs.append(np.full(int(ok.sum()), si, dtype=np.int64))
    if not xs:
        return None
    x, y, c = np.concatenate(xs), np.concatenate(ys), np.concatenate(cs)
    basal_src = y.mean(0, keepdims=True).repeat(len(y), 0)
    perm = rng.permutation(len(y))
    n_val = max(32, min(256, len(y) // 10))
    val_i, tr_i = (perm[:n_val], perm[n_val:]) if len(y) > n_val else (perm, perm)
    genes = y.shape[1]
    if kind == "ciger":
        m = CIGERNet(genes, len(src)).to(device)
    elif kind == "transigen":
        m = TranSiGenNet(genes).to(device)
    else:
        m = ChemCPANet(genes, len(src)).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)

    def one_hot(idx):
        o = torch.zeros((len(idx), len(src)), device=device)
        o[torch.arange(len(idx), device=device), idx] = 1.0
        return o

    def predict(fp, cc, bb, train=False):
        if kind == "ciger":
            return m(fp, cc)
        if kind == "transigen":
            out = m(fp, bb)
            return out if train else out[0]
        return m(fp, bb, cc)

    best, best_state, wait = 1e9, None, 0
    t0 = time.time()
    for _epoch in range(max_epochs):
        m.train()
        rng.shuffle(tr_i)
        for s in range(0, len(tr_i), 256):
            sl = tr_i[s:s + 256]
            fp = torch.tensor(x[sl], device=device)
            yy = torch.tensor(y[sl], device=device)
            cc = one_hot(torch.tensor(c[sl], device=device))
            bb = torch.tensor(basal_src[sl], device=device)
            opt.zero_grad(set_to_none=True)
            if kind == "transigen":
                pred, mu, lv = predict(fp, cc, bb, train=True)
                kld = -0.5 * (1 + lv - mu.square() - lv.exp()).mean()
                loss = (pred - yy).square().mean() + 0.1 * kld
            else:
                pred = predict(fp, cc, bb)
                loss = (pred - yy).square().mean()
                if kind == "ciger":
                    loss = loss + 0.5 * rank_loss(pred, yy)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        m.eval()
        with torch.no_grad():
            pred = predict(
                torch.tensor(x[val_i], device=device),
                one_hot(torch.tensor(c[val_i], device=device)),
                torch.tensor(basal_src[val_i], device=device),
            )
            vloss = float((pred - torch.tensor(y[val_i], device=device)).square().mean())
        if vloss < best:
            best, wait, best_state = vloss, 0, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= 8:
                break
    if best_state:
        m.load_state_dict(best_state)
    return m, src, y.mean(0), time.time() - t0, predict, one_hot


def infer_extra(m, predict, data, target, names, recs, src, ymean, device):
    qfp, qok = C.ecfp_matrix(names, recs)
    look = {str(n): i for i, n in enumerate(data[target]["chemical"]["names"])}
    truth = np.stack([C.to_np(data[target]["chemical"]["y"])[look[str(n)]] for n in names])
    basal = torch.tensor(np.repeat(ymean[None], len(names), 0), device=device)
    cell = torch.full((len(names), len(src)), 1.0 / len(src), device=device)
    pred = np.zeros_like(truth)
    m.eval()
    with torch.no_grad():
        for s in range(0, len(names), 128):
            sl = slice(s, s + 128)
            pred[sl] = predict(torch.tensor(qfp[sl], device=device), cell[sl], basal[sl]).float().cpu().numpy()
    return pred, truth, float(qok.mean()), qok

