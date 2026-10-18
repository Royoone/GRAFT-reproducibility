"""Fixed ridge, gated ridge and low-rank remix formulas used in the paper."""
from __future__ import annotations
import torch
from torch import nn

ALL_KINDS = ("ridge", "kernel", "ridge_remix")

def l2_rows(x, eps=1e-12):
    n = torch.sqrt((x * x).sum(-1, keepdim=True)).clamp_min(eps)
    return x / n

def design_torch(features, eps=1e-12):
    feat = l2_rows(features, eps)
    ones = torch.ones(feat.shape[0], 1, device=feat.device, dtype=feat.dtype)
    return torch.cat([feat, ones], dim=1)

def ridge_delta(p0, y, feat, anchor_idx, query_idx, lam=1.0):
    """Differentiable dual ridge residual for query rows."""
    if anchor_idx.numel() == 0:
        return torch.zeros(query_idx.numel(), p0.shape[1], device=p0.device, dtype=p0.dtype)
    phi = design_torch(feat)
    phi_a = phi[anchor_idx]
    resid = y[anchor_idx] - p0[anchor_idx]
    k = phi_a.shape[0]
    eye = torch.eye(k, device=p0.device, dtype=p0.dtype)
    gram = phi_a @ phi_a.T + lam * eye
    coeff = torch.linalg.solve(gram, resid)
    return (phi[query_idx] @ phi_a.T) @ coeff

def kernel_delta(p0, y, feat, anchor_idx, query_idx, tau=1.0):
    """Soft retrieval of anchor residuals by row-L2 cosine."""
    if anchor_idx.numel() == 0:
        return torch.zeros(query_idx.numel(), p0.shape[1], device=p0.device, dtype=p0.dtype)
    q = l2_rows(feat[query_idx])
    a = l2_rows(feat[anchor_idx])
    resid = y[anchor_idx] - p0[anchor_idx]
    logits = (q @ a.T) / tau
    w = torch.softmax(logits, dim=-1)
    return w @ resid

def topk_indices(values, k):
    return torch.topk(values.abs(), k, dim=-1).indices

def gather_k(mat, idx):
    return torch.gather(mat, 1, idx)

def pearson_rows(a, b, eps=1e-12):
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    na = torch.sqrt((a * a).sum(-1)).clamp_min(eps)
    nb = torch.sqrt((b * b).sum(-1)).clamp_min(eps)
    r = (a * b).sum(-1) / (na * nb)
    const = (a.max(-1).values - a.min(-1).values) < 1e-8
    r = torch.where(const, torch.zeros_like(r), r)
    return r

def topk_r(truth, pred, k=20, exclude=None):
    idx = topk_indices(truth, k)
    r = pearson_rows(gather_k(truth, idx), gather_k(pred, idx))
    if exclude is None:
        return r.mean()
    return r[exclude].mean()

def topk_mse(truth, pred, k=20):
    idx = topk_indices(truth, k)
    err = gather_k(pred, idx) - gather_k(truth, idx)
    return (err * err).mean()

class GatedAdapter(nn.Module):
    def __init__(self, n_genes=976, kind="ridge", rank=8, n_feat=64, init_alpha=-3.0):
        super().__init__()
        if kind not in ALL_KINDS:
            raise ValueError(kind)
        self.kind = kind
        self.n_feat = n_feat
        self.raw_alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        self.raw_gate = nn.Parameter(torch.full((n_genes,), -1.5))
        self.log_lam = nn.Parameter(torch.tensor(0.0))
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        self.raw_mix = nn.Parameter(torch.tensor(-4.0))
        self.U = nn.Parameter(0.01 * torch.randn(n_genes, rank))
        self.V = nn.Parameter(0.01 * torch.randn(rank, n_genes))

    def scale(self):
        return torch.sigmoid(self.raw_alpha)

    def gate(self):
        return torch.sigmoid(self.raw_gate)

    def lam(self):
        return torch.nn.functional.softplus(self.log_lam) + 0.05

    def tau(self):
        return torch.nn.functional.softplus(self.log_tau) + 0.05

    def remix(self, delta):
        mix = torch.sigmoid(self.raw_mix)
        return delta + mix * ((delta @ self.V.T) @ self.U.T)

    def forward(self, p0, y, feat, anchor_idx, query_idx, feat_alt=None):
        base = p0[query_idx]
        if anchor_idx.numel() == 0:
            return base
        if self.kind == "kernel":
            delta = kernel_delta(p0, y, feat, anchor_idx, query_idx, tau=self.tau())
        else:
            delta = ridge_delta(p0, y, feat, anchor_idx, query_idx, lam=self.lam())
            if self.kind == "ridge_remix":
                delta = self.remix(delta)
        return base + self.scale() * delta * self.gate()


def d0_preserve_loss(truth, pred, p0, keep20, mu=0.25, nu=2.0):
    """Outer objective: raise query Top-20 r, never reward falling below D0."""
    r1 = topk_r(truth, pred, 20, keep20)
    r0 = topk_r(truth, p0, 20, keep20)
    r50 = topk_r(truth, pred, 50)
    mse20 = topk_mse(truth, pred, 20)
    stay = ((pred - p0) ** 2).mean()
    drop = torch.relu(r0 - r1)
    loss = (1.0 - r1) + 0.25 * (1.0 - r50) + 0.05 * mse20 + mu * stay + nu * drop
    return loss, {
        "r20": float(r1.detach()),
        "r0": float(r0.detach()),
        "drop": float(drop.detach()),
        "stay": float(stay.detach()),
        "scale": None,
    }
