"""Algebraically equivalent vectorized author forward used in the final fit."""
from types import MethodType,SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F

def fast_forward(self, data, cross_perturb=True):
    x = data.x
    if self.no_perturb:
        return x.reshape(-1, self.num_genes)
    p = data.unipert_tensor
    if self._fast_detach_unipert: p = p.detach()
    b = len(p)
    g = self.num_genes
    device = p.device
    idx = torch.arange(g, device=device)
    full_idx = idx.repeat(b)
    emb = self.gene_emb(full_idx)
    base = self.emb_trans(self.bn_emb(emb))
    # All samples have identical positional embeddings and gene graph. Computing
    # this branch once and repeating its outputs gives identical summed gradients.
    pos = self.emb_pos(idx)
    for i, layer in enumerate(self.layers_emb_pos):
        pos = layer(pos, self.G_coexpress, self.G_coexpress_weight)
        if i < len(self.layers_emb_pos)-1: pos = pos.relu()
    base = self.emb_trans_v2(base + .2 * pos.repeat(b, 1)).reshape(b, g, -1)
    pert = self.unipert_emb_trans(p)
    if cross_perturb: pert = F.normalize(pert, dim=1)
    if b == 1:
        total = self.pert_fuse(pert.repeat(2, 1))[:1]
    else:
        total = self.pert_fuse(pert)
    base = (base + total[:, None, :]).reshape(b*g, -1)
    base = self.bn_pert_base(base)
    base = self.transform(base)
    out = self.recovery_w(base).reshape(b, g, -1)
    out = (out.unsqueeze(-1) * self.indv_w1).sum(2) + self.indv_b1
    cross = self.cross_gene_state(out.reshape(b, g, -1).squeeze(2))
    cross = cross.repeat(1, g).reshape(b, g, -1)
    out = (torch.cat([out, cross], dim=2) * self.indv_w2).sum(2) + self.indv_b2
    out = out + x.reshape(b, g)
    if self.uncertainty:
        return out, self.uncertainty_w(base).reshape(b, g)
    return out

def enable_fast_forward(model, detach_unipert=True):
    model._fast_detach_unipert = detach_unipert
    model.forward = MethodType(fast_forward, model)
    return model

def fast_batch_for(x, genes, device):
    # One contiguous host->device copy replaces one copy per condition.
    t = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=device)
    return SimpleNamespace(x=torch.zeros((len(t)*genes, 1), device=device), unipert_tensor=t)
