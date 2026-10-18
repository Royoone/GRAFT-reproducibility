"""GRAFT response prediction and optional drug-anchor adaptation.

The network consumes prepared perturbation-level responses, not raw counts.
The ordered gene panel, frozen 256-dimensional UniPert representations, source
references and checkpoint-specific response scale are supplied by the caller.
All registered parameter names match the released model and adapter weights.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = ["GRAFT", "GRAFTAdapter", "GRAFTSaturating"]

Pack = Mapping[str, Tensor]
State = Mapping[str, Tensor]


class GRAFT(nn.Module):
    """Infer chemical transcriptional responses from a genetic support set.

    A support pack contains ``x`` (N x 256 frozen genetic representations),
    ``mean`` (N x G source reference responses), and ``delta`` (N x G target
    responses minus source references). A query pack contains chemical ``x``
    (Q x 256) and chemical source ``mean`` (Q x G). Response-valued inputs and
    predictions use the checkpoint's common scaled response units. Scale
    conversion and reference construction belong outside this module.

    The default architecture uses a bounded cross-moment summary and a residual
    context encoder. ``forward`` accepts the state returned by ``encode``;
    ``inference`` combines both operations. These methods handle one context at
    a time, with any number of query drugs sharing its state.
    """

    def __init__(
        self,
        genes: int = 976,
        seed: int = 52,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.genes = genes
        self.identity = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, 32), nn.Tanh()
        )
        self.ref = nn.Sequential(nn.Linear(genes, 16), nn.Tanh())
        self.diff = nn.Linear(genes, 16, bias=False)
        self.moment = nn.Linear(48 * 16, 32, bias=False)
        self.query = nn.Sequential(
            nn.Linear(256 + genes, 64), nn.GELU(), nn.LayerNorm(64)
        )
        self.hyper = nn.Sequential(
            nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 128)
        )
        self.post = nn.Sequential(nn.LayerNorm(64), nn.GELU())
        self.output = nn.Linear(64, genes, bias=False)
        nn.init.normal_(self.output.weight, std=0.01)
        self.skip = nn.Linear(32, 32, bias=False)
        self.mlp = nn.Sequential(
            nn.LayerNorm(32), nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32)
        )
        nn.init.eye_(self.skip.weight)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.to(device)

    def covariance(self, p: Pack) -> Tensor:
        """Return the 48 x 16 centered, column-normalized cross-moment."""
        raw = p["delta"].mean(0)
        f = torch.cat((self.identity(p["x"]), self.ref(p["mean"])), -1)
        f = f - f.mean(0)
        v = torch.tanh(self.diff(p["delta"] - raw))
        v = v - v.mean(0)
        f = f / f.square().mean(0).clamp_min(0.05**2).sqrt()
        v = v / v.square().mean(0).clamp_min(0.05**2).sqrt()
        return (f.T @ v) / len(f)

    def bound(self, c: Tensor) -> Tensor:
        """Bound summary RMS by one, using the checkpoint's numerical floor."""
        rms = c.square().mean().add(1e-8).sqrt()
        return c / rms.clamp_min(1.0)

    def summarize(self, p: Pack) -> Tensor:
        """Project the genetic cross-moment to a bounded 32-dimensional summary."""
        return self.bound(self.moment(self.covariance(p).flatten()))

    def E(self, c: Tensor) -> Tensor:
        """Apply the residual context encoder before zero-state subtraction."""
        return self.skip(c) + self.mlp(c)

    def encode(
        self, support: Mapping[str, Pack], mode: str = "true"
    ) -> dict[str, Tensor]:
        """Encode ``{'genetic': pack}`` into ``raw`` and zero-centered ``z``.

        Missing or empty genetic support, or ``mode='off'``/``'none'``, produces
        a zero state. Its prediction is exactly the source reference. A single
        support condition also has a zero centered cross-moment.
        """
        p = None if mode in ("off", "none") else support.get("genetic")
        if p is None or p["x"].shape[0] == 0:
            raw = self.output.weight.new_zeros(32)
        else:
            raw = self.summarize(p)
        return {"raw": raw, "z": self.E(raw) - self.E(torch.zeros_like(raw))}

    def decode_with_features(
        self, q: Pack, state: State, gain: float = 1.0
    ) -> tuple[Tensor, Tensor]:
        """Return scaled query predictions and the conditional 64-D features.

        The second result is the post-normalization, post-GELU feature used by
        the anchor adapter, before row L2 normalization and constant appending.
        ``gain=1`` is the trained prediction setting.
        """
        h = self.query(torch.cat((q["x"], q["mean"]), -1))
        film = self.hyper(state["z"]) - self.hyper(torch.zeros_like(state["z"]))
        gamma, beta = film.chunk(2, dim=-1)
        conditional = h * (1 + 0.5 * gamma.tanh()) + beta
        post64 = self.post(conditional)
        pred = q["mean"] + gain * self.output(post64 - self.post(h))
        return pred, post64

    def decode(self, q: Pack, state: State, gain: float = 1.0) -> Tensor:
        """Predict scaled query responses from a previously encoded context."""
        return self.decode_with_features(q, state, gain)[0]

    def forward(self, q: Pack, state: State, gain: float = 1.0) -> Tensor:
        return self.decode(q, state, gain)

    @torch.no_grad()
    def inference(
        self,
        q: Pack,
        support: Mapping[str, Pack],
        gain: float = 1.0,
        chunk: int = 128,
        *,
        return_features: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Encode one context and predict query rows in evaluation-mode chunks.

        If ``return_features`` is true, also return the Q x 64 conditional
        features. Predictions remain in scaled response units in either case.
        """
        if chunk < 1:
            raise ValueError("chunk must be a positive integer")
        self.eval()
        state = self.encode(support)
        predictions, features = [], []
        for start in range(0, len(q["x"]), chunk):
            query_chunk = {key: value[start : start + chunk] for key, value in q.items()}
            pred, post64 = self.decode_with_features(query_chunk, state, gain)
            predictions.append(pred)
            if return_features:
                features.append(post64)
        pred_all = (
            torch.cat(predictions)
            if predictions
            else q["mean"].new_empty((0, self.genes))
        )
        if not return_features:
            return pred_all
        feat_all = (
            torch.cat(features) if features else q["mean"].new_empty((0, 64))
        )
        return pred_all, feat_all


def l2_rows(x: Tensor, eps: float = 1e-12) -> Tensor:
    """Normalize feature rows with the training implementation's norm floor."""
    norm = torch.sqrt((x * x).sum(-1, keepdim=True)).clamp_min(eps)
    return x / norm


def design_torch(features: Tensor, eps: float = 1e-12) -> Tensor:
    """Append a constant to L2-normalized conditional features (64 -> 65)."""
    normalized = l2_rows(features, eps)
    ones = torch.ones(
        normalized.shape[0], 1, device=normalized.device, dtype=normalized.dtype
    )
    return torch.cat([normalized, ones], dim=1)


def ridge_delta(
    p0: Tensor,
    y: Tensor,
    feat: Tensor,
    anchor_idx: Tensor,
    query_idx: Tensor,
    lam: Tensor | float = 1.0,
) -> Tensor:
    """Predict query residuals by dual ridge regression on measured anchors."""
    if anchor_idx.numel() == 0:
        return torch.zeros(
            query_idx.numel(), p0.shape[1], device=p0.device, dtype=p0.dtype
        )
    phi = design_torch(feat)
    phi_a = phi[anchor_idx]
    residual = y[anchor_idx] - p0[anchor_idx]
    eye = torch.eye(phi_a.shape[0], device=p0.device, dtype=p0.dtype)
    coefficient = torch.linalg.solve(phi_a @ phi_a.T + lam * eye, residual)
    return (phi[query_idx] @ phi_a.T) @ coefficient


class GRAFTAdapter(nn.Module):
    """Apply the frozen, learned drug-anchor residual correction.

    ``p0`` and ``y`` are N x G baseline predictions and observed responses in
    original response units, and ``feat`` is the N x 64 feature matrix returned
    by ``GRAFT.decode_with_features``. Only ``y[anchor_idx]`` is read; unobserved
    rows can contain NaN. Anchors and evaluation queries must be disjoint.

    The default ``ridge_remix`` uses normalized 65-D design vectors, learned
    positive ridge regularization, a low-rank residual remix, a global scale
    and gene-wise gates. ``ridge`` omits only the low-rank remix. All learned
    parameters stay fixed during use; fitting anchor coefficients is a closed-
    form solve. The unused ``log_tau`` parameter is retained for strict weight
    compatibility. With no anchors, predictions equal the baseline exactly.
    """

    def __init__(
        self,
        n_genes: int = 976,
        kind: str = "ridge_remix",
        rank: int = 8,
        n_feat: int = 64,
        init_alpha: float = -3.0,
    ) -> None:
        super().__init__()
        if kind not in ("ridge", "ridge_remix"):
            raise ValueError("kind must be 'ridge_remix' or 'ridge'")
        self.kind = kind
        self.n_feat = n_feat
        self.raw_alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        self.raw_gate = nn.Parameter(torch.full((n_genes,), -1.5))
        self.log_lam = nn.Parameter(torch.tensor(0.0))
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        self.raw_mix = nn.Parameter(torch.tensor(-4.0))
        self.U = nn.Parameter(0.01 * torch.randn(n_genes, rank))
        self.V = nn.Parameter(0.01 * torch.randn(rank, n_genes))

    def scale(self) -> Tensor:
        return torch.sigmoid(self.raw_alpha)

    def gate(self) -> Tensor:
        return torch.sigmoid(self.raw_gate)

    def lam(self) -> Tensor:
        return F.softplus(self.log_lam) + 0.05

    def remix(self, delta: Tensor) -> Tensor:
        return delta + torch.sigmoid(self.raw_mix) * ((delta @ self.V.T) @ self.U.T)

    def delta(
        self,
        p0: Tensor,
        y: Tensor,
        feat: Tensor,
        anchor_idx: Tensor,
        query_idx: Tensor,
        feat_alt: Tensor | None = None,
    ) -> Tensor:
        """Return ridge residuals, with the learned remix when configured."""
        correction = ridge_delta(p0, y, feat, anchor_idx, query_idx, lam=self.lam())
        return self.remix(correction) if self.kind == "ridge_remix" else correction

    def forward(
        self,
        p0: Tensor,
        y: Tensor,
        feat: Tensor,
        anchor_idx: Tensor,
        query_idx: Tensor,
        feat_alt: Tensor | None = None,
    ) -> Tensor:
        """Return adapted query responses; only anchor outcomes are consumed."""
        base = p0[query_idx]
        if anchor_idx.numel() == 0:
            return base
        correction = self.delta(p0, y, feat, anchor_idx, query_idx, feat_alt)
        return base + self.scale() * correction * self.gate()


def _seed_saturating(seed: int) -> None:
    """The saturating-mapping recipe uses the original deterministic initialization."""
    import random
    import numpy as np

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GRAFTSaturating(nn.Module):
    def __init__(self, genes=976, seed=52, device="cpu", initialization="direct"):
        super().__init__()
        if initialization == "pruned":
            _initialize_pruned_saturating(self, genes, seed, device)
            return
        if initialization != "direct":
            raise ValueError("Unknown saturating initialization protocol")

        _seed_saturating(int(seed))
        self.genes = int(genes)
        self.identity = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 32), nn.Tanh())
        self.ref = nn.Sequential(nn.Linear(self.genes, 16), nn.Tanh())
        self.diff = nn.Linear(self.genes, 16, bias=False)
        self.moment = nn.Linear(48 * 16, 32, bias=False)
        self.state = nn.Sequential(nn.Linear(32, 32), nn.Tanh())
        self.query = nn.Sequential(nn.Linear(256 + self.genes, 64), nn.GELU(), nn.LayerNorm(64))
        self.hyper = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 128))
        self.post = nn.Sequential(nn.LayerNorm(64), nn.GELU())
        self.output = nn.Linear(64, self.genes, bias=False)
        nn.init.normal_(self.output.weight, std=0.01)
        self.to(device)

    def summarize(self, p):
        raw = p["delta"].mean(0)
        f = torch.cat((self.identity(p["x"]), self.ref(p["mean"])), -1)
        f = f - f.mean(0)
        v = torch.tanh(self.diff(p["delta"] - raw))
        v = v - v.mean(0)
        f = f / f.square().mean(0).clamp_min(0.05 ** 2).sqrt()
        v = v / v.square().mean(0).clamp_min(0.05 ** 2).sqrt()
        return torch.tanh(self.moment(((f.T @ v) / len(f)).flatten()))

    def encode(self, support):
        raw = self.output.weight.new_zeros(32) if "genetic" not in support else self.summarize(support["genetic"])
        z = self.state(raw) - self.state(torch.zeros_like(raw))
        return dict(z=z, raw=raw)

    def decode(self, q, state, gain=1.0):
        h = self.query(torch.cat((q["x"], q["mean"]), -1))
        film = self.hyper(state["z"]) - self.hyper(torch.zeros_like(state["z"]))
        gamma, beta = film.chunk(2)
        return q["mean"] + gain * self.output(self.post(h * (1 + 0.5 * gamma.tanh()) + beta) - self.post(h))

    def forward(self, q, state, gain=1.0):
        return self.decode(q, state, gain=gain)

    @torch.no_grad()
    def inference(self, q, support, gain=1.0, chunk=128):
        self.eval()
        state = self.encode(support)
        return torch.cat([self.decode({k: v[i:i + chunk] for k, v in q.items()}, state, gain) for i in range(0, len(q["x"]), chunk)])


def _initialize_pruned_saturating(model, genes, seed, device):
    """Initialize the paper's pruned source-fold recipe without extra dependencies.

    Construction and slicing follow SetHyperModel -> R1 -> combo_safe, including
    discarded layers that consume the original random stream. The retained
    tensors use the same public parameter names as GRAFTSaturating.
    """
    _seed_saturating(seed)
    model.genes = genes
    model.identity = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 32), nn.Tanh())
    model.ref = nn.Sequential(nn.Linear(genes, 16), nn.Tanh())
    differences = nn.ModuleList([nn.Linear(genes, 16, bias=False) for _ in range(2)])
    backgrounds = nn.ModuleList([nn.Linear(genes, 32, bias=False) for _ in range(2)])
    tokens = nn.ModuleList([nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 32), nn.Tanh()) for _ in range(2)])
    state = nn.Sequential(nn.Linear(128, 32), nn.Tanh())
    query = nn.Sequential(nn.Linear(256 + genes * 2 + 1, 64), nn.GELU(), nn.LayerNorm(64))
    hyper = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 128))
    post = nn.Sequential(nn.LayerNorm(64), nn.GELU())
    output = nn.Linear(64, genes, bias=False)
    nn.init.normal_(output.weight, std=0.01)
    del tokens
    moments = nn.ModuleList([nn.Linear(48 * 16, 32, bias=False) for _ in range(2)])
    del backgrounds
    def sliced(old, columns):
        weight = old.weight.detach()[:, columns]
        new = nn.Linear(weight.shape[1], weight.shape[0], bias=old.bias is not None)
        with torch.no_grad():
            new.weight.copy_(weight)
            if old.bias is not None:
                new.bias.copy_(old.bias)
        return new
    query[0] = sliced(query[0], slice(0, 256 + genes))
    state[0] = sliced(state[0], slice(32, 64))
    model.diff = differences[0]
    model.state = state
    model.query = query
    model.hyper = hyper
    model.post = post
    model.output = output
    model.moment = moments[0]
    model.to(device)
