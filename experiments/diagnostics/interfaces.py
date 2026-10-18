"""Retrain the eighteen conditional-interface controls used in Supplementary Fig. 2e.

These are fixed scientific component interventions: ten additions to the GRAFT
cross-moment encoder and eight replacements of that encoder. All registered
modules and their initialization order are retained, including modules unused
by a particular intervention. Query drug labels never condition prediction.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn

from ..backbone.data import (FIVE, SEEDS, episode, load_sources, pack, prepare,
                            query_names, references, source_scale, training_weights)
from ..backbone.model import GRAFT
from ..backbone.train import objective
from ..metrics import score


def seed_all(seed):
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

PLUS = {
    "attn_set": "self-attn on 256 G tokens, added to the moment state",
    "attn_cross": "query-to-G cross-attention on the drug stem",
    "graph_gene": "source-only gene kNN mix of G residuals before the moment",
    "token_pool": "per-G token mean-pool added to the moment",
    "hyper_lowrank": "z generates a rank-8 residual on the output",
    "maml1": "one first-order inner step on source D when the episode has it",
    "dual_ctx": "separate context branch on the source-mean profile",
    "ecfp_cat": "end-to-end ECFP4-256 concatenated into the query stem",
    "film_x2": "two successive FiLM stages from the same z",
    "attn_graph": "attn_set plus gene-graph mix",
}

REPLACE = {
    "nm_attn_set": dict(kind="attn_set", maml=False),
    "nm_token_pool": dict(kind="token_pool", maml=False),
    "nm_graph_gene": dict(kind="graph_gene", maml=False),
    "nm_attn_cross": dict(kind="attn_cross", maml=False),
    "nm_attn_graph": dict(kind="attn_graph", maml=False),
    "nm_dual_ctx": dict(kind="dual_ctx", maml=False),
    "nm_ecfp_cat": dict(kind="ecfp_cat", maml=False),
    "nm_maml_attn": dict(kind="attn_set", maml=True),
}

TOKEN_K = 256

class InterfaceModel(GRAFT):
    def __init__(self, kind, genes=976, seed=52, device="cpu", graph=None, no_moment=False, maml=False):
        if kind not in PLUS:
            raise KeyError(kind)
        seed_all(seed)
        extra_in = 256 if kind == "ecfp_cat" else 0
        super().__init__(genes=genes, seed=seed, device="cpu")
        self.width, self.state_dim, self.moment_out = 64, 32, 32
        self.kind = kind
        self.no_moment = bool(no_moment)
        self.maml = bool(maml)
        self.ecfp_dim = extra_in
        if extra_in:
            self.query = nn.Sequential(
                nn.Linear(256 + genes + extra_in, self.width),
                nn.GELU(),
                nn.LayerNorm(self.width),
            )
        self.tok_proj = nn.Sequential(nn.Linear(64, 32), nn.GELU())
        self.g_attn = nn.MultiheadAttention(32, 4, batch_first=True)
        self.g_pool = nn.Linear(32, self.moment_out)
        self.q_proj = nn.Linear(self.width, 32)
        self.cross_attn = nn.MultiheadAttention(32, 4, batch_first=True)
        self.cross_back = nn.Linear(32, self.width)
        self.token_mlp = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, self.moment_out))
        self.ctx = nn.Sequential(nn.Linear(genes, self.width), nn.GELU(), nn.LayerNorm(self.width))
        self.ctx_gate = nn.Parameter(torch.tensor(0.3))
        self.hyper2 = nn.Sequential(
            nn.Linear(self.state_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2 * self.width),
        )
        rank = 8
        self.lr_a = nn.Linear(self.state_dim, self.width * rank)
        self.lr_b = nn.Linear(self.state_dim, rank * genes)
        self.rank = rank
        if graph is not None:
            self.register_buffer("gene_graph", graph)
        else:
            self.gene_graph = None
        self.to(device)

    def _tokens(self, p):
        raw = p["delta"].mean(0)
        f = torch.cat((self.identity(p["x"]), self.ref(p["mean"])), -1)
        f = f - f.mean(0)
        delta = p["delta"]
        if self.kind in ("graph_gene", "attn_graph") and self.gene_graph is not None:
            delta = delta + delta @ self.gene_graph
        v = torch.tanh(self.diff(delta - raw))
        v = v - v.mean(0)
        f = f / f.square().mean(0).clamp_min(0.05 ** 2).sqrt()
        v = v / v.square().mean(0).clamp_min(0.05 ** 2).sqrt()
        tok = torch.cat((f, v), -1)
        n = tok.size(0)
        if n > TOKEN_K:
            idx = torch.randperm(n, device=tok.device)[:TOKEN_K]
            tok = tok[idx]
            f, v = f[idx], v[idx]
        return tok, f, v

    def summarize(self, p):
        tok, f, v = self._tokens(p)
        if self.no_moment:
            raw = self.output.weight.new_zeros(self.moment_out)
            if self.kind in ("attn_set", "attn_graph"):
                h = self.tok_proj(tok).unsqueeze(0)
                a, _ = self.g_attn(h, h, h, need_weights=False)
                raw = raw + torch.tanh(self.g_pool(a.mean(1).squeeze(0)))
            if self.kind in ("token_pool", "graph_gene"):
                raw = raw + torch.tanh(self.token_mlp(tok).mean(0))
            if self.kind == "attn_cross":
                raw = raw + torch.tanh(self.g_pool(self.tok_proj(tok).mean(0)))
            return raw
        cov = (f.T @ v) / max(len(f), 1)
        raw = self.bound(self.moment(cov.flatten()))
        if self.kind in ("attn_set", "attn_graph"):
            h = self.tok_proj(tok).unsqueeze(0)
            a, _ = self.g_attn(h, h, h, need_weights=False)
            raw = raw + torch.tanh(self.g_pool(a.mean(1).squeeze(0)))
        if self.kind == "token_pool":
            raw = raw + torch.tanh(self.token_mlp(tok).mean(0))
        return raw

    def encode(self, support):
        state = super().encode(support)
        if "genetic" in support and self.kind == "attn_cross":
            tok, _, _ = self._tokens(support["genetic"])
            state["gtok"] = self.tok_proj(tok)
        else:
            state["gtok"] = None
        return state

    def decode(self, q, state, gain=1.0, x=None, extra=None):
        xx = q["x"] if x is None else x
        if self.kind == "ecfp_cat":
            if extra is None:
                extra = xx.new_zeros((len(xx), self.ecfp_dim))
            h = self.query(torch.cat((xx, q["mean"], extra), -1))
        else:
            h = self.query(torch.cat((xx, q["mean"]), -1))
        if self.kind == "dual_ctx":
            h = h + self.ctx_gate.tanh() * self.ctx(q["mean"])
        if self.kind == "attn_cross" and state.get("gtok") is not None:
            qh = self.q_proj(h).unsqueeze(0)
            kv = state["gtok"].unsqueeze(0)
            ca, _ = self.cross_attn(qh, kv, kv, need_weights=False)
            h = h + self.cross_back(ca.squeeze(0))
        z = state["z"]
        film = self.hyper(z) - self.hyper(torch.zeros_like(z))
        gamma, beta = film.chunk(2)
        conditional = h * (1 + 0.5 * gamma.tanh()) + beta
        if self.kind == "film_x2":
            film2 = self.hyper2(z) - self.hyper2(torch.zeros_like(z))
            g2, b2 = film2.chunk(2)
            conditional = conditional * (1 + 0.5 * g2.tanh()) + b2
        post_c = self.post(conditional)
        post_h = self.post(h)
        resid = post_c - post_h
        out = self.output(resid)
        if self.kind == "hyper_lowrank":
            a = self.lr_a(z).view(self.width, self.rank)
            b = self.lr_b(z).view(self.rank, self.genes)
            out = out + resid @ a @ b
        return q["mean"] + gain * out

    def chemical_as_query(self, p):
        return {k: p[k] for k in ("x", "mean", "variance", "count") if k in p}

def gene_mix_matrix(data, sources, device, k=20, thresh=0.4):
    corrs = []
    for c in sources:
        y = data[c]["chemical"]["y"]
        z = y - y.mean(0, keepdim=True)
        z = z / torch.linalg.vector_norm(z, dim=0, keepdim=True).clamp_min(1e-8)
        corrs.append(z.T @ z)
    corr = torch.stack(corrs).mean(0)
    corr.fill_diagonal_(-2)
    values, columns = torch.topk(corr, min(k, corr.size(0) - 1), dim=1)
    rows = torch.arange(corr.size(0), device=corr.device)[:, None].expand_as(columns)
    keep = values > thresh
    a = torch.zeros_like(corr)
    a[rows[keep], columns[keep]] = values[keep]
    a = 0.5 * (a + a.T)
    deg = a.sum(1).clamp_min(1e-6)
    a = a / deg[:, None]
    return a.detach().to(device)

VARIANTS = tuple(PLUS) + tuple(REPLACE)
INNER_LR = .05


def resolve_variant(name):
    if name in REPLACE:
        spec = REPLACE[name]
        return spec["kind"], True, bool(spec["maml"])
    if name in PLUS:
        return name, False, False
    raise ValueError("Unknown interface intervention: " + name)


def build(variant, genes, seed, device, graph=None):
    kind, no_moment, maml = resolve_variant(variant)
    return InterfaceModel(kind, genes, seed, device, graph=graph,
                          no_moment=no_moment, maml=maml)


def load_ecfp_tables(annotation_file, data):
    """Exact radius-2, 256-bit fingerprints; missing/invalid structures are zero."""
    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import AllChem
    recs = {}
    for row in pd.read_csv(annotation_file).itertuples():
        smi = getattr(row, "SMILES", None)
        if isinstance(smi, str) and smi:
            recs[str(row.Compound)] = smi
    tables = {}
    with rdBase.BlockLogs():
        for cell in FIVE:
            names = np.asarray(data[cell]["chemical"]["names"]).astype(str)
            mat = np.zeros((len(names), 256), dtype=np.float32)
            for i, name in enumerate(names):
                smi = recs.get(str(name))
                mol = Chem.MolFromSmiles(smi) if smi else None
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256)
                    DataStructs.ConvertToNumpyArray(fp, mat[i])
            tables[cell] = mat
    return tables


def names_to_ecfp(table, names, lookup_names, device):
    lookup = {str(n): i for i, n in enumerate(np.asarray(lookup_names).astype(str))}
    return torch.as_tensor(table[np.asarray([lookup[str(n)] for n in names])], device=device)


def inner_adapt(model, support, state):
    """One first-order source-drug step; restore parameters before the outer update."""
    if not (model.kind == "maml1" or model.maml) or "chemical" not in support:
        return None
    p = support["chemical"]
    parameters = [p for p in list(model.hyper.parameters()) + list(model.output.parameters())
                  if p.requires_grad]
    prediction = model.decode(model.chemical_as_query(p), state)
    loss = (prediction - p["y"]).square().mean()
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    backup = [p.detach().clone() for p in parameters]
    with torch.no_grad():
        for p, gradient in zip(parameters, gradients):
            if gradient is not None:
                p.add_(-INNER_LR * gradient)
    return backup, parameters


def restore(bundle):
    if bundle is not None:
        backup, parameters = bundle
        with torch.no_grad():
            for p, original in zip(parameters, backup):
                p.copy_(original)


def support_indices(n, seed, label):
    """The fixed draw-0 genetic ordering used by the component evaluation."""
    order = np.random.default_rng(np.random.SeedSequence([seed, 17])).permutation(n)
    if label == "full":
        return order
    count = int(label)
    if not 0 <= count <= n:
        raise ValueError("The requested genetic-support count exceeds the target data")
    return order[:count]


@torch.no_grad()
def evaluate(model, data, target, sources, names, scale, seed, label, ecfp=None):
    ref = references(data, sources, data[target], ("genetic", "chemical"))
    lookup = {str(n): i for i, n in enumerate(data[target]["chemical"]["names"])}
    qi = np.asarray([lookup[str(n)] for n in names], dtype=int)
    query = pack(data[target], ref, "chemical", qi, scale, labels=False)
    gi = support_indices(len(data[target]["genetic"]["names"]), seed, label)
    support = {"genetic": pack(data[target], ref, "genetic", gi, scale)} if len(gi) else {}
    extra = (names_to_ecfp(ecfp[target], names, data[target]["chemical"]["names"], query["x"].device)
             if model.kind == "ecfp_cat" else None)
    model.eval()
    state = model.encode(support)
    chunks = []
    for i in range(0, len(names), 128):
        q = {k: v[i:i + 128] for k, v in query.items()}
        ex = extra[i:i + 128] if extra is not None else None
        chunks.append(model.decode(q, state, gain=1., extra=ex))
    prediction = (torch.cat(chunks).float().cpu().numpy() * scale).astype(np.float64)
    # Sealed target drug responses are accessed only after inference.
    truth = data[target]["chemical"]["y"][qi].float().cpu().numpy().astype(np.float64)
    metrics, _ = score(truth, prediction, names)
    return prediction, truth, metrics


def train_one(data, target, seed, variant, output, device, ecfp=None, *, smoke=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Interface output already exists: " + str(output))
    sources = [cell for cell in FIVE if cell != target]
    src = {cell: data[cell] for cell in sources}
    kind, no_moment, maml = resolve_variant(variant)
    if kind == "ecfp_cat" and ecfp is None:
        raise ValueError("The ECFP4 interface requires its fixed fingerprint table")
    scale = source_scale(src, sources)
    graph = gene_mix_matrix(src, sources, device) if kind in ("graph_gene", "attn_graph") else None
    model = build(variant, len(src[sources[0]]["genes"]), seed, device, graph)
    cache = prepare(src, sources, scale, "real", seed)
    weights = training_weights(src, sources)
    rng = np.random.default_rng(seed + 1849)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
    steps = 10 if smoke else 200
    history = []
    start = time.monotonic()
    for step in range(steps):
        support, query, ix = episode(src, sources, cache, scale, rng, step, "real")
        model.train()
        extra = None
        if kind == "ecfp_cat":
            names = src[ix["cell"]]["chemical"]["names"][ix["query"]]
            extra = names_to_ecfp(ecfp[ix["cell"]], names,
                                  src[ix["cell"]]["chemical"]["names"], query["x"].device)
        state = model.encode(support)
        bundle = inner_adapt(model, support, state)
        prediction = model.decode(query, state, extra=extra)
        loss = objective(prediction, query["y"], weights[ix["cell"]][ix["query"]])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        restore(bundle)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise FloatingPointError(f"Nonfinite interface training: {variant}, step {step}")
        optimizer.step()
        history.append({"step":step+1,"loss":float(loss.detach()),"source":ix["cell"],
                        "source_drug_inner_step":bundle is not None})
    names = query_names(data)
    if smoke:
        names = names[:8]
    scores, predictions = {}, {}
    for label in ("0", "20", "full"):
        prediction, truth, metrics = evaluate(model, data, target, sources, names, scale,
                                               seed, label, ecfp)
        scores[label] = metrics
        if label != "0":
            predictions[label] = prediction
    record = {"complete":True,"paper_training_complete":not smoke,"smoke":smoke,
              "variant":variant,"kind":kind,"no_moment":no_moment,"maml":maml,
              "target":target,"seed":seed,"sources":sources,"scale":scale,"steps":steps,
              "n_params":sum(p.numel() for p in model.parameters()),"scores":scores,
              "history":history,"seconds":time.monotonic()-start,"n_query":len(names),
              "target_drug_training":False,"target_query_labels_used_for_selection":False,
              "checkpoint_selection":"fixed final step; no target performance selection",
              "graph_training_sources":sources if graph is not None else [],
              "token_limit":256,"inner_learning_rate":INNER_LR if kind=="maml1" or maml else None}
    output.mkdir(parents=True, exist_ok=False)
    torch.save({"state_dict":model.state_dict(),"variant":variant,"seed":seed,
                "sources":sources,"scale":scale,"steps":steps}, output/"model.pt")
    np.savez_compressed(output/"predictions.npz",names=names,truth=truth,**predictions)
    (output/"COMPLETE.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
    return {"variant":variant,"cell":target,"seed":seed,"n_params":record["n_params"],
            "r20_full":scores["full"]["r20"],"r20_20":scores["20"]["r20"],
            "smoke":smoke,"steps":steps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("data/processed/sources"))
    parser.add_argument("--annotations",type=Path,default=Path("data/annotations/compounds.csv"))
    parser.add_argument("--target",choices=FIVE,required=True)
    parser.add_argument("--seed",type=int,choices=SEEDS,default=52)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--variants",choices=VARIANTS,nargs="+",default=list(VARIANTS))
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--smoke",action="store_true",help="Ten source episodes, including inner steps, and eight evaluation queries")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if len(args.variants) != len(set(args.variants)):
        parser.error("Specify each scientific intervention once")
    destinations = {v:args.output/v/args.target/str(args.seed) for v in args.variants}
    if any(p.exists() for p in destinations.values()):
        parser.error("An output fold already exists; choose a fresh --output directory")
    data = load_sources(args.data,args.device)
    if len(query_names(data)) != 644:
        raise ValueError("The component experiment requires the fixed 644-query panel")
    need_ecfp = any(resolve_variant(v)[0]=="ecfp_cat" for v in args.variants)
    ecfp = load_ecfp_tables(args.annotations,data) if need_ecfp else None
    rows = []
    for variant in args.variants:
        row = train_one(data,args.target,args.seed,variant,destinations[variant],args.device,ecfp,smoke=args.smoke)
        rows.append(row)
        print(json.dumps(row),flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output/f"interfaces_{args.target}_{args.seed}.csv",index=False)


if __name__ == "__main__":
    main()
