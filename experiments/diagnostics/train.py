"""Train one of the six capacity configurations displayed in FS2c."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..backbone.data import FIVE, SEEDS, episode, load_sources, pack, prepare, query_names, references, sha256, source_scale, training_weights
from ..backbone.evaluate import score
from ..backbone.train import model_hash, objective
from .evaluate import genetic_indices

CONFIGURATIONS = {
    "state16": dict(state_dim=16, width=64, hyper_depth=2, hyper_hidden=32, moment_out=32),
    "state64": dict(state_dim=64, width=64, hyper_depth=2, hyper_hidden=64, moment_out=32),
    "width32": dict(state_dim=32, width=32, hyper_depth=2, hyper_hidden=32, moment_out=32),
    "width128": dict(state_dim=32, width=128, hyper_depth=2, hyper_hidden=32, moment_out=32),
    "hyper1": dict(state_dim=32, width=64, hyper_depth=1, hyper_hidden=32, moment_out=32),
    "moment16": dict(state_dim=32, width=64, hyper_depth=2, hyper_hidden=32, moment_out=16),
}


class CapacityModel(nn.Module):
    """The fixed state, readout, hypernetwork and moment-size interventions."""
    def __init__(self, configuration, genes=976, seed=52, device="cpu"):
        super().__init__()
        cfg = CONFIGURATIONS[configuration]
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.genes = int(genes)
        self.state_dim = cfg["state_dim"]
        self.width = cfg["width"]
        self.hyper_depth = cfg["hyper_depth"]
        self.hyper_hidden = cfg["hyper_hidden"]
        self.moment_out = cfg["moment_out"]
        self.identity = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 32), nn.Tanh())
        self.ref = nn.Sequential(nn.Linear(self.genes, 16), nn.Tanh())
        self.diff = nn.Linear(self.genes, 16, bias=False)
        self.moment = nn.Linear(48 * 16, self.moment_out, bias=False)
        self.query = nn.Sequential(nn.Linear(256 + self.genes, self.width), nn.GELU(), nn.LayerNorm(self.width))
        if self.hyper_depth <= 1:
            self.hyper = nn.Sequential(nn.Linear(self.state_dim, 2 * self.width))
        else:
            self.hyper = nn.Sequential(nn.Linear(self.state_dim, self.hyper_hidden), nn.GELU(), nn.Linear(self.hyper_hidden, 2 * self.width))
        self.post = nn.Sequential(nn.LayerNorm(self.width), nn.GELU())
        self.output = nn.Linear(self.width, self.genes, bias=False)
        nn.init.normal_(self.output.weight, std=0.01)
        self.skip = nn.Linear(self.moment_out, self.state_dim, bias=False)
        self.mlp = nn.Sequential(nn.LayerNorm(self.moment_out), nn.Linear(self.moment_out, 2 * self.state_dim), nn.GELU(), nn.Linear(2 * self.state_dim, self.state_dim))
        nn.init.eye_(self.skip.weight)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.to(device)

    def bound(self, c):
        return c / c.square().mean().add(1e-8).sqrt().clamp_min(1.)

    def state(self, c):
        return self.skip(c) + self.mlp(c)

    def summarize(self, p):
        raw = p["delta"].mean(0)
        f = torch.cat((self.identity(p["x"]), self.ref(p["mean"])), -1)
        f = f - f.mean(0)
        v = torch.tanh(self.diff(p["delta"] - raw))
        v = v - v.mean(0)
        f = f / f.square().mean(0).clamp_min(.05 ** 2).sqrt()
        v = v / v.square().mean(0).clamp_min(.05 ** 2).sqrt()
        cov = (f.T @ v) / len(f)
        return self.bound(self.moment(cov.flatten()))

    def encode(self, support):
        raw = self.output.weight.new_zeros(self.moment_out) if "genetic" not in support else self.summarize(support["genetic"])
        z = self.state(raw) - self.state(torch.zeros_like(raw))
        return dict(z=z, raw=raw)

    def decode(self, q, state, gain=1.):
        h = self.query(torch.cat((q["x"], q["mean"]), -1))
        z = state["z"]
        film = self.hyper(z) - self.hyper(torch.zeros_like(z))
        gamma, beta = film.chunk(2)
        conditional = h * (1 + .5 * gamma.tanh()) + beta
        return q["mean"] + gain * self.output(self.post(conditional) - self.post(h))

    def forward(self, q, state, gain=1.):
        return self.decode(q, state, gain)

    @torch.no_grad()
    def inference(self, q, support, gain=1., chunk=128):
        self.eval()
        state = self.encode(support)
        return torch.cat([self.decode({key:value[lo:lo + chunk] for key, value in q.items()}, state, gain)
                          for lo in range(0, len(q["x"]), chunk)])


def fit_capacity(data, sources, configuration, seed, device="cpu", steps=200):
    """The original episodes, source weights, objective and optimizer order."""
    if set(data) != set(sources) or len(sources) != 4:
        raise ValueError("Capacity training receives only its four source contexts")
    scale = source_scale(data, sources)
    model = CapacityModel(configuration, len(data[sources[0]]["genes"]), seed, device)
    initial_hash = model_hash(model)
    cache = prepare(data, sources, scale, "real", seed)
    weights = training_weights(data, sources)
    rng = np.random.default_rng(seed + 1849)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
    history = []
    start = time.monotonic()
    for step in range(steps):
        support, query, indices = episode(data, sources, cache, scale, rng, step, "real")
        model.train()
        prediction = model(query, model.encode(support))
        loss = objective(prediction, query["y"], weights[indices["cell"]][indices["query"]])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite capacity training")
        optimizer.step()
        history.append(dict(step=step + 1, loss=float(loss.detach()), gradient_norm_before_clip=float(norm),
                            cell=indices["cell"], n_query=len(indices["query"]), genetic_support="genetic" in support))
    return model, scale, history, dict(initial_state_sha256=initial_hash, final_state_sha256=model_hash(model),
                                      seconds=time.monotonic() - start)


@torch.no_grad()
def evaluate_capacity(model, data, target, sources, names, scale, seed):
    refs = references(data, sources, data[target], ("genetic", "chemical"))
    lookup = {str(name): i for i, name in enumerate(data[target]["chemical"]["names"])}
    qi = np.array([lookup[str(name)] for name in names])
    q = pack(data[target], refs, "chemical", qi, scale, labels=False)
    saved = data[target]["chemical"]["y"]
    truth = saved[qi].cpu().numpy().astype(np.float64)
    predictions, rows = {}, []
    data[target]["chemical"]["y"] = torch.full_like(saved, float("nan"))
    try:
        for label in ("0", "20", "full"):
            gi = genetic_indices(len(data[target]["genetic"]["names"]), seed, 0, label)
            support = {"genetic": pack(data[target], refs, "genetic", gi, scale)} if len(gi) else {}
            prediction = (model.inference(q, support).cpu().numpy() * scale).astype(np.float64)
            predictions[label] = prediction
            rows.append(dict(cell=target, target=target, seed=seed, label=label, n_g=len(gi),
                             n_query=len(names), **score(truth, prediction, names)))
    finally:
        data[target]["chemical"]["y"] = saved
    return predictions, truth, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/sources"))
    parser.add_argument("--configuration", choices=tuple(CONFIGURATIONS), required=True)
    parser.add_argument("--target", choices=FIVE, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/diagnostics/capacity"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    dest = args.output / args.configuration / args.target / str(args.seed)
    if dest.exists():
        raise FileExistsError(dest)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = load_sources(args.data, args.device)
    names = query_names(data)
    if len(names) != 644:
        raise ValueError("Capacity evaluation requires the frozen 644 queries")
    sources = [cell for cell in FIVE if cell != args.target]
    model, scale, history, trajectory = fit_capacity({cell:data[cell] for cell in sources}, sources,
                                                     args.configuration, args.seed, args.device)
    predictions, truth, rows = evaluate_capacity(model, data, args.target, sources, names, scale, args.seed)
    dest.mkdir(parents=True)
    np.savez_compressed(dest / "model.npz", **{key:value.detach().cpu().numpy() for key, value in model.state_dict().items()})
    np.savez_compressed(dest / "predictions.npz", names=names, genes=data[args.target]["genes"], truth=truth,
                        **{key:predictions[key] for key in ("20", "full")})
    pd.DataFrame(rows).assign(variant=args.configuration).to_csv(dest / "metrics.csv", index=False)
    pd.DataFrame([dict(variant=args.configuration, cell=args.target, seed=args.seed, r=rows[-1]["r20"])]).to_csv(dest / "capacity.csv", index=False)
    pd.DataFrame(history).to_csv(dest / "history.csv", index=False)
    metadata = dict(complete=True, model="GRAFT capacity intervention", variant=args.configuration,
                    target=args.target, seed=args.seed, sources=sources, steps=200, scale=scale, gain=1.,
                    cfg=CONFIGURATIONS[args.configuration], n_params=sum(p.numel() for p in model.parameters()),
                    target_query_labels_sealed=True, checkpoint_sha256=sha256(dest / "model.npz"),
                    source_files={cell:sha256(args.data / f"{cell}.npz") for cell in sources}, **trajectory)
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (dest / "COMPLETE.json").write_text(json.dumps(dict(metadata, scores=rows), indent=2) + "\n")
    print(dest)


if __name__ == "__main__":
    main()
