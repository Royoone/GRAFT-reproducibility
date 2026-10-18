"""Recompute FS2a/b/d and FS4g-j from the fixed internal GRAFT checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..backbone.data import FIVE, SEEDS, load_sources, pack, query_names, references, sha256
from ..backbone.evaluate import EXCLUDED_R20, score
from ..backbone.predict import load_paper_model
from ..backbone.train import model_hash
from ..metrics import corr

BUDGETS = ("0", "1", "2", "5", "10", "20", "30", "50", "75", "100", "150", "200", "full")
DIMENSIONS = (0, 1, 2, 4, 8, 16, 24, 32)


def genetic_indices(n, seed, draw, label):
    """The paper's nested SeedSequence ordering, including its full-set order."""
    base = np.random.default_rng(np.random.SeedSequence([seed, 17])).permutation(n)
    order = base if draw == 0 else np.random.default_rng(np.random.SeedSequence([seed, 17, draw])).permutation(n)
    count = n if label == "full" else int(label)
    if not 0 <= count <= n:
        raise ValueError("Invalid genetic support count")
    return base if label == "full" else order[:count]


def inputs(data, target, sources, names, scale):
    refs = references(data, sources, data[target], ("genetic", "chemical"))
    lookup = {str(name): i for i, name in enumerate(data[target]["chemical"]["names"])}
    qi = np.array([lookup[str(name)] for name in names])
    q = pack(data[target], refs, "chemical", qi, scale, labels=False)
    truth = data[target]["chemical"]["y"][qi].cpu().numpy().astype(np.float64)
    ng = len(data[target]["genetic"]["names"])
    genetic = pack(data[target], refs, "genetic", np.arange(ng), scale)
    return refs, q, truth, genetic


@torch.no_grad()
def predict_state(model, query, state, scale):
    """Retain the original 256-query batches and float64 scale restoration."""
    z = state["z"]
    film = model.hyper(z) - model.hyper(torch.zeros_like(z))
    gamma, beta = film.chunk(2)
    parts = []
    for lo in range(0, len(query["x"]), 256):
        q = {key: value[lo:lo + 256] for key, value in query.items()}
        h = model.query(torch.cat((q["x"], q["mean"]), -1))
        parts.append(q["mean"] + model.output(model.post(h * (1 + .5 * gamma.tanh()) + beta) - model.post(h)))
    return torch.cat(parts).cpu().numpy().astype(np.float64) * scale


def metric_record(truth, prediction, names):
    result = score(truth, prediction, names)
    result["all_r"] = float(np.mean(corr(truth, prediction, constant_pred_zero=False)))
    return result


def per_drug(truth, prediction, names, target, seed, arm):
    ix = np.argpartition(np.abs(truth), -20, axis=1)[:, -20:]
    y, p = np.take_along_axis(truth, ix, 1), np.take_along_axis(prediction, ix, 1)
    r = corr(y, p)
    r[np.isin(names, EXCLUDED_R20)] = np.nan
    mse = ((p - y) ** 2).mean(1)
    return [dict(cell=target, seed=seed, drug=name, arm=arm, r20=float(rr), mse20=float(mm))
            for name, rr, mm in zip(names, r, mse)]


@torch.no_grad()
def evaluate_frozen(model, data, target, sources, names, scale, seed, task="all"):
    if target not in FIVE or list(sources) != [cell for cell in FIVE if cell != target]:
        raise ValueError("Diagnostics require the paper's internal four-source fold")
    if task not in ("all", "budgets", "dimensions", "controls"):
        raise ValueError("Unknown diagnostic task")
    refs, q, truth, genetic = inputs(data, target, sources, names, scale)
    ng = len(genetic["x"])
    full = model.encode({"genetic": genetic})
    rows, drug_rows = [], []

    def record(state, kind, label, draw=0, n_g=ng, donor="", drug=False):
        prediction = predict_state(model, q, state, scale)
        row = dict(cell=target, target=target, seed=seed, task=kind, label=str(label), draw=draw,
                   n_g=n_g, count=n_g, n_g_full=ng, n_query=len(names), donor=donor)
        row.update(metric_record(truth, prediction, names))
        rows.append(row)
        if drug:
            drug_rows.extend(per_drug(truth, prediction, names, target, seed, label))

    saved, before = data[target]["chemical"]["y"], model_hash(model)
    data[target]["chemical"]["y"] = torch.full_like(saved, float("nan"))
    try:
        if task in ("all", "budgets"):
            for label in BUDGETS:
                for draw in ((0,) if label in ("0", "full") else (0, 1, 2)):
                    gi = genetic_indices(ng, seed, draw, label)
                    support = {"genetic": pack(data[target], refs, "genetic", gi, scale)} if len(gi) else {}
                    record(model.encode(support), "g_budget", label, draw, len(gi))
        if task in ("all", "dimensions"):
            orders = [("topabs", np.argsort(-np.abs(full["z"].cpu().numpy()))),
                      ("random", np.random.default_rng(seed + 19).permutation(32))]
            for count in DIMENSIONS:
                for kind, order in orders:
                    mask = torch.zeros_like(full["z"])
                    mask[order[:count]] = 1
                    record({"z": full["z"] * mask}, "z_" + kind, count)
        if task in ("all", "controls"):
            record(full, "frozen", "full", drug=True)
            record({"z": torch.zeros_like(full["z"])}, "frozen", "state_off", n_g=0, drug=True)
            shared = sorted(set.intersection(*(set(map(str, data[cell]["genetic"]["names"])) for cell in FIVE)))
            lookup = {str(name): i for i, name in enumerate(data[target]["genetic"]["names"])}
            gi = np.array([lookup[name] for name in shared])
            matched = pack(data[target], refs, "genetic", gi, scale)
            record(model.encode({"genetic": matched}), "frozen", "full_matched", n_g=len(gi))
            for draw in range(5):
                text = (f"genetic-contribution:{target}:{seed}" if draw == 0
                        else f"graft-frozen-ablation:{target}:{seed}:draw{draw}")
                rng_seed = int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)
                order = np.random.default_rng(rng_seed).permutation(ng)
                ix = torch.as_tensor(order, device=genetic["x"].device)
                raw = dict(genetic)
                raw["y"] = genetic["y"][ix]
                raw["delta"] = raw["y"] - genetic["mean"]
                record(model.encode({"genetic": raw}), "frozen", "raw_response_shuffle", draw)
                residual = dict(genetic)
                residual["delta"] = genetic["delta"][ix]
                residual["y"] = genetic["mean"] + residual["delta"]
                record(model.encode({"genetic": residual}), "frozen", "residual_shuffle", draw)
            for donor in sources:
                lookup = {str(name): i for i, name in enumerate(data[donor]["genetic"]["names"])}
                di = np.array([lookup[name] for name in shared])
                replacement = dict(matched)
                replacement["y"] = data[donor]["genetic"]["y"][di] / scale
                replacement["delta"] = replacement["y"] - replacement["mean"]
                record(model.encode({"genetic": replacement}), "frozen", "source_cell_input", n_g=len(gi), donor=donor)
    finally:
        data[target]["chemical"]["y"] = saved
    if model_hash(model) != before:
        raise RuntimeError("Frozen diagnostics changed model parameters")
    return rows, drug_rows, full["z"].cpu().tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/sources"))
    parser.add_argument("--checkpoints", type=Path, default=Path("data/checkpoints/benchmark"))
    parser.add_argument("--target", choices=FIVE, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--task", choices=("all", "budgets", "dimensions", "controls"), default="all")
    parser.add_argument("--output", type=Path, default=Path("results/diagnostics/frozen"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    dest = args.output / args.target / str(args.seed) / args.task
    if dest.exists():
        raise FileExistsError(dest)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = load_sources(args.data, args.device)
    names = query_names(data)
    if len(names) != 644:
        raise ValueError("Diagnostics require the frozen 644 query identities")
    model, scale, sources, checkpoint, metadata = load_paper_model(args.checkpoints, args.target, args.seed, 976, args.device)
    rows, drug_rows, state = evaluate_frozen(model, data, args.target, sources, names, scale, args.seed, args.task)
    dest.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(dest / "metrics.csv", index=False)
    if drug_rows:
        pd.DataFrame(drug_rows).to_csv(dest / "per_drug.csv", index=False)
    (dest / "state.json").write_text(json.dumps(dict(cell=args.target, seed=args.seed, z=state), indent=2) + "\n")
    info = dict(complete=True, target=args.target, seed=args.seed, task=args.task, sources=sources,
                n_queries=len(names), scale=scale, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                target_query_labels_sealed=True, model_parameters_unchanged=True, mapping=metadata["mapping"],
                rows=len(rows), source_files={cell:sha256(args.data / f"{cell}.npz") for cell in FIVE})
    (dest / "COMPLETE.json").write_text(json.dumps(info, indent=2) + "\n")
    print(dest)


if __name__ == "__main__":
    main()
