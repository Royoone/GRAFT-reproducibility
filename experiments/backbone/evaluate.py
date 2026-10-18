"""Sealed held-out evaluation on the paper's fixed internal query panel."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .data import FIVE, pack, query_names, references

EXCLUDED_R20 = ("BRD-K43389675", "BRD-K49371609", "BRD-K17516382")


def score(truth, prediction, names):
    """Top-|truth| scores, preserving the paper's stated r20 exclusions."""
    y, p = np.asarray(truth, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 2 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Scoring requires matching finite drug-by-gene matrices")
    rows = np.arange(len(y))[:, None]
    result = {}
    for k in (20, 50, 100):
        ix = np.argpartition(np.abs(y), -k, axis=-1)[:, -k:]
        tk, pk = y[rows, ix], p[rows, ix]
        a, b = tk - tk.mean(1, keepdims=True), pk - pk.mean(1, keepdims=True)
        na, nb = np.sqrt((a * a).sum(1)), np.sqrt((b * b).sum(1))
        r = (a * b).sum(1) / np.maximum(na * nb, 1e-12)
        const_t, const_p = np.ptp(tk, axis=1) == 0, np.ptp(pk, axis=1) == 0
        r[const_t] = np.nan
        r[(~const_t) & const_p] = 0.
        keep = np.array([str(n) not in EXCLUDED_R20 for n in names]) if k == 20 else np.ones(len(y), bool)
        pix = np.argpartition(np.abs(p), -k, axis=-1)[:, -k:]
        recall = [len(set(a) & set(b)) / k for a, b in zip(ix, pix)]
        error = pk - tk
        result.update({f"r{k}": float(np.nanmean(r[keep])),
                       f"recall{k}": float(np.mean(recall)),
                       f"mse{k}": float(np.mean(error**2)),
                       f"mae{k}": float(np.mean(np.abs(error))),
                       f"n_r{k}": int(np.isfinite(r[keep]).sum())})
    return result


def _model_hash(model):
    value = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        value.update(key.encode())
        value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


@torch.no_grad()
def infer_sealed(model, data, target, sources, names, scale):
    """Keep target chemical labels outside prediction and leave weights fixed."""
    if target in sources:
        raise ValueError("The held-out target cannot occur in the source list")
    refs = references(data, sources, data[target], ("genetic", "chemical"))
    lookup = {str(n): i for i, n in enumerate(data[target]["chemical"]["names"])}
    qi = np.array([lookup[str(n)] for n in names], dtype=int)
    if len(set(qi)) != len(qi) or not len(qi):
        raise ValueError("Queries must be nonempty and unique")
    q = pack(data[target], refs, "chemical", qi, scale, labels=False)
    saved = data[target]["chemical"]["y"]
    truth = saved[qi].cpu().numpy().astype(np.float64)
    before = _model_hash(model)
    data[target]["chemical"]["y"] = torch.full_like(saved, float("nan"))
    try:
        gi = np.arange(len(data[target]["genetic"]["names"]))
        support = {"genetic": pack(data[target], refs, "genetic", gi, scale)} if len(gi) else {}
        model.eval()
        state = model.encode(support)
        parts = []
        for lo in range(0, len(qi), 128):
            qq = {k: v[lo:lo + 128] for k, v in q.items()}
            parts.append(model(qq, state, gain=1.).cpu().numpy() * scale)
        pred = np.concatenate(parts).astype(np.float64)
        source = (q["mean"].cpu().numpy() * scale).astype(np.float64)
    finally:
        data[target]["chemical"]["y"] = saved
    if _model_hash(model) != before:
        raise RuntimeError("Evaluation changed the frozen model parameters")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise FloatingPointError("Nonfinite prediction or held-out truth")
    return pred, truth, source


def evaluate_internal(model, data, target, sources, scale, output):
    if target not in FIVE or list(sources) != [c for c in FIVE if c != target]:
        raise ValueError("Internal evaluation requires the declared four-source fold")
    names = query_names(data)
    if len(names) != 644:
        raise ValueError(f"Expected the frozen 644-drug panel, found {len(names)}")
    pred, truth, source = infer_sealed(model, data, target, sources, names, scale)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "predictions.npz", names=names, genes=data[target]["genes"],
                        prediction=pred.astype(np.float32), truth=truth.astype(np.float32),
                        source_mean=source.astype(np.float32))
    info = {"target": target, "sources": list(sources), "n_query": len(names),
            "n_genetic": len(data[target]["genetic"]["names"]), "scale": scale,
            "target_drug_poison_passed": True, "model_parameters_unchanged": True,
            "query_names": names.tolist(), "prediction": score(truth, pred, names),
            "source_mean": score(truth, source, names)}
    (output / "metrics.json").write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")
    return info
