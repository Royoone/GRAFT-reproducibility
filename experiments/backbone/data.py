"""Frozen source-table loading and the original GRAFT episode construction.

The reference, packing and episode algebra retains the archived operation and
RNG order. The portable data files add weights_rest to the seven source fields;
no archive results directory or third-party module is required.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

FIVE = ("A375", "A549", "HT29", "MCF7", "PC3")
CELLS = FIVE
SEEDS = (52, 229, 564)
QUERY_SALT = "cancer_fraction_screen_v1:"


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _strings(array, label):
    if array.ndim != 1 or array.dtype.kind not in ("U", "S"):
        raise ValueError(f"{label} must be a one-dimensional non-pickle string array")
    result = array.astype(str)
    if len(set(result)) != len(result) or any(not x or x != x.strip() for x in result):
        raise ValueError(f"{label} must contain unique, nonempty, unpadded names")
    return result


def load_sources(folder, device="cpu", cells=FIVE):
    """Read prepared source NPZs without sorting any row or gene axis."""
    data, genes = {}, None
    for cell in cells:
        path = Path(folder) / f"{cell}.npz"
        with np.load(path, allow_pickle=False) as archive:
            panel = _strings(archive["genes"], f"{cell} genes")
            if len(panel) != 976:
                raise ValueError(f"{cell} requires the frozen 976-gene panel")
            if genes is None:
                genes = panel.copy()
            np.testing.assert_array_equal(genes, panel)
            row = {"genes": genes}
            for modality in ("genetic", "chemical"):
                names = _strings(archive[f"{modality}_names"], f"{cell} {modality} IDs")
                x = archive[f"{modality}_embeddings"]
                y = archive[f"{modality}_native"]
                if x.shape != (len(names), 256) or y.shape != (len(names), len(genes)):
                    raise ValueError(f"{cell} {modality} dimensions do not match its named axes")
                if x.dtype != np.float32 or y.dtype != np.float32:
                    raise ValueError("The frozen embeddings and native responses must be float32")
                if not np.isfinite(x).all() or not np.isfinite(y).all():
                    raise ValueError(f"Nonfinite {cell} {modality} values")
                row[modality] = {"names": names.copy(), "x": torch.tensor(x, device=device),
                                 "y": torch.tensor(y, device=device)}
            weights = archive["weights_rest"]
            if weights.shape != row["chemical"]["y"].shape or weights.dtype != np.float32:
                raise ValueError(f"{cell} weights_rest must match chemical rows and genes, in float32")
            if not np.isfinite(weights).all() or (weights < 0).any():
                raise ValueError(f"Invalid {cell} weights_rest")
            if not np.allclose(weights.sum(1), 1, atol=1e-5, rtol=1e-5):
                raise ValueError(f"{cell} gene weights must sum to one per drug")
            row["weights_rest"] = weights.copy()
            data[cell] = row
    return data


def load(folder, device="cpu", arm="real"):
    if arm != "real":
        raise ValueError("The portable training entry point uses the frozen real-response recipe")
    return load_sources(folder, device)


def references(data, sources, context, modalities):
    if not sources or any(data[c] is context for c in sources):
        raise ValueError("Reference leakage")
    out = {}
    for m in modalities:
        names = context[m]["names"]
        lookup = {str(n): i for i, n in enumerate(names)}
        example = data[sources[0]][m]["y"]
        total = example.new_zeros((len(names), example.shape[1]))
        sq = torch.zeros_like(total)
        count = example.new_zeros(len(names))
        for c in sources:
            p = data[c][m]
            pairs = [(lookup[str(n)], i) for i, n in enumerate(p["names"]) if str(n) in lookup]
            if pairs:
                dst, src = map(list, zip(*pairs))
                y = p["y"][src]
                total[dst] += y
                sq[dst] += y.square()
                count[dst] += 1
        fallback = torch.stack([data[c][m]["y"].mean(0) for c in sources]).mean(0)
        mu = total / count.clamp_min(1)[:, None]
        mean = torch.where(count[:, None] > 0, mu, fallback[None, :])
        out[m] = dict(mean=mean,
                      variance=(sq / count.clamp_min(1)[:, None] - mu.square()).clamp_min(0),
                      count=count)
    return out


def pack(context, ref, m, ix, scale, *, labels=True):
    p, r = context[m], ref[m]
    out = dict(x=p["x"][ix], mean=r["mean"][ix] / scale,
               variance=r["variance"][ix] / scale**2, count=r["count"][ix, None])
    if labels:
        out["y"] = p["y"][ix] / scale
        out["delta"] = out["y"] - out["mean"]
    return out


def permute(p, cell, seed):
    value = int(hashlib.sha256(f"genetic-contribution:{cell}:{seed}".encode()).hexdigest()[:16], 16)
    ix = np.random.default_rng(value).permutation(len(p["x"]))
    result = dict(p)
    result["y"] = p["y"][ix]
    result["delta"] = result["y"] - result["mean"]
    return result


def prepare(data, sources, scale, arm="real", seed=0):
    mods = ("chemical",) if arm == "none" else ("genetic", "chemical")
    refs = {c: references(data, [s for s in sources if s != c], data[c], mods) for c in sources}
    gs = {}
    if arm != "none":
        for c in sources:
            p = pack(data[c], refs[c], "genetic", np.arange(len(data[c]["genetic"]["x"])), scale)
            gs[c] = permute(p, c, seed) if arm == "shuffle" else p
    return refs, gs


def episode(data, sources, cache, scale, rng, step, arm="real"):
    refs, gs = cache
    c = sources[int(rng.integers(len(sources)))]
    p = data[c]["chemical"]
    order = rng.permutation(len(p["x"]))
    n = min(128, len(order) // 2)
    qi, ci = order[:n], order[n:n + 128]
    support = {}
    if arm != "none" and step % 10 != 8:
        support["genetic"] = gs[c]
    if step % 10 in (8, 9):
        support["chemical"] = pack(data[c], refs[c], "chemical", ci, scale)
    assert not set(ci) & set(qi)
    return support, pack(data[c], refs[c], "chemical", qi, scale), dict(cell=c, query=qi)


def source_scale(data, sources):
    return float(torch.stack([data[c]["chemical"]["y"].square().mean() for c in sources]).mean().sqrt())


def training_weights(data, sources):
    return {c: torch.tensor(data[c]["weights_rest"], device=data[c]["chemical"]["y"].device)
            for c in sources}


def query_names(data):
    common = set.intersection(*(set(map(str, data[c]["chemical"]["names"])) for c in FIVE))
    ordered = sorted(common, key=lambda n: hashlib.sha256((QUERY_SALT + n).encode()).hexdigest())
    return np.asarray(ordered[:len(ordered) // 5], dtype=str)
