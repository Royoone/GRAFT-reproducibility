"""Source-only training of the fixed adapter recipes from the manuscript."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .model import GatedAdapter, d0_preserve_loss
from ..backbone.data import FIVE as CELLS, SEEDS, load_sources, query_names

ARCHIVE_EXCLUDED_R20 = ("BRD-K43389675", "BRD-K49371609", "BRD-K17516382")
BUDGETS = (1, 2, 5, 10, 20)
TRAIN_BUDGETS = BUDGETS

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def save_checkpoint(checkpoint, path):
    folder = path.parent
    np.savez_compressed(folder / "adapter.npz", **{k:v.detach().cpu().numpy() for k,v in checkpoint["state_dict"].items()})
    write_json(folder / "metadata.json", {k:v for k,v in checkpoint.items() if k != "state_dict"})

def feat_for(kind, fold):
    return (fold["uni"], None) if kind == "kernel" else (fold["post"], None)

def load_fold(feat_dir: Path, cell: str, seed: int):
    path = feat_dir / f"{cell}_{seed}.npz"
    with np.load(path, allow_pickle=False) as z:
        names = z["names"].astype(str)
        return {
            "names": names,
            "p0": torch.tensor(z["pred"], dtype=torch.float32),
            "y": torch.tensor(z["true"], dtype=torch.float32),
            "post": torch.tensor(z["post64"], dtype=torch.float32),
            "uni": torch.tensor(z["unipert"], dtype=torch.float32),
        }

def query_and_pool(names, queries):
    lookup = {str(n): i for i, n in enumerate(names)}
    q_idx = torch.tensor([lookup[n] for n in queries], dtype=torch.long)
    shared = names[:3220]
    pool = [n for n in shared if n not in set(queries)]
    p_idx = torch.tensor([lookup[n] for n in pool], dtype=torch.long)
    keep20 = torch.tensor([n not in ARCHIVE_EXCLUDED_R20 for n in queries], dtype=torch.bool)
    return q_idx, p_idx, keep20

def to_device(fold, device):
    out = {"names": fold["names"]}
    for k in ("p0", "y", "post", "uni"):
        out[k] = fold[k].to(device)
    return out

@torch.no_grad()
def val_delta(model, fold, q_idx, p_idx, keep20, feat_key, k=20, draws=3):
    model.eval()
    feat, alt = feat_for(model.kind, fold)
    p0, y = fold["p0"], fold["y"]
    scores = []
    for draw in range(draws):
        rng = np.random.default_rng(1000 + draw)
        a = torch.tensor(rng.choice(p_idx.cpu().numpy(), size=k, replace=False), device=p0.device)
        pred = model(p0, y, feat, a, q_idx, feat_alt=alt)
        loss, stats = d0_preserve_loss(y[q_idx], pred, p0[q_idx], keep20)
        scores.append(stats["r20"] - stats["r0"])
    model.train()
    return float(np.mean(scores))

def train_target(feat_dir, out_dir, target, kind, device, queries, epochs=40, steps=16, lr=0.02, mu=0.25, nu=2.0):
    torch.manual_seed(52)
    torch.cuda.manual_seed_all(52)
    train_cells = [c for c in CELLS if c != target]
    val_cell = train_cells[-1]
    fit_cells = train_cells[:-1]
    feat_key = "uni" if kind in ("kernel", "conf_kernel") else "post"
    cache = {}
    for cell in train_cells:
        for seed in SEEDS:
            fold = to_device(load_fold(feat_dir, cell, seed), device)
            q_idx, p_idx, keep20 = query_and_pool(fold["names"], queries)
            cache[(cell, seed)] = (fold, q_idx.to(device), p_idx.to(device), keep20.to(device))

    model = GatedAdapter(kind=kind).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = {"val": -1e9, "state": None, "epoch": -1}
    log = []
    rng = np.random.default_rng(7 + sum(map(ord, target + kind)))

    for epoch in range(epochs):
        model.train()
        running = []
        for step in range(steps):
            cell = str(rng.choice(fit_cells))
            seed = int(rng.choice(SEEDS))
            fold, q_idx, p_idx, keep20 = cache[(cell, seed)]
            if rng.random() < 0.15:
                k = 0
                a = torch.zeros(0, dtype=torch.long, device=device)
            else:
                k = int(rng.choice(BUDGETS))
                pick = rng.choice(p_idx.cpu().numpy(), size=min(k, int(p_idx.numel())), replace=False)
                a = torch.tensor(pick, dtype=torch.long, device=device)
            feat, alt = feat_for(kind, fold)
            pred = model(fold["p0"], fold["y"], feat, a, q_idx, feat_alt=alt)
            loss, stats = d0_preserve_loss(fold["y"][q_idx], pred, fold["p0"][q_idx], keep20, mu=mu, nu=nu)
            if a.numel() == 0:
                loss = loss + 0.05 * model.scale()
            opt.zero_grad()
            if loss.requires_grad:
                loss.backward()
            else:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            stats["loss"] = float(loss.detach())
            stats["scale"] = float(model.scale().detach())
            stats["k"] = k
            running.append(stats)
        val = val_delta(model, *cache[(val_cell, 52)][:4], feat_key)
        rec = {
            "epoch": epoch,
            "train_r20": float(np.mean([s["r20"] for s in running])),
            "train_r0": float(np.mean([s["r0"] for s in running])),
            "train_drop": float(np.mean([s["drop"] for s in running])),
            "scale": float(np.mean([s["scale"] for s in running])),
            "val_delta": val,
        }
        log.append(rec)
        if val > best["val"]:
            best = {"val": val, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "epoch": epoch}
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(
                f"{target} {kind} ep{epoch:02d} r20={rec['train_r20']:.3f} "
                f"d0={rec['train_r0']:.3f} drop={rec['train_drop']:.3f} "
                f"scale={rec['scale']:.3f} valΔ={val:+.3f}",
                flush=True,
            )

    if best["state"] is None:
        best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    dest = out_dir / target / kind
    dest.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        {
            "state_dict": best["state"],
            "kind": kind,
            "target": target,
            "best_epoch": best["epoch"],
            "best_val_delta": best["val"],
            "train_cells": fit_cells,
            "val_cell": val_cell,
            "mu": mu,
            "nu": nu,
        },
        dest / "adapter.npz",
    )
    write_json(dest / "train_log.json", {"log": log, "best_epoch": best["epoch"], "best_val_delta": best["val"]})
    print(f"saved {dest} best_ep={best['epoch']} valΔ={best['val']:+.4f}", flush=True)

def train_external(feat_dir, out_dir, kind, device, queries, epochs=40, steps=16):
    torch.manual_seed(52)
    torch.cuda.manual_seed_all(52)
    fit_cells = list(CELLS)
    val_cell = "PC3"
    cache = {}
    for cell in fit_cells:
        for seed in SEEDS:
            fold = to_device(load_fold(feat_dir, cell, seed), device)
            q_idx, p_idx, keep20 = query_and_pool(fold["names"], queries)
            cache[(cell, seed)] = (fold, q_idx.to(device), p_idx.to(device), keep20.to(device))
    model = GatedAdapter(kind=kind).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    best = {"val": -1e9, "state": None, "epoch": -1}
    log = []
    rng = np.random.default_rng(7 + sum(map(ord, "external" + kind)))
    for epoch in range(epochs):
        model.train()
        running = []
        for _step in range(steps):
            cell = str(rng.choice(fit_cells))
            seed = int(rng.choice(SEEDS))
            fold, q_idx, p_idx, keep20 = cache[(cell, seed)]
            if rng.random() < 0.15:
                k = 0
                a = torch.zeros(0, dtype=torch.long, device=device)
            else:
                k = int(rng.choice(TRAIN_BUDGETS))
                pick = rng.choice(p_idx.cpu().numpy(), size=min(k, int(p_idx.numel())), replace=False)
                a = torch.tensor(pick, dtype=torch.long, device=device)
            feat, alt = feat_for(kind, fold)
            pred = model(fold["p0"], fold["y"], feat, a, q_idx, feat_alt=alt)
            loss, stats = d0_preserve_loss(fold["y"][q_idx], pred, fold["p0"][q_idx], keep20)
            if a.numel() == 0:
                loss = loss + 0.05 * model.scale()
            opt.zero_grad()
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            stats["loss"] = float(loss.detach())
            stats["scale"] = float(model.scale().detach())
            stats["k"] = k
            running.append(stats)
        val = val_delta(model, *cache[(val_cell, 52)][:4], "post")
        rec = {
            "epoch": epoch,
            "train_r20": float(np.mean([s["r20"] for s in running])),
            "train_r0": float(np.mean([s["r0"] for s in running])),
            "scale": float(np.mean([s["scale"] for s in running])),
            "val_delta": val,
        }
        log.append(rec)
        if val > best["val"]:
            best = {"val": val, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "epoch": epoch}
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"external {kind} ep{epoch:02d} r20={rec['train_r20']:.3f} d0={rec['train_r0']:.3f} "
                  f"scale={rec['scale']:.3f} valΔ={val:+.3f}", flush=True)
    if best["state"] is None:
        best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    dest = out_dir / "external" / kind
    dest.mkdir(parents=True, exist_ok=True)
    save_checkpoint({
        "state_dict": best["state"], "kind": kind, "target": "external",
        "best_epoch": best["epoch"], "best_val_delta": best["val"],
        "train_cells": fit_cells, "val_cell": val_cell,
    }, dest / "adapter.npz")
    write_json(dest / "train_log.json", {"log": log, "best_epoch": best["epoch"], "best_val_delta": best["val"]})
    print(f"saved {dest} best_ep={best['epoch']} valΔ={best['val']:+.4f}", flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed/sources"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=(*CELLS, "external"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--kind", choices=("ridge", "ridge_remix", "kernel"), default="ridge_remix", help=argparse.SUPPRESS)
    parser.add_argument("--epochs", type=int, default=40, help=argparse.SUPPRESS)
    parser.add_argument("--steps", type=int, default=16, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.target == "external" and args.kind == "kernel":
        parser.error("The external paper recipe uses ridge or ridge_remix")
    dest = args.output / args.target / args.kind
    if (dest / "adapter.npz").exists():
        raise FileExistsError(f"Adapter already exists: {dest}")
    torch.set_num_threads(2)
    queries = query_names(load_sources(args.data)).tolist()
    if len(queries) != 644:
        raise ValueError("The adapter recipe requires the frozen 644 source queries")
    if args.target == "external":
        train_external(args.features, args.output, args.kind, args.device, queries, args.epochs, args.steps)
    else:
        train_target(args.features, args.output, args.target, args.kind, args.device, queries, args.epochs, args.steps)

if __name__ == "__main__":
    main()
