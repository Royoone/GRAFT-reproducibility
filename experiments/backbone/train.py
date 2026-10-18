"""Train one source-only GRAFT fold with the frozen 200-step recipe.

Example: python -m experiments.backbone.train --data data/processed/sources
         --output results/backbone --target A375 --seed 52
         --steps 200 --device cpu
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import FIVE, episode, load_sources, prepare, sha256, source_scale, training_weights
from .model import GRAFT, GRAFTSaturating


def objective(pred, truth, weights):
    """Original weighted MSE plus centred response cosine objective."""
    mse = (pred - truth).square()
    value = (mse * weights.detach()).sum(-1).mean()
    p = pred - pred.mean(-1, keepdim=True)
    t = truth - truth.mean(-1, keepdim=True)
    return .5 * value + .1 * (1 - F.cosine_similarity(p, t, dim=-1)).mean()


def model_hash(model):
    value = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value.update(name.encode())
        value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def train_one(data, sources, *, seed=52, mapping="cross_moment_residual", steps=200, device="cpu", initialization="direct"):
    """Train without accepting or consulting a held-out target context."""
    if set(data) != set(sources) or len(sources) < 2:
        raise ValueError("Training must receive only its declared source contexts")
    if mapping not in ("cross_moment_residual", "saturating") or steps < 1:
        raise ValueError("Use a declared mapping and positive training steps")
    if initialization not in ("direct", "pruned") or (mapping == "cross_moment_residual" and initialization != "direct"):
        raise ValueError("Initialization must match the fixed mapping recipe")
    scale = source_scale(data, sources)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Source chemical RMS scale must be finite and positive")
    cache = prepare(data, sources, scale, "real", seed)
    weights = training_weights(data, sources)
    genes = len(data[sources[0]]["genes"])
    np.random.seed(seed)
    model_class = GRAFT if mapping == "cross_moment_residual" else GRAFTSaturating
    model = (model_class(genes=genes, seed=seed, device=device) if mapping == "cross_moment_residual"
             else model_class(genes=genes, seed=seed, device=device, initialization=initialization))
    initial = model_hash(model)
    rng = np.random.default_rng(seed + 1849)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
    stream = hashlib.sha256()
    history = []
    started = time.monotonic()
    for step in range(steps):
        support, q, ix = episode(data, sources, cache, scale, rng, step, "real")
        stream.update(ix["cell"].encode())
        stream.update(ix["query"].tobytes())
        model.train()
        pred = model(q, model.encode(support))
        loss = objective(pred, q["y"], weights[ix["cell"]][ix["query"]])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise FloatingPointError(f"Nonfinite training at step {step}")
        opt.step()
        history.append(dict(step=step + 1, loss=float(loss.detach()),
                            gradient_norm_before_clip=float(norm), cell=ix["cell"],
                            n_query=len(ix["query"]), genetic_support="genetic" in support))
        if (step + 1) % 50 == 0 or step + 1 == steps:
            print(json.dumps(dict(step=step + 1, loss=history[-1]["loss"],
                                  seconds=time.monotonic() - started)), flush=True)
    metadata = dict(model="GRAFT",
                    mapping=mapping, initialization=initialization, seed=int(seed), steps=int(steps), sources=list(sources),
                    scale=scale, gain=1., optimizer="AdamW", learning_rate=.001,
                    weight_decay=1e-5, gradient_clip=1., episode_rng_seed_offset=1849,
                    loss="0.5*weighted_MSE + 0.1*(1-centred_cosine)",
                    weighting="frozen source signature-level drug-vs-rest weights_rest",
                    target_drug_training=False, input_amplitude_augmentation=False,
                    extra_difference_loss=False, n_genes=genes,
                    genes=data[sources[0]]["genes"].tolist(),
                    gene_panel_sha256=hashlib.sha256(("\n".join(data[sources[0]]["genes"].tolist()) + "\n").encode()).hexdigest(),
                    parameters=sum(p.numel() for p in model.parameters()),
                    query_stream_sha256=stream.hexdigest(), initial_state_sha256=initial,
                    final_state_sha256=model_hash(model),
                    runtime={"numpy": np.__version__, "torch": str(torch.__version__),
                             "device": str(device), "threads": torch.get_num_threads()},
                    seconds=time.monotonic() - started)
    return model, metadata, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=(*FIVE, "external"), required=True)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--mapping", choices=("cross_moment_residual", "saturating"), default="cross_moment_residual",
                        help=argparse.SUPPRESS)
    parser.add_argument("--initialization", choices=("direct", "pruned"), default="direct", help=argparse.SUPPRESS)
    parser.add_argument("--steps", type=int, default=200,
                        help="Frozen paper recipe uses 200; smaller values are explicitly smoke tests")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate the held-out internal 644-drug panel after training")
    args = parser.parse_args()
    if args.steps < 1 or args.threads < 1:
        parser.error("steps and threads must be positive")
    if args.evaluate and args.target == "external":
        parser.error("External target tables are evaluated separately; omit --evaluate for source-five training")
    sources = [c for c in FIVE if c != args.target]
    dest = args.output / args.target / args.mapping / str(args.seed)
    if (dest / "model.pt").exists():
        raise FileExistsError(f"Checkpoint already exists: {dest}; use a new output directory")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = load_sources(args.data, args.device, cells=sources)
    model, metadata, history = train_one(data, sources, seed=args.seed, mapping=args.mapping,
                                          steps=args.steps, device=args.device, initialization=args.initialization)
    metadata.update(target=args.target,
                    source_files={c: {"file": str((args.data / f"{c}.npz").resolve()),
                                      "sha256": sha256(args.data / f"{c}.npz")} for c in sources},
                    recipe_complete=args.steps == 200)
    dest.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), **metadata, history=history), dest / "model.pt")
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    (dest / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
    if args.evaluate:
        from .evaluate import evaluate_internal
        all_data = load_sources(args.data, args.device)
        evaluate_internal(model, all_data, args.target, sources, metadata["scale"], dest / "evaluation")


if __name__ == "__main__":
    main()
