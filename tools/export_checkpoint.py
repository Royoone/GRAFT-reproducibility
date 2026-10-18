"""Export a completed GRAFT training checkpoint for the fixed NPZ loaders.

The output is one explicitly chosen fold directory, never a checkpoint root.
No packaged checkpoint is selected or replaced automatically.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.backbone.data import FIVE, SEEDS
from experiments.backbone.model import GRAFT, GRAFTSaturating


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def state_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def validate(checkpoint):
    require(isinstance(checkpoint, dict), "Expected a GRAFT training-checkpoint dictionary")
    require(checkpoint.get("model") == "GRAFT" and "state_dict" in checkpoint,
            "Expected the schema written by experiments.backbone.train")
    require(type(checkpoint.get("steps")) is int and checkpoint["steps"] == 200
            and checkpoint.get("recipe_complete") is True,
            "Export requires a complete 200-step training checkpoint; diagnostics cannot be exported")
    required = ("mapping", "initialization", "seed", "sources", "target", "scale", "gain",
                "n_genes", "genes", "gene_panel_sha256", "parameters", "final_state_sha256",
                "query_stream_sha256", "initial_state_sha256", "source_files", "history", "runtime")
    require(all(key in checkpoint for key in required), "Incomplete GRAFT training metadata")
    seed, target = checkpoint["seed"], checkpoint["target"]
    require(type(seed) is int and seed in SEEDS, "Seed must match one of the three manuscript seeds")
    require(target in (*FIVE, "external"), "Invalid source-fold target")
    sources = [cell for cell in FIVE if cell != target]
    require(checkpoint["sources"] == sources, "Source contexts or their order do not match the fold")
    require(isinstance(checkpoint["source_files"], dict)
            and set(checkpoint["source_files"]) == set(sources), "Incomplete source-file provenance")
    for name in sources:
        row = checkpoint["source_files"][name]
        require(isinstance(row, dict) and isinstance(row.get("file"), str)
                and isinstance(row.get("sha256"), str) and len(row["sha256"]) == 64,
                "Invalid source-file provenance: " + name)
    for key in ("scale", "gain"):
        value = checkpoint[key]
        require(type(value) in (float, int) and math.isfinite(value), "Invalid " + key)
    require(checkpoint["scale"] > 0 and checkpoint["gain"] == 1., "Invalid response scale or gain")
    recipe = {"optimizer": "AdamW", "learning_rate": .001, "weight_decay": 1e-5,
              "gradient_clip": 1., "episode_rng_seed_offset": 1849,
              "target_drug_training": False, "input_amplitude_augmentation": False,
              "extra_difference_loss": False,
              "loss": "0.5*weighted_MSE + 0.1*(1-centred_cosine)",
              "weighting": "frozen source signature-level drug-vs-rest weights_rest"}
    require(all(key in checkpoint and checkpoint[key] == value for key, value in recipe.items()),
            "Checkpoint does not describe the fixed GRAFT training recipe")
    genes = checkpoint["genes"]
    require(type(checkpoint["n_genes"]) is int and checkpoint["n_genes"] == 976
            and isinstance(genes, list) and len(genes) == 976
            and all(isinstance(gene, str) and gene for gene in genes)
            and len(set(genes)) == 976, "Expected the ordered 976-gene panel")
    gene_hash = hashlib.sha256(("\n".join(genes) + "\n").encode()).hexdigest()
    require(gene_hash == checkpoint["gene_panel_sha256"], "Gene-panel checksum mismatch")
    history = checkpoint["history"]
    require(isinstance(history, list) and len(history) == 200, "Expected all 200 training-history records")
    for step, row in enumerate(history, 1):
        require(isinstance(row, dict) and row.get("step") == step and row.get("cell") in sources,
                "Invalid training-history step or source context")
        for key in ("loss", "gradient_norm_before_clip"):
            value = row.get(key)
            require(type(value) in (float, int) and math.isfinite(value), "Nonfinite training history")
    mapping, initialization = checkpoint["mapping"], checkpoint["initialization"]
    require(mapping in ("cross_moment_residual", "saturating"), "Unsupported mapping")
    require(initialization in ("direct", "pruned")
            and (mapping != "cross_moment_residual" or initialization == "direct"),
            "Initialization does not match the mapping")
    model = (GRAFT(976, seed, "cpu") if mapping == "cross_moment_residual"
             else GRAFTSaturating(976, seed, "cpu", initialization=initialization))
    state = checkpoint["state_dict"]
    require(isinstance(state, Mapping) and set(state) == set(model.state_dict()),
            "Model parameter keys do not match the declared architecture")
    arrays = {}
    for name, expected in model.state_dict().items():
        tensor = state[name]
        require(isinstance(tensor, torch.Tensor) and tensor.layout == torch.strided
                and tensor.dtype == torch.float32 and tensor.shape == expected.shape,
                "Invalid float32 parameter shape or type: " + name)
        require(bool(torch.isfinite(tensor).all()), "Nonfinite model parameter: " + name)
        arrays[name] = tensor.detach().cpu().contiguous().numpy()
    require(checkpoint["parameters"] == sum(p.numel() for p in model.parameters()),
            "Parameter-count metadata mismatch")
    require(state_sha256(state) == checkpoint["final_state_sha256"], "Final model-state checksum mismatch")
    model.load_state_dict(state, strict=True)
    metadata = {key: value for key, value in checkpoint.items() if key not in ("state_dict", "history")}
    # Reject unsupported metadata types or NaN values before creating output.
    json.dumps(metadata, allow_nan=False)
    return arrays, metadata


def export_checkpoint(input_path, output):
    input_path, output = Path(input_path), Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new fold directory: {output}")
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=True)
    arrays, metadata = validate(checkpoint)
    original_hash = sha256(input_path)
    output.mkdir(parents=True, exist_ok=False)
    npz_path, metadata_path = output / "model.npz", output / "metadata.json"
    try:
        np.savez_compressed(npz_path, **arrays)
        metadata.update(file="model.npz", original_checkpoint_sha256=original_hash,
                        sha256=sha256(npz_path), bytes=npz_path.stat().st_size)
        metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    except Exception:
        npz_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        output.rmdir()
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="model.pt written by backbone training")
    parser.add_argument("--output", type=Path, required=True, help="New single-fold directory containing model.npz and metadata.json")
    args = parser.parse_args()
    try:
        result = export_checkpoint(args.input, args.output)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(result)


if __name__ == "__main__":
    main()
