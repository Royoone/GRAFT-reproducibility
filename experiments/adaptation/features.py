"""Extract the fixed Figure 4 backbone predictions and conditional features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..backbone.data import FIVE, SEEDS, load_sources, pack, references, sha256
from ..backbone.model import GRAFTSaturating

EXTERNAL = ("HepG2", "MCF10A", "Jurkat", "K562_L1000", "HAP1")


def load_external(path, genes, device):
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["genes"].astype(str), genes)
        row = {"genes": genes}
        for kind in ("genetic", "chemical"):
            names = archive[f"{kind}_names"].astype(str)
            x, y = archive[f"{kind}_embeddings"], archive[f"{kind}_native"]
            if x.shape != (len(names), 256) or y.shape != (len(names), len(genes)):
                raise ValueError(f"Invalid external {kind} axes")
            if not np.isfinite(x).all() or not np.isfinite(y).all() or len(set(names)) != len(names):
                raise ValueError(f"Invalid external {kind} values or duplicate names")
            row[kind] = dict(names=names, x=torch.tensor(x, dtype=torch.float32, device=device),
                             y=torch.tensor(y, dtype=torch.float32, device=device))
    return row


def load_fixed_backbone(checkpoint_root, cell, seed, genes, device):
    root = Path(checkpoint_root)
    folder = (root / "adaptation_backbone" / cell / str(seed) if cell in FIVE
              else root / "benchmark" / "external" / str(seed))
    metadata = json.loads((folder / "metadata.json").read_text())
    sources = [c for c in FIVE if c != cell]
    if list(metadata["sources"]) != sources:
        raise ValueError("Backbone source list does not match the paper fold")
    if metadata.get("mapping", "saturating") != "saturating":
        raise ValueError("Figure 4 uses its fixed saturating backbone recipe")
    initialization = "pruned" if cell in FIVE else "direct"
    if metadata.get("initialization", initialization) != initialization:
        raise ValueError("Backbone initialization does not match the fixed paper experiment")
    model = GRAFTSaturating(len(genes), seed, device, initialization=initialization)
    path = folder / "model.npz"
    if metadata.get("sha256") and sha256(path) != metadata["sha256"]:
        raise ValueError("Backbone tensor checksum mismatch")
    with np.load(path, allow_pickle=False) as archive:
        state = {k.replace("diff.0.", "diff.").replace("moment.0.", "moment."):
                 torch.as_tensor(archive[k], device=device) for k in archive.files}
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    scale = float(metadata["scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid frozen backbone response scale")
    return model, scale, sources, path


@torch.no_grad()
def extract(model, data, target, sources, names, embeddings, scale, *, chunk=128):
    """Retain the original rowwise feature algebra and source reference pack."""
    device = data[sources[0]]["chemical"]["x"].device
    observed = torch.full((len(names), len(data[target]["genes"])), float("nan"), device=device)
    positions = {str(n): i for i, n in enumerate(names)}
    for i, name in enumerate(data[target]["chemical"]["names"]):
        if str(name) in positions:
            observed[positions[str(name)]] = data[target]["chemical"]["y"][i]
    target_row = dict(data[target])
    target_row["chemical"] = dict(names=names, x=embeddings, y=observed)
    refs = references(data, sources, target_row, ("genetic", "chemical"))
    q = pack(target_row, refs, "chemical", np.arange(len(names)), scale, labels=False)
    gi = np.arange(len(target_row["genetic"]["names"]))
    support = {"genetic": pack(target_row, refs, "genetic", gi, scale)} if len(gi) else {}
    state = model.encode(support)
    pred, post, source, unipert = [], [], [], []
    for lo in range(0, len(names), chunk):
        qq = {k: v[lo:lo + chunk] for k, v in q.items()}
        h = model.query(torch.cat((qq["x"], qq["mean"]), -1))
        film = model.hyper(state["z"]) - model.hyper(torch.zeros_like(state["z"]))
        gamma, beta = film.chunk(2)
        conditional = h * (1 + .5 * gamma.tanh()) + beta
        feature = model.post(conditional)
        prediction = qq["mean"] + model.output(model.post(conditional) - model.post(h))
        pred.append(prediction.cpu().numpy() * scale)
        post.append(feature.cpu().numpy())
        source.append(qq["mean"].cpu().numpy() * scale)
        unipert.append(qq["x"].cpu().numpy())
    return dict(names=names, genes=data[target]["genes"], pred=np.concatenate(pred),
                post64=np.concatenate(post), src=np.concatenate(source),
                true=observed.cpu().numpy(), unipert=np.concatenate(unipert),
                z=state["z"].cpu().numpy(), scale=np.asarray(scale), sources=np.array(sources))


def generate(source_dir, external_dir, checkpoint_root, output, cell, seed, device="cpu"):
    data = load_sources(source_dir, device)
    genes = data[FIVE[0]]["genes"]
    if cell in FIVE:
        # Exactly the shared 3220 rows used by Figure 4's sealed query/pool lists.
        common = sorted(set.intersection(*(set(data[c]["chemical"]["names"]) for c in FIVE)))
        names = np.asarray(common, dtype=str)
        lookup = {str(n): i for i, n in enumerate(data[cell]["chemical"]["names"])}
        embeddings = data[cell]["chemical"]["x"][[lookup[n] for n in names]]
        if len(names) != 3220:
            raise ValueError("The fixed Figure 4 source panel must have 3220 shared drug records")
    else:
        data[cell] = load_external(Path(external_dir) / f"{cell}.npz", genes, device)
        union = set.union(*(set(data[c]["chemical"]["names"]) for c in FIVE))
        lookup = {str(n): i for i, n in enumerate(data[cell]["chemical"]["names"])}
        names = np.asarray(sorted(set(lookup) & union), dtype=str)
        embeddings = data[cell]["chemical"]["x"][[lookup[n] for n in names]]
    model, scale, sources, checkpoint = load_fixed_backbone(checkpoint_root, cell, seed, genes, device)
    result = extract(model, data, cell, sources, names, embeddings, scale)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{cell}_{seed}.npz"
    if path.exists():
        raise FileExistsError(f"Feature file already exists: {path}")
    np.savez_compressed(path, **result)
    (path.with_suffix(".json")).write_text(json.dumps(dict(cell=cell, seed=seed, sources=sources,
            checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), n_drugs=len(names),
            n_genes=len(genes), n_genetic=len(data[cell]["genetic"]["names"]),
            query_panel="fixed paper drug records; no outcome-based row selection"), indent=2) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/sources"))
    parser.add_argument("--external-data", type=Path, default=Path("data/processed/external"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("data/checkpoints"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/adaptation/features"))
    parser.add_argument("--cell", choices=(*FIVE, *EXTERNAL), required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    generate(args.data, args.external_data, args.checkpoint_root, args.output, args.cell, args.seed, args.device)


if __name__ == "__main__":
    main()
