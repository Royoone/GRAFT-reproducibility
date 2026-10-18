"""Recompute the manuscript's fixed internal and external GRAFT benchmarks.

The checkpoint mapping is prescribed by the experiment, not selected at runtime.
Target chemical responses are sealed during inference and read only for scoring.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import warnings
import numpy as np
import torch
from scipy.stats import rankdata
from .data import FIVE, SEEDS, load_sources, query_names, sha256
from .model import GRAFT, GRAFTSaturating
from .evaluate import infer_sealed

EXTERNAL = ("HepG2", "MCF10A", "Jurkat", "K562_L1000", "HAP1")
EXPECTED_QUERIES = dict(HepG2=4530, MCF10A=1484, Jurkat=1404, K562_L1000=96, HAP1=86)
EXCLUDED = ("BRD-K43389675", "BRD-K49371609", "BRD-K17516382")

def corr(a,b,constant_pred_zero=True):
 ac=a-a.mean(1,keepdims=True);bc=b-b.mean(1,keepdims=True)
 na=np.sqrt((ac*ac).sum(1));nb=np.sqrt((bc*bc).sum(1))
 v=(ac*bc).sum(1)/np.maximum(na*nb,1e-12)
 v[(na==0)|(nb==0)]=np.nan
 if constant_pred_zero:v[(np.ptp(a,axis=1)>0)&(np.ptp(b,axis=1)==0)]=0
 return v

def score(y,p,names):
 y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
 if y.shape!=p.shape or len(names)!=len(y):raise ValueError('Shape mismatch')
 if not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('Nonfinite prediction or truth')
 if len(set(names))!=len(names):raise ValueError('Duplicate query names')
 n,g=y.shape;per={};flat={}
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning)
  for k in (20,50,100):
   ix=np.argpartition(np.abs(y),-k,axis=1)[:,-k:]
   jx=np.argpartition(np.abs(p),-k,axis=1)[:,-k:]
   yy=np.take_along_axis(y,ix,axis=1);pp=np.take_along_axis(p,ix,axis=1)
   r=corr(yy,pp);rho=corr(rankdata(yy,method='average',axis=1),rankdata(pp,method='average',axis=1))
   # Historical exclusion also affects rho20 because both use the same mask.
   if k==20:r[np.isin(names,EXCLUDED)]=np.nan;rho[np.isin(names,EXCLUDED)]=np.nan
   vals={'r':r,'rho':rho,'mse':((pp-yy)**2).mean(1),'mae':np.abs(pp-yy).mean(1)}
   vals['recall']=np.array([np.intersect1d(a,b,assume_unique=True).size/k for a,b in zip(ix,jx)])
   for key,sgn in [('pos_p',1),('neg_p',-1)]:
    a=np.argpartition(y*sgn,-k,axis=1)[:,-k:];b=np.argpartition(p*sgn,-k,axis=1)[:,-k:]
    vals[key]=np.array([np.intersect1d(aa,bb,assume_unique=True).size/k for aa,bb in zip(a,b)])
   for key,v in vals.items():per[f'{key}{k}']=v;flat[f'{key}{k}']=float(np.nanmean(v))
   flat[f'n_r{k}']=int(np.isfinite(r).sum());flat[f'rmse{k}']=float(np.sqrt(flat[f'mse{k}']))
   if k==20:per['sign20']=np.mean(np.sign(yy)==np.sign(pp),axis=1)
  # Matches GRAFT supplementary compute.py for full-signature correlation.
  per['all_r']=corr(y,p,constant_pred_zero=False)
  for key in ('all_r','sign20'):flat[key]=float(np.nanmean(per[key]))
 return flat,per

def load_external(path, genes, device):
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["genes"].astype(str), genes)
        row = {"genes": genes}
        for modality in ("genetic", "chemical"):
            names = archive[f"{modality}_names"].astype(str)
            x, y = archive[f"{modality}_embeddings"], archive[f"{modality}_native"]
            if x.shape != (len(names), 256) or y.shape != (len(names), len(genes)):
                raise ValueError(f"Invalid external {modality} axes")
            if len(set(names)) != len(names) or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f"Invalid external {modality} values or IDs")
            row[modality] = dict(names=names, x=torch.tensor(x, dtype=torch.float32, device=device),
                                  y=torch.tensor(y, dtype=torch.float32, device=device))
        names = archive["query_ids"].astype(str)
        if len(set(names)) != len(names) or not set(names).issubset(row["chemical"]["names"]):
            raise ValueError("External fixed query identities must be unique measured drug records")
    return row, names


def load_paper_model(checkpoint_root, target, seed, n_genes, device):
    cohort = target if target in FIVE else "external"
    folder = Path(checkpoint_root)/cohort/str(seed)
    metadata = json.loads((folder/"metadata.json").read_text())
    sources = [c for c in FIVE if c != target]
    mapping = "cross_moment_residual" if target in FIVE else "saturating"
    if metadata["mapping"] != mapping or list(metadata["sources"]) != sources or int(metadata["seed"]) != seed:
        raise ValueError("Checkpoint does not match the fixed manuscript experiment")
    checkpoint = folder/"model.npz"
    if sha256(checkpoint) != metadata["sha256"]:
        raise ValueError("Frozen checkpoint checksum mismatch")
    model = (GRAFT(n_genes, seed, device) if target in FIVE
             else GRAFTSaturating(n_genes, seed, device))
    with np.load(checkpoint, allow_pickle=False) as archive:
        if any(archive[k].dtype != np.float32 or not np.isfinite(archive[k]).all() for k in archive.files):
            raise ValueError("Invalid frozen backbone tensors")
        model.load_state_dict({k:torch.as_tensor(archive[k], device=device) for k in archive.files}, strict=True)
    model.eval().requires_grad_(False)
    scale = float(metadata["scale"])
    if not np.isfinite(scale) or scale <= 0 or metadata.get("gain",1) != 1:
        raise ValueError("Invalid trained response scale or gain")
    return model, scale, sources, checkpoint, metadata


def predict_target(data_root, checkpoint_root, target, seed, output, device="cpu"):
    data_root = Path(data_root)
    data = load_sources(data_root/"sources", device)
    genes = data[FIVE[0]]["genes"]
    if target in FIVE:
        names = query_names(data)
        if len(names) != 644:
            raise ValueError("The fixed internal benchmark requires 644 query drugs")
    elif target in EXTERNAL:
        data[target], names = load_external(data_root/"external"/f"{target}.npz", genes, device)
        if len(names) != EXPECTED_QUERIES[target]:
            raise ValueError("The external matched query panel differs from the paper contract")
    else:
        raise ValueError("Unknown manuscript benchmark context")
    model, scale, sources, checkpoint, metadata = load_paper_model(checkpoint_root, target, seed, len(genes), device)
    pred, truth, reference = infer_sealed(model, data, target, sources, names, scale)
    scores, per_drug = score(truth, pred, names)
    source_scores, _ = score(truth, reference, names)
    output = Path(output)/target/str(seed)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"predictions.npz").exists():
        raise FileExistsError(f"Benchmark predictions already exist: {output}")
    np.savez_compressed(output/"predictions.npz", names=names, genes=genes,
                        prediction=pred.astype(np.float32), truth=truth.astype(np.float32),
                        source_mean=reference.astype(np.float32))
    np.savez_compressed(output/"per_drug_scores.npz", names=names, **per_drug)
    record = dict(method="GRAFT", cohort="internal" if target in FIVE else "external",
                  cell=target, seed=seed, n_query=len(names), n_genetic=len(data[target]["genetic"]["names"]),
                  n_genes=len(genes), sources=sources, mapping=metadata["mapping"], scale=scale,
                  checkpoint_sha256=sha256(checkpoint),
                  original_checkpoint_sha256=metadata["original_checkpoint_sha256"],
                  target_drug_poison_passed=True, source_mean=source_scores, **scores)
    (output/"metrics.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoints", type=Path, default=Path("data/checkpoints/benchmark"))
    parser.add_argument("--output", type=Path, default=Path("results/benchmark_predictions"))
    parser.add_argument("--target", choices=(*FIVE, *EXTERNAL), required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    record = predict_target(args.data, args.checkpoints, args.target, args.seed, args.output, args.device)
    print(json.dumps({k:record[k] for k in ["cell","seed","n_query","r20","mse20"]}), flush=True)

if __name__ == "__main__":
    main()
