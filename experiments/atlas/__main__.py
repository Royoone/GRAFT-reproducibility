"""Recompute the fixed manuscript Atlas13 analysis from prediction/observed arrays."""
import argparse,json,os,hashlib
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--input',type=Path,required=True,help='atlas13_entities.npz: separate prediction and observed layers')
 p.add_argument('--catalog',type=Path,required=True,help='chemical_catalog.csv in the same entity order')
 p.add_argument('--hallmark',type=Path,required=True,help='Exact paper Hallmark GMT')
 p.add_argument('--record-catalog',type=Path,help='Default: record_catalog.csv beside the chemical catalogue')
 p.add_argument('--coverage',type=Path,help='Default: coverage.csv beside the chemical catalogue')
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
 p.add_argument('--threads',type=int,default=8)
 a=p.parse_args()
 a.record_catalog=a.record_catalog or a.catalog.parent/'record_catalog.csv'
 a.coverage=a.coverage or a.catalog.parent/'coverage.csv'
 for path in [a.input,a.catalog,a.hallmark,a.record_catalog,a.coverage]:
  if not path.is_file():p.error(f'Input file not found: {path}')
 for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[key]=str(a.threads)
 from .analysis import run_analysis
 from .statistics import extend_statistics
 from .fingerprints import export_fingerprints
 import torch
 device=('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
 if device=='cuda' and not torch.cuda.is_available():p.error('CUDA was requested but is unavailable')
 if a.output.resolve() in {x.parent.resolve() for x in [a.input,a.catalog,a.hallmark,a.record_catalog,a.coverage]}:p.error('Choose an output directory separate from immutable inputs')
 a.output.mkdir(parents=True,exist_ok=True)
 run_analysis(a.input,a.catalog,a.hallmark,a.output,a.record_catalog,a.coverage,device,a.threads)
 extend_statistics(a.output);export_fingerprints(a.output)
 inputs={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in [a.input,a.catalog,a.hallmark,a.record_catalog,a.coverage]}
 (a.output/'run.json').write_text(json.dumps({'input_sha256':inputs,'device':device,'threads':a.threads,'parameters':{'genes':976,'neighbor_topk':30,'pca_components':50,'pca_solver':'randomized','clusters':8,'kmeans_n_init':10,'kmeans_batch_size':4096,'umap_neighbors':200,'umap_min_dist':0.50,'umap_epochs':200,'seed':20260926},'observed_used_for_embedding':False,'parameter_search':False},indent=2)+'\n')
if __name__=='__main__':main()
