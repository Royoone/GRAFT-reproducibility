"""Exercise the fixed analysis on a small, real Atlas13 subset (not manuscript estimates)."""
import argparse,json,os
from pathlib import Path
import numpy as np
import pandas as pd


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--input',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--hallmark',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[key]='2'
 from .analysis import MOA,run_analysis
 from .statistics import extend_statistics
 from .fingerprints import export_fingerprints
 z=np.load(a.input);cat=pd.read_csv(a.catalog).fillna('');M=z['observed_mask'].astype(bool);common=M[:5].all(0)
 # Inclusion uses fixed catalogue order/labels and measurement availability, never predicted outcomes.
 selected=set(np.flatnonzero(common)[:128].tolist())
 for cls in MOA:
  mem=cat[cls].to_numpy(bool);selected.update(np.flatnonzero(mem)[:4].tolist());selected.update(np.flatnonzero(mem&common)[:12].tolist())
 for left,right in [('PI3K','mTOR'),('mTOR','PI3K')]:selected.update(np.flatnonzero(cat[left].to_numpy(bool)&~cat[right].to_numpy(bool)&common)[:4].tolist())
 selected=np.array(sorted(selected));sub=cat.iloc[selected].reset_index(drop=True)
 if not len(selected)>30:raise ValueError('Too few real chemicals for the fixed top30-neighbour analysis')
 data=a.output/'input';data.mkdir(exist_ok=True)
 pred=z['prediction'] if 'prediction' in z.files else z['response']
 np.savez_compressed(data/'atlas13_entities.npz',prediction=pred[:,selected],observed=z['observed'][:,selected],observed_mask=M[:,selected],chemical_keys=z['chemical_keys'][selected],genes=z['genes'],cells=z['cells'])
 sub.to_csv(data/'chemical_catalog.csv',index=False)
 # Census for a smoke subset containing one representative real ID per chemical entity.
 sub[['drug_id','chemical_key']].to_csv(data/'record_catalog.csv',index=False)
 pd.DataFrame([dict(cell=str(cell),records=len(sub),observed_records=int(mask.sum()),completed_records=int((~mask).sum())) for cell,mask in zip(z['cells'],M[:,selected])]).to_csv(data/'coverage.csv',index=False)
 out=a.output/'analysis';run_analysis(data/'atlas13_entities.npz',data/'chemical_catalog.csv',a.hallmark,out,data/'record_catalog.csv',data/'coverage.csv',device='cpu',threads=2,expected_entities=None)
 extend_statistics(out);export_fingerprints(out)
 report={'scope':'Operational smoke test on real, unmodified drug-response profiles; not manuscript estimates.','n_real_chemical_entities':len(sub),'contexts':13,'genes':976,'pca_components':50,'clusters':8,'umap_neighbors':200,'umap_min_dist':0.50,'umap_epochs':200,'seed':20260926,'selection':'fixed catalogue order, class labels and observation mask; no response-based selection','analysis_finished':True}
 (a.output/'smoke_test.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
