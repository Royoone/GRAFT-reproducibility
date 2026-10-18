"""Supplement four target-QC transcripts using NCBI/HGNC-verified old symbols.

This leaves the released 976-gene model input unchanged. The original perturbation
names remain unchanged; only transcript-feature correspondence is resolved.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
SOURCE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--work',type=Path,required=True)
P=parser.parse_args().work.resolve();O=P/'processed';S=P/'sparse_alias_rows'
z=np.load(O/'graft_input_976.npz');q=np.load(O/'cell_qc_summary.npz')
am=pd.read_csv(SOURCE/'verified_target_aliases.csv');rows=pd.read_csv(S/'retained_rows.tsv',sep='\t')
assert set(am.expression_symbol)==set(rows.symbol)
groups=q['group_index'];cell_ok=q['cell_ok'];total=q['total_umi'];C=len(z['contexts']);K=len(z['perturbations']);N=K+2
counts=np.bincount(groups[cell_ok],minlength=C*N).reshape(C,N)
out={prefix+mode:np.full((C,K,len(am)),np.nan,dtype=np.float32) for prefix in ['effects__','stderr__'] for mode in ['lognorm_nt']}
for j,a in enumerate(am.itertuples(index=False)):
 r=rows.set_index('symbol').loc[a.expression_symbol]
 with (S/r.file).open('rb') as f:
  n=int(np.fromfile(f,dtype='<u8',count=1)[0]);ix=np.fromfile(f,dtype='<u4',count=n);v=np.fromfile(f,dtype='<u4',count=n)
 ok=cell_ok[ix];ix=ix[ok];v=v[ok].astype(np.float64);gr=groups[ix]
 x=np.log1p(v/total[ix]*1e4)
 sm=np.bincount(gr,weights=x,minlength=C*N).astype(float).reshape(C,N);ss=np.bincount(gr,weights=x*x,minlength=C*N).astype(float).reshape(C,N)
 mean=np.divide(sm,counts,out=np.full_like(sm,np.nan),where=counts>0);var=np.maximum(np.divide(ss-sm*mean,counts-1,out=np.full_like(sm,np.nan),where=counts>1),0)
 ctrlcols=[K]
 nc=counts[:,ctrlcols].sum(1);cs=sm[:,ctrlcols].sum(1);cq=ss[:,ctrlcols].sum(1);cm=cs/nc;cv=np.maximum((cq-cs*cs/nc)/(nc-1),0)
 out['effects__lognorm_nt'][:,:,j]=mean[:,:K]-cm[:,None]
 out['stderr__lognorm_nt'][:,:,j]=np.sqrt(var[:,:K]/np.maximum(counts[:,:K],1)+cv[:,None]/nc[:,None])

np.savez_compressed(O/'alias_target_effects.npz',contexts=z['contexts'],donors=z['donors'],perturbations=z['perturbations'],genes=am.perturbation_symbol.to_numpy(dtype=str),expression_symbols=am.expression_symbol.to_numpy(dtype=str),n_cells=z['n_cells'],valid=z['valid'],**out)
qc=pd.read_csv(O/'clone_target_qc.csv');qc['target_expression_symbol']=qc.target
for j,a in enumerate(am.itertuples(index=False)):
 k=list(z['perturbations']).index(a.perturbation_symbol)
 for c,clone in enumerate(z['contexts']):
  mask=qc.clone.eq(clone)&qc.target.eq(a.perturbation_symbol)
  qc.loc[mask,'target_expression_symbol']=a.expression_symbol
  for mode in ['lognorm_nt']:qc.loc[mask,'target_downregulation__'+mode]=float(out['effects__'+mode][c,k,j])
qc.to_csv(O/'clone_target_qc_with_verified_aliases.csv',index=False)
print('Verified alias target expression restored:',len(am),'genes;',len(qc),'clone-target conditions. Input976 unchanged.')
