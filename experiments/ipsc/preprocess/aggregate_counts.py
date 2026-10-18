"""Aggregate retained sparse counts using full-transcriptome cell library sizes.

Zero-expression cells remain in every group denominator. Only NonTarget cells
define the same-line control reference; unassigned cells are retained only in QC.
Cell-level sampling standard errors describe precision, not biological replicates.
"""
from pathlib import Path
import json,re,sys,argparse
import numpy as np
import pandas as pd

SOURCE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--work',type=Path,required=True,help='Contains raw/ and sparse_rows/; outputs processed/')
args=parser.parse_args()
HERE=args.work.resolve()
BASE=HERE.parent
SPARSE=HERE/'sparse_rows'
OUT=HERE/'processed'
OUT.mkdir(parents=True,exist_ok=True)

def main():
 summary=json.loads((SPARSE/'scan_summary.json').read_text())
 assert summary['complete']
 cols=(SPARSE/'counts_header.csv').read_text().strip().split(',')[1:]
 assert len(cols)==summary['n_cells'] and len(set(cols))==len(cols)
 total=np.fromfile(SPARSE/'total_umi.uint64',dtype='<u8')
 mt=np.fromfile(SPARSE/'mitochondrial_umi.uint64',dtype='<u8')
 detected=np.fromfile(SPARSE/'n_detected.uint32',dtype='<u4')
 m=pd.read_csv(HERE/'raw/TargetedScreen_Cell-Metadata.tsv.gz',sep='\t')
 m['count_id']=m.Cell_ID.str.replace(r'^PC-(P\d+)-D3_',r'\1_',regex=True)
 assert not m.count_id.duplicated().any()
 m['target']=m.Guide_Call.str.split('_').str[0]
 contexts=np.array(sorted(m.Cell_Line.unique()))
 donors=np.array([x.split('_')[0] for x in contexts])
 perts=np.array(sorted(set(m.target)-{'NonTarget','unassigned'}))
 conds=list(perts)+['NonTarget','unassigned'];K=len(perts);C=len(contexts);NCOND=K+2
 ci={c:i for i,c in enumerate(contexts)};ti={t:i for i,t in enumerate(conds)}
 m=m.set_index('count_id').reindex(cols)
 cell_ok=m.Cell_Line.notna().to_numpy()&(total>0)&(detected>=100)
 groups=np.full(len(cols),-1,dtype=np.int32)
 eligible=m.loc[cell_ok]
 groups[cell_ok]=eligible.Cell_Line.map(ci).to_numpy()*NCOND+eligible.target.map(ti).to_numpy()
 counts=np.bincount(groups[cell_ok],minlength=C*NCOND).reshape(C,NCOND)
 n_target=counts[:,:K];n_nt=counts[:,K];n_un=counts[:,K+1]
 n_guide=np.zeros((C,K),dtype=np.int16)
 for (c,t),s in eligible[eligible.target.isin(perts)].groupby(['Cell_Line','target'],observed=True):n_guide[ci[c],ti[t]]=s.Guide_Call.nunique()
 idx=pd.read_csv(SPARSE/'retained_rows.tsv',sep='\t')
 genes=np.array(sorted(idx.symbol.unique()));G=len(genes)
 geneindex={g:i for i,g in enumerate(genes)}
 output_genes=np.load(SOURCE/'output_genes_976.npy').astype(str)
 measured=np.isin(output_genes,genes)
 arrays={key:np.full((C,K,G),np.nan,dtype=np.float32) for key in ['effects__lognorm_nt','stderr__lognorm_nt']}
 control_means={prefix+key:np.zeros((C,G),dtype=np.float32) for prefix in ['control_mean__','control_var__'] for key in ['lognorm_nt']}
 gene_qc=[]
 for gi,g in enumerate(genes):
  matches=idx[idx.symbol.eq(g)];ixs=[];vs=[]
  for r in matches.itertuples(index=False):
   with (SPARSE/r.file).open('rb') as f:
    nnz=int(np.fromfile(f,dtype='<u8',count=1)[0]);ix=np.fromfile(f,dtype='<u4',count=nnz);v=np.fromfile(f,dtype='<u4',count=nnz)
   assert len(v)==nnz
   ixs.append(ix);vs.append(v)
  if len(ixs)>1:
   dense=np.zeros(len(cols),dtype=np.uint64)
   for ix,v in zip(ixs,vs):np.add.at(dense,ix,v)
   ix=np.flatnonzero(dense);v=dense[ix]
  else:ix,v=ixs[0],vs[0]
  use=cell_ok[ix];ix=ix[use];v=v[use].astype(np.float64);gr=groups[ix]
  gene_qc.append({'gene':g,'feature_rows':len(matches),'nonzero_cells':len(ix),'total_counts':int(v.sum()),'original_features':'|'.join(matches.feature)})
  values=np.log1p(v/total[ix]*10000.0)
  sm=np.bincount(gr,weights=values,minlength=C*NCOND).reshape(C,NCOND).astype(np.float64)
  ss=np.bincount(gr,weights=values*values,minlength=C*NCOND).reshape(C,NCOND).astype(np.float64)
  mean=np.divide(sm,counts,out=np.full_like(sm,np.nan),where=counts>0)
  var=np.divide(ss-sm*mean,counts-1,out=np.full_like(sm,np.nan),where=counts>1)
  var=np.maximum(var,0)
  cm,cv=mean[:,K],var[:,K]
  arrays['effects__lognorm_nt'][:,:,gi]=mean[:,:K]-cm[:,None]
  arrays['stderr__lognorm_nt'][:,:,gi]=np.sqrt(var[:,:K]/np.maximum(n_target,1)+cv[:,None]/n_nt[:,None])
  control_means['control_mean__lognorm_nt'][:,gi]=cm
  control_means['control_var__lognorm_nt'][:,gi]=cv
  if (gi+1)%100==0:print(f'AGGREGATED {gi+1}/{G}',flush=True)
 valid_nt=(n_target>=10)&(n_nt[:,None]>=50)&(n_guide>=2)
 # QC arrays retain measured estimates alongside validity masks.
 # Model-input arrays replace invalid condition rows with NaN.
 shared={'contexts':contexts,'donors':donors,'perturbations':perts,'n_cells':n_target,'n_guides':n_guide,'n_controls':n_nt,'valid':valid_nt}
 np.savez_compressed(OUT/'clone_target_effects_all_selected_genes.npz',genes=genes,**shared,**arrays,**control_means)
 output={}
 for key,a in arrays.items():
  b=np.full((C,K,len(output_genes)),np.nan,dtype=np.float32)
  for j,g in enumerate(output_genes):
   if measured[j]:b[:,:,j]=a[:,:,geneindex[g]]
  b[~valid_nt]=np.nan
  output[key]=b
 for key,a in control_means.items():
  b=np.full((C,len(output_genes)),np.nan,dtype=np.float32)
  for j,g in enumerate(output_genes):
   if measured[j]:b[:,j]=a[:,geneindex[g]]
  output[key]=b
 np.savez_compressed(OUT/'graft_input_976.npz',genes=output_genes,measured_gene_mask=measured,**shared,**output)
 pd.DataFrame(gene_qc).to_csv(OUT/'retained_gene_qc.csv',index=False)
 rows=[]
 for c,line in enumerate(contexts):
  ix=cell_ok&(groups//NCOND==c)
  for k,t in enumerate(perts):
   r={'clone':line,'donor':donors[c],'target':t,'n_cells':int(n_target[c,k]),'n_guides':int(n_guide[c,k]),'n_controls_nt':int(n_nt[c]),'valid':bool(valid_nt[c,k])}
   if t in geneindex:
    j=geneindex[t]
    for key in ['lognorm_nt']:r['target_downregulation__'+key]=float(arrays['effects__'+key][c,k,j])
   rows.append(r)
 pd.DataFrame(rows).to_csv(OUT/'clone_target_qc.csv',index=False)
 clone_qc=[]
 for c,line in enumerate(contexts):
  ix=cell_ok&(groups//NCOND==c)
  clone_qc.append({'clone':line,'donor':donors[c],'n_cells':int(ix.sum()),'n_nt':int(n_nt[c]),'n_unassigned':int(n_un[c]),'median_umi':float(np.median(total[ix])),'median_detected':float(np.median(detected[ix])),'median_mito_fraction':float(np.median(mt[ix]/total[ix])),'n_valid_targets':int(valid_nt[c].sum())})
 pd.DataFrame(clone_qc).to_csv(OUT/'clone_qc.csv',index=False)
 np.savez_compressed(OUT/'cell_qc_summary.npz',total_umi=total,n_detected=detected,mitochondrial_umi=mt,cell_ok=cell_ok,group_index=groups)
 report={'source':'Feng targeted screen Figshare 27989294 file51072902','counts_shape':[summary['n_gene_rows'],summary['n_cells']],'metadata_cells':int(m.Cell_Line.notna().sum()),'retained_cells':int(cell_ok.sum()),'quality_filter':'existing metadata + total_UMI>0 + >=100 detected genes','n_clones':C,'n_targets':K,'n_selected_genes':G,'n_measured_976':int(measured.sum()),'missing_976':output_genes[~measured].tolist(),'normalization':{'lognorm':'mean over cells(log1p(raw count / full-gene cell total UMI * 10000))'},'controls':{'nt':'NonTarget guide calls only, clone-specific reference'},'delta':'perturbed mean minus matched-clone control mean','valid':'n_target_cells>=10 and n_guides>=2 and n_controls>=50','missing_handling':'NaN and validity masks; no zero filling of unmeasured genes or conditions','stderr':'cell-level sampling SEM of perturbation minus control; not biological replicate confidence','batch':'clone maintained; this output does not regress inlet or pool; inlet and pool remain part of the recorded experimental metadata','notes':['Zero-expression cells contribute to group sample counts.','Duplicate feature rows with identical symbols are summed in count space before log transform.','The unmatched extra count column is excluded by barcode join.','Knockdown magnitude is provided separately; it is not silently thresholded.','No standardization against target drug outcomes or arbitrary effect amplitude scaling is applied.']}
 (OUT/'preprocessing_metadata.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
