"""Check scientific kernels against the frozen manuscript tables, without fitting UMAP."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import hypergeom,rankdata
import hashlib
from .analysis import norm,bh


def validate(reference,output):
 reference=Path(reference);output=Path(output);output.mkdir(parents=True,exist_ok=True)
 checks={}
 # Gene-profile correlation is calculated from976 exported gene means, not rounded figure labels.
 d=pd.read_csv(reference/'pi3k_mtor_disjoint_gene_profiles.csv')
 ci=pd.read_csv(reference/'pi3k_mtor_correlation_ci.csv')
 corr_deltas=[]
 for mode,q in d.groupby('mode',sort=False):
  p=norm(np.stack([q.pi3k.to_numpy(),q.mtor.to_numpy()]))
  r=float(p[0]@p[1]);expected=float(ci[ci['mode']==mode].r.iloc[0])
  corr_deltas.append(abs(r-expected));checks[mode+'_pi3k_mtor_r']=r
 # The exported gene means were serialized from float32 arrays.
 assert max(corr_deltas)<2e-7,corr_deltas
 checks['fingerprint_correlation_max_abs_difference']=max(corr_deltas)
 checks['fingerprint_tolerance']=2e-7
 # Every original ORA row uses the976-gene background and directional top50 set.
 h=pd.read_csv(reference/'hallmark_ora.csv')
 p=hypergeom.sf(h.overlap.to_numpy()-1,976,h.set_n.to_numpy(),50)
 p_delta=float(np.max(np.abs(p-h.p.to_numpy())));q_delta=float(np.max(np.abs(bh(p)-h.q.to_numpy())))
 assert p_delta<1e-12 and q_delta<1e-12,(p_delta,q_delta)
 checks.update(ora_rows=len(h),ora_p_max_abs_difference=p_delta,ora_q_max_abs_difference=q_delta)
 # Real per-drug predicted programme values and fixed membership reproduce all class means.
 z=np.load(reference/'response_metrics.npz');scores=z['hallmark_scores'];membership=z['membership'];names=list(z['hallmarks'].astype(str))
 cat=pd.read_csv(reference/'chemical_catalog.csv');class_names=['MEK','PI3K','mTOR','HDAC','Topoisomerase','Proteasome','CDK','EGFR','Tubulin','HSP inhibition','Glucocorticoid','Estrogen agonist','Estrogen antagonist','Retinoid','Dopamine antagonist','Serotonin antagonist']
 cells=['A375','A549','HT29','MCF7','PC3','HepG2','MCF10A','Jurkat','K562_L1000','HAP1','RPE1','neurons','Luhmes']
 assert np.array_equal(membership,np.column_stack([cat[k].to_numpy(bool) for k in class_names]))
 table=pd.read_csv(reference/'hallmark_class_scores.csv');errors=[]
 for row in table.itertuples():
  value=float(scores[cells.index(row.cell),membership[:,class_names.index(row.moa)],names.index(row.hallmark)].mean())
  errors.append(abs(value-row.mean))
 assert max(errors)<1e-6,max(errors)
 checks.update(class_programme_rows=len(errors),class_programme_max_abs_difference=max(errors),source='Original frozen predictions/programme cache and manuscript tables; no model or UMAP fit performed.')
 (output/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
 return checks


def validate_input(input_path,catalog_path,hallmark_path,reference,output):
 reference=Path(reference);output=Path(output);output.mkdir(parents=True,exist_ok=True)
 input_path=Path(input_path);catalog_path=Path(catalog_path);hallmark_path=Path(hallmark_path)
 with np.load(input_path) as z:X=z['prediction'];obs=z['observed'];M=z['observed_mask'];genes=z['genes'];keys=z['chemical_keys']
 cat=pd.read_csv(catalog_path);v=np.load(reference/'response_metrics.npz');classes=['MEK','PI3K','mTOR','HDAC','Topoisomerase','Proteasome','CDK','EGFR','Tubulin','HSP inhibition','Glucocorticoid','Estrogen agonist','Estrogen antagonist','Retinoid','Dopamine antagonist','Serotonin antagonist'];mem=np.column_stack([cat[k].to_numpy(bool) for k in classes])
 assert np.array_equal(keys,cat.chemical_key)
 rms=np.sqrt(np.mean(X**2,axis=-1));rmse=float(abs(rms-v['rms']).max());assert rmse<1e-6
 classmean=np.stack([X[:,mem[:,j]].mean(1) for j in range(len(classes))]);f=np.load(reference/'class_profiles.npz');classerr=float(abs(classmean-f['response']).max());assert classerr<1e-6
 sets={}
 for ln in (hallmark_path).read_text().splitlines():
  p=ln.split('\t');ix=np.flatnonzero(np.isin(genes,p[2:]));
  if len(ix)>=5:sets[p[0]]=ix
 assert np.array_equal(np.array(list(sets)),v['hallmarks'])
 maxerr=0.
 for lo in range(0,len(cat),128):
  hi=min(lo+128,len(cat));ranks=(rankdata(X[:,lo:hi],axis=-1)-.5)/976*2-1
  scores=np.stack([ranks[:,:,ix].mean(-1) for ix in sets.values()],axis=-1).astype('float32');maxerr=max(maxerr,float(abs(scores-v['hallmark_scores'][:,lo:hi]).max()))
 assert maxerr<1e-6
 report={'full_input_sha256':hashlib.file_digest((input_path).open('rb'),'sha256').hexdigest(),'n_entities':len(cat),'contexts':13,'genes':976,'predicted_rms_max_abs_difference':rmse,'class_profile_max_abs_difference':classerr,'all_predicted_hallmark_scores_max_abs_difference':maxerr,'n_hallmarks':len(sets),'embedding_fit_performed':False,'full_gene_profiles_verified':True}
 (output/'full_input_verification.json').write_text(json.dumps(report,indent=2)+'\n')
 return report

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--input',type=Path,help='Optionally verify all predicted gene profiles against the paper cache')
 p.add_argument('--catalog',type=Path);p.add_argument('--hallmark',type=Path)
 a=p.parse_args();result=validate(a.reference,a.output)
 if a.input is not None:
  if a.catalog is None or a.hallmark is None:p.error('--input also requires --catalog and --hallmark')
  result['full_input']=validate_input(a.input,a.catalog,a.hallmark,a.reference,a.output)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
