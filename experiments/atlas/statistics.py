"""Original manuscript post-analysis statistics; unchanged tests and grouping."""
from pathlib import Path
import numpy as np,pandas as pd,json
from scipy.stats import rankdata

def extend_statistics(output):
    """Community annotation and matched-drug summaries for the atlas narrative."""
    O=Path(output)
    z=np.load(O/'atlas13_entities.npz');X=z['response'];Xobs=z['observed'];M=z['observed_mask'];genes=z['genes'];cells=z['cells'];A=pd.read_csv(O/'chemical_catalog.csv').fillna('')
    v=np.load(O/'response_metrics.npz');scores=v['hallmark_scores'];h=v['hallmarks'];mem=v['membership'];common=v['common5'];rms=v['rms'];obsrms=v['observed_rms'];obsscores=v['observed_hallmark_scores']
    u=pd.read_csv(O/'umap_coordinates.csv.gz');cl=u.cluster.to_numpy().reshape(13,-1);moa=list(json.loads((O/'analysis_summary.json').read_text())['moa']);N=X.shape[1]
    rng=np.random.default_rng(20260926)
    def save(d,name):d.to_csv(O/(name+'.csv'),index=False)
    def corr(a,b):
     a=a-a.mean(-1,keepdims=True);b=b-b.mean(-1,keepdims=True);return np.sum(a*b,axis=-1)/np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-12)
    # Cluster mean program scores, one value per chemical averaged over its cluster members.
    rows=[];centroids=[]
    for k in range(1,9):
     mask=cl==k;centroids.append(X[mask].mean(0));count=mask.sum(0);eligible=count>0
     per=(scores*mask[:,:,None]).sum(0)[eligible]/count[eligible,None]
     for hi,name in enumerate(h):rows.append(dict(cluster=k,hallmark=name,mean=float(per[:,hi].mean()),n_entities=int(eligible.sum()),n_points=int(mask.sum())))
    save(pd.DataFrame(rows),'community_hallmarks');centroids=np.stack(centroids)
    ig=sorted(set(i for row in centroids for i in np.r_[np.argsort(row)[:3],np.argsort(row)[-3:]]))
    save(pd.DataFrame([dict(cluster=k+1,gene=str(genes[i]),response=float(centroids[k,i])) for k in range(8) for i in ig]),'community_gene_fingerprints')
    # Remove shared annotated members before comparing each pair of class fingerprints.
    rows=[];cirows=[]
    for j in range(len(moa)):
     for k in range(len(moa)):
      ma=mem[:,j]&~mem[:,k] if j!=k else mem[:,j];mb=mem[:,k]&~mem[:,j] if j!=k else mem[:,k]
      for mode,dat,mask in [('Atlas13',X.mean(0),np.ones(N,bool)),('Measured5',Xobs[:5].mean(0),common)]:
       aa=dat[ma&mask];bb=dat[mb&mask]
       value=float(corr(aa.mean(0),bb.mean(0))) if len(aa) and len(bb) else np.nan
       rows.append(dict(moa1=moa[j],moa2=moa[k],mode=mode,n1=len(aa),n2=len(bb),r=value))
       if (moa[j],moa[k])==('PI3K','mTOR'):
        bs=[]
        for it in range(2000):bs.append(corr(aa[rng.integers(len(aa),size=len(aa))].mean(0),bb[rng.integers(len(bb),size=len(bb))].mean(0)))
        lo,hi=np.quantile(bs,[.025,.975]);cirows.append(dict(moa1=moa[j],moa2=moa[k],mode=mode,n1=len(aa),n2=len(bb),r=value,lo=lo,hi=hi))
    save(pd.DataFrame(rows),'disjoint_class_correlations');save(pd.DataFrame(cirows),'pi3k_mtor_correlation_ci')
    # Matched five-background class amplitudes: each chemical supplies all five observations.
    rows=[];drugrows=[]
    for j,cls in enumerate(moa):
     ix=np.flatnonzero(mem[:,j]&common)
     if not len(ix):continue
     for c,cell in enumerate(cells[:5]):
      xx=obsrms[c,ix];boot=xx[rng.integers(len(ix),size=(2000,len(ix)))].mean(1);lo,hi=np.quantile(boot,[.025,.975])
      rows.append(dict(moa=cls,cell=cell,n=len(ix),mean=float(xx.mean()),lo=lo,hi=hi))
      for i,value in zip(ix,xx):drugrows.append(dict(moa=cls,cell=cell,chemical_key=A.chemical_key.iloc[i],name=A.name.iloc[i],rms=float(value)))
    save(pd.DataFrame(rows),'matched5_class_rms_ci');save(pd.DataFrame(drugrows),'matched5_class_rms_drugs')
    # MoA coverage in each context: all alias-unified labels, no exclusive partition.
    rows=[]
    for c,cell in enumerate(cells):
     for j,cls in enumerate(moa):rows.append(dict(cell=cell,moa=cls,total=int(mem[:,j].sum()),observed=int((M[c]&mem[:,j]).sum())))
    save(pd.DataFrame(rows),'moa_coverage')
    # Selected class program patterns in matched-five data and the full atlas.
    rows=[]
    for j,cls in enumerate(moa):
     for c,cell in enumerate(cells):
      ix=mem[:,j]&common
      for hi,name in enumerate(h):
       rows.append(dict(moa=cls,cell=cell,hallmark=name,matched5_n=int(ix.sum()),matched5_mean=float(obsscores[c,ix,hi].mean()) if c<5 else np.nan,class_mean=float(scores[c,mem[:,j],hi].mean())))
    save(pd.DataFrame(rows),'matched_program_scores')
    print('EXTENDED COMPLETE',flush=True)
