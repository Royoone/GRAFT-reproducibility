"""Fixed manuscript Atlas13 analysis, migrated from the original author script.

No parameter selection is performed: 976 genes, PCA50, k-means8, UMAP200/.50.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import rankdata,hypergeom,mannwhitneyu,spearmanr
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import torch
CELLS=['A375','A549','HT29','MCF7','PC3','HepG2','MCF10A','Jurkat','K562_L1000','HAP1','RPE1','neurons','Luhmes']
MOA={'MEK':'mek inhibitor','PI3K':'pi3k inhibitor','mTOR':'mtor inhibitor','HDAC':'hdac inhibitor','Topoisomerase':'topoisomerase inhibitor','Proteasome':'proteasome inhibitor','CDK':'cdk inhibitor','EGFR':'egfr inhibitor','Tubulin':'tubulin inhibitor','HSP inhibition':'hsp inhibitor','Glucocorticoid':'glucocorticoid receptor agonist','Estrogen agonist':'estrogen receptor agonist','Estrogen antagonist':'estrogen receptor antagonist','Retinoid':'retinoid receptor agonist','Dopamine antagonist':'dopamine receptor antagonist','Serotonin antagonist':'serotonin receptor antagonist'}
SEED=20260926;rng=np.random.default_rng(SEED)
def bh(p):
 p=np.asarray(p);order=np.argsort(p);q=np.empty(len(p));q[order]=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];return np.minimum(q,1)
def boot(x,n=2000):
 x=np.asarray(x);x=x[np.isfinite(x)];b=np.mean(rng.choice(x,(n,len(x)),replace=True),axis=1);return float(x.mean()),*np.quantile(b,[.025,.975])
def norm(x):
 x=x-x.mean(-1,keepdims=True);return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)

def run_analysis(input_path,catalog_path,hallmark_path,output,record_catalog_path,coverage_path,device='cpu',threads=8,expected_entities=7755):
    global rng
    rng=np.random.default_rng(SEED)
    # Read the verified frozen-model layer; keep observed responses separate.
    O=Path(output);O.mkdir(parents=True,exist_ok=True)
    def save(d,n):d.to_csv(O/(n+'.csv'),index=False)
    z=np.load(input_path)
    required={'observed','observed_mask','genes','cells','chemical_keys'}
    missing=required-set(z.files)
    if missing:raise ValueError(f'Missing Atlas13 arrays: {sorted(missing)}')
    if 'prediction' not in z.files and 'response' not in z.files:raise ValueError('The archive must contain the frozen prediction layer')
    X=z['prediction'].copy() if 'prediction' in z.files else z['response'].copy()
    Xtruth=z['observed'];M=z['observed_mask'].astype(bool);genes=z['genes'];CELLS=z['cells'].astype(str).tolist()
    a=pd.read_csv(catalog_path).fillna('')
    catalog=pd.read_csv(record_catalog_path).fillna('');C=pd.read_csv(coverage_path)
    membership=np.column_stack([a[k].to_numpy(dtype=bool) for k in MOA]);N=len(a)
    if X.shape!=(13,len(a),976) or not np.isfinite(X).all():raise ValueError('Expected finite predictions with shape13 × catalogue entities ×976')
    if expected_entities is not None and len(a)!=expected_entities:raise ValueError(f'Expected {expected_entities} chemical entities for the manuscript analysis')
    if Xtruth.shape!=X.shape or M.shape!=X.shape[:2]:raise ValueError('Observed layer/mask shape differs from prediction')
    if not np.isfinite(Xtruth[M]).all():raise ValueError('Available observed profiles must be finite')
    if len(set(genes.astype(str)))!=976:raise ValueError('Expected976 unique genes')
    if CELLS!=['A375','A549','HT29','MCF7','PC3','HepG2','MCF10A','Jurkat','K562_L1000','HAP1','RPE1','neurons','Luhmes']:raise ValueError('Atlas13 context order differs from the manuscript')
    assert np.array_equal(a.chemical_key.to_numpy(),z['chemical_keys'])
    for table,name in [(a,'chemical_catalog'),(catalog,'record_catalog'),(C,'coverage')]:save(table,name)
    observed=np.where(M[:,:,None],Xtruth,np.nan).astype('float32')
    # All public response coordinates and class summaries below use X (prediction).
    np.savez_compressed(O/'atlas13_entities.npz',response=X,prediction=X,observed=observed,observed_mask=M,chemical_keys=z['chemical_keys'],genes=genes,cells=z['cells'])
    Y=norm(X);Ytruth=norm(Xtruth);rms=np.sqrt(np.mean(X**2,axis=-1));observed_rms=np.where(M,np.sqrt(np.mean(Xtruth**2,axis=-1)),np.nan);common=M[:5].all(0)
    print('LOADED PREDICTION',X.shape,'OBSERVED',M.sum(),flush=True)
    # Curated Hallmark sets: all eligible sets, without inspecting outcomes.
    sets={}
    for ln in Path(hallmark_path).read_text().splitlines():
     p=ln.split('\t');ix=np.flatnonzero(np.isin(genes,p[2:]));
     if len(ix)>=5:sets[p[0]]=ix
    ranks=(rankdata(X,axis=-1)-.5)/976*2-1
    scores=np.stack([ranks[:,:,ix].mean(-1) for ix in sets.values()],axis=-1).astype('float32');del ranks
    truth_ranks=(rankdata(Xtruth,axis=-1)-.5)/976*2-1
    truth_scores=np.stack([truth_ranks[:,:,ix].mean(-1) for ix in sets.values()],axis=-1).astype('float32');del truth_ranks
    truth_scores[~M]=np.nan
    np.savez_compressed(O/'response_metrics.npz',rms=rms,hallmark_scores=scores,hallmarks=np.array(list(sets)),membership=membership,common5=common,observed_rms=observed_rms,observed_hallmark_scores=truth_scores)
    # Per-drug measured five-context pattern and amplitude variation; all five matched.
    ids=np.flatnonzero(common);sim=np.einsum('cng,dng->ncd',Ytruth[:5,ids],Ytruth[:5,ids]);div=1-sim[:,np.triu_indices(5,1)[0],np.triu_indices(5,1)[1]].mean(1)
    cv=observed_rms[:5,ids].std(0)/observed_rms[:5,ids].mean(0);ratio=observed_rms[:5,ids].max(0)/observed_rms[:5,ids].mean(0)
    v=a.iloc[ids][['chemical_key','drug_id','name','moa']].copy();v['pattern_diversity']=div;v['rms_cv']=cv;v['max_to_mean_rms']=ratio;v['mean_rms']=observed_rms[:5,ids].mean(0);save(v,'measured_five_drug_variation')
    rows=[]
    for j,k in enumerate(MOA):
     ix=membership[ids,j]
     if ix.sum()<4:continue
     for metric,values in [('Pattern diversity',div),('RMS CV',cv),('Mean RMS',observed_rms[:5,ids].mean(0))]:
      mean,lo,hi=boot(values[ix]);p=mannwhitneyu(values[ix],values[~ix],alternative='two-sided').pvalue
      rows.append(dict(moa=k,metric=metric,n=int(ix.sum()),mean=mean,lo=lo,hi=hi,other_mean=float(values[~ix].mean()),p=p))
    q=pd.DataFrame(rows);q['q']=bh(q.p);save(q,'moa_measured_variation')
    # All eligible raw MoA terms, exploratory class screening, same family of tests.
    terms=sorted(set(t for v in a.moa for t in v.split('|') if t));rows=[]
    for term in terms:
     ix=a.moa.iloc[ids].map(lambda v:term in v.split('|')).to_numpy()
     if ix.sum()<10:continue
     mean,lo,hi=boot(div[ix],1000);p=mannwhitneyu(div[ix],div[~ix],alternative='two-sided').pvalue
     rows.append(dict(term=term,n=int(ix.sum()),diversity=mean,lo=lo,hi=hi,rms_cv=float(cv[ix].mean()),other_diversity=float(div[~ix].mean()),p=p))
    t=pd.DataFrame(rows);t['q']=bh(t.p);save(t,'all_eligible_moa_variation')
    # Pairwise context concordance uses matched chemicals in each pair.
    rows=[]
    for i,c1 in enumerate(CELLS):
     for j,c2 in enumerate(CELLS):
      ss=(Y[i]*Y[j]).sum(-1);m=M[i]&M[j]
      rows.append(dict(cell1=c1,cell2=c2,atlas_mean=float(ss.mean()),measured_n=int(m.sum()),measured_mean=float((Ytruth[i,m]*Ytruth[j,m]).sum(-1).mean()) if m.any() else np.nan))
    save(pd.DataFrame(rows),'context_concordance')
    # Class centroid and directional Hallmark overrepresentation.
    classmean=np.stack([X[:,membership[:,j]].mean(1) for j in range(len(MOA))]);classnorm=norm(classmean)
    np.savez_compressed(O/'class_profiles.npz',response=classmean,cells=np.array(CELLS),moa=np.array(list(MOA)),genes=genes)
    rows=[];score_rows=[]
    for j,k in enumerate(MOA):
     mem=membership[:,j];n=int(mem.sum())
     for ci,c in enumerate(CELLS+['Pooled']):
      mean=classmean[j,ci] if ci<13 else classmean[j].mean(0)
      for direction,ix in [('Down',np.argsort(mean)[:50]),('Up',np.argsort(mean)[-50:])]:
       for h,g in sets.items():
        overlap=np.intersect1d(ix,g);p=hypergeom.sf(len(overlap)-1,976,len(g),50)
        rows.append(dict(moa=k,cell=c,direction=direction,hallmark=h,overlap=len(overlap),set_n=len(g),fold=len(overlap)/(50*len(g)/976),p=p,genes='|'.join(genes[overlap])))
     for ci,c in enumerate(CELLS):
      for hi,h in enumerate(sets):
       z=scores[ci,mem,hi];mu,lo,hh=boot(z,500)
       ms=mem&M[ci];meas=float(truth_scores[ci,ms,hi].mean()) if ms.any() else np.nan
       score_rows.append(dict(moa=k,cell=c,hallmark=h,n=n,mean=mu,lo=lo,hi=hh,measured_n=int(ms.sum()),measured_mean=meas))
    h=pd.DataFrame(rows);h['q']=bh(h.p);save(h,'hallmark_ora');save(pd.DataFrame(score_rows),'hallmark_class_scores')
    # Per-class measured response strength by background, paired drugs for LINCS five.
    rows=[]
    for j,k in enumerate(MOA):
     for ci,c in enumerate(CELLS):
      ix=membership[:,j];mu,lo,hi=boot(rms[ci,ix]);ms=ix&M[ci]
      rows.append(dict(moa=k,cell=c,n=int(ix.sum()),rms=mu,lo=lo,hi=hi,measured_n=int(ms.sum()),measured_rms=float(observed_rms[ci,ms].mean()) if ms.any() else np.nan,matched5_n=int((ix&common).sum()),matched5_rms=float(observed_rms[ci,ix&common].mean()) if ci<5 and (ix&common).any() else np.nan))
    save(pd.DataFrame(rows),'class_rms')
    # kNN on original 976-gene pattern, within each context: distinct chemical entities.
    print('KNN START',flush=True);torch.set_num_threads(threads);torch.backends.cuda.matmul.allow_tf32=False;neighbors=[]
    for ci,c in enumerate(CELLS):
     T=torch.tensor(Y[ci],device=device);parts=[]
     for st in range(0,N,512):
      s=T[st:st+512]@T.T;rr=torch.arange(len(s),device=device);s[rr,torch.arange(st,st+len(s),device=device)]=-10
      parts.append(torch.topk(s,30,dim=1).indices.cpu().numpy())
     neighbors.append(np.concatenate(parts));print('KNN',c,flush=True)
    knn=np.stack(neighbors);np.savez_compressed(O/'knn976.npz',indices=knn)
    rows=[]
    for j,k in enumerate(MOA):
     ix=membership[:,j];n=int(ix.sum());purity=membership[knn,j].mean(-1);perdrug=purity[:,ix].mean(0);expected=(n-1)/(N-1);mu,lo,hi=boot(perdrug)
     rows.append(dict(moa=k,n=n,neighbor_fraction=mu,lo=lo,hi=hi,expected=expected,fold=mu/expected,fold_lo=lo/expected,fold_hi=hi/expected))
    save(pd.DataFrame(rows),'moa_knn_enrichment')
    # Chemical-balanced response map; MoA labels are used only after embedding.
    print('PCA START',flush=True);flat=Y.reshape(-1,976);pca=PCA(n_components=50,random_state=SEED,svd_solver='randomized');pc=pca.fit_transform(flat).astype('float32');np.save(O/'response_pca50.npy',pc)
    print('PCA variance',pca.explained_variance_ratio_.sum(),flush=True)
    km=MiniBatchKMeans(n_clusters=8,random_state=SEED,batch_size=4096,n_init=10);clusters=km.fit_predict(pc)
    # Summarize response communities with chemical-level prevalence, not replicate-level p-values.
    rows=[]
    for cl in range(8):
     frac=(clusters.reshape(13,N)==cl).mean(0)
     for j,k in enumerate(MOA):
      mu,lo,hi=boot(frac[membership[:,j]])
      rows.append(dict(cluster=cl+1,moa=k,n=int(membership[:,j].sum()),fraction=mu,lo=lo,hi=hi,background=float(frac.mean()),fold=mu/max(frac.mean(),1e-12)))
    save(pd.DataFrame(rows),'cluster_moa_composition')
    import umap
    print('UMAP START',flush=True)
    embedding_input_sha256=hashlib.sha256(pc.tobytes()).hexdigest()
    previous=json.loads((O/'analysis_summary.json').read_text()) if (O/'analysis_summary.json').exists() else {}
    if (O/'umap_coordinates.csv.gz').exists() and previous.get('embedding_input_sha256')==embedding_input_sha256 and previous.get('umap_n_neighbors')==200 and previous.get('umap_min_dist')==.50:
     u=pd.read_csv(O/'umap_coordinates.csv.gz')[['UMAP1','UMAP2']].to_numpy()
     assert len(u)==len(pc)
    else:
     u=umap.UMAP(n_neighbors=200,min_dist=.50,n_components=2,metric='euclidean',random_state=SEED,n_jobs=1,low_memory=True,n_epochs=200,verbose=True).fit_transform(pc)
    coords=pd.DataFrame(dict(chemical_index=np.tile(np.arange(N),13),cell=np.repeat(CELLS,N),UMAP1=u[:,0],UMAP2=u[:,1],measured=M.ravel(),rms=rms.ravel(),vector_source='prediction',observed_rms=observed_rms.ravel(),cluster=clusters+1,display_moa=np.tile(a.display_moa,13)))
    coords.to_csv(O/'umap_coordinates.csv.gz',index=False)
    for hi,h in enumerate(sets):coords[h]=scores[:,:,hi].ravel()
    coords.to_pickle(O/'umap_with_programs.pkl')
    meta=dict(embedding_input_sha256=embedding_input_sha256,seed=SEED,cells=CELLS,n_records=len(catalog),n_entities=N,n_points=13*N,n_genes=976,observed_records=int(C.observed_records.sum()),completed_records=int(C.completed_records.sum()),observed_entities=int(M.sum()),common_measured5=int(common.sum()),moa=MOA,moa_counts={k:int(membership[:,j].sum()) for j,k in enumerate(MOA)},pca_variance=float(pca.explained_variance_ratio_.sum()),hallmark_tests=len(h) if False else 16*14*2*len(sets),n_hallmarks=len(sets),umap='row-centered L2 -> PCA50 -> UMAP k200 min_dist0.50',umap_n_neighbors=200,umap_min_dist=.50,cluster='MiniBatchKMeans k8 on PCA50; descriptive response communities',source='prediction at every position; observed layer is stored separately and used only in explicitly labelled measured analyses',n_epochs=200,experimental_coverage_flag='availability only, not vector provenance',observed_used_for_embedding=False)
    (O/'analysis_summary.json').write_text(json.dumps(meta,indent=2));print('ANALYSIS COMPLETE',json.dumps(meta),flush=True)

    np.savez_compressed(O/'measured_response_metrics.npz',rms=observed_rms,hallmark_scores=truth_scores,observed_mask=M,common5=common)
    print('ALL PREDICTION ANALYSIS FINISHED',flush=True)
    return meta
