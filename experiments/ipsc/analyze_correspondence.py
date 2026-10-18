"""Reproduce the fixed Figure 7 drug/genetic association and retention analysis.

The 235-entity audit is the fixed input, including mapping and knockdown flags.
The adopted retention applies exclusions only to non-positive associations;
it is outcome-conditioned. No new lines, examples, score signs or filters are
selected. Exact Spearman tails use the original exhaustive-rank dynamic program.
"""
from pathlib import Path
from collections import Counter
import argparse, hashlib, json, math, sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

REPO=Path(__file__).resolve().parents[2]
LINES=['eipl_3','fiaj_3','iudw_1','jejf_3','kolf_3','oikd_5','paab_3','pipw_4','tolg_6','zapk_3']
_NULL={}
def null(n):
    if n in _NULL:return _NULL[n]
    states=[None]*(1<<n);states[0]=Counter({0:1})
    for mask in range((1<<n)-1):
        position=bin(mask).count('1')+1
        for k in range(n):
            if mask&(1<<k):continue
            target=mask|(1<<k)
            if states[target] is None:states[target]=Counter()
            for score,count in states[mask].items():states[target][score+position*(k+1)]+=count
    assert sum(states[-1].values())==math.factorial(n)
    _NULL[n]=states[-1];return _NULL[n]

def stats(x,y):
    good=np.isfinite(x)&np.isfinite(y);x=x[good];y=y[good];n=len(x)
    pr=rankdata(-x);mr=rankdata(-y)
    if n<3 or np.std(x)==0 or np.std(y)==0:return {'rho':np.nan,'p_exact':np.nan,'n_lines':n}
    rho=float(spearmanr(pr,mr).statistic)
    if abs(rho)<1e-12:rho=0.
    if len(np.unique(pr))!=n or len(np.unique(mr))!=n:
        raise ValueError('The adopted exact-permutation calculation expects tie-free ranks')
    center=n*(n+1)**2//2;deviation=abs(int(2*np.dot(pr,mr))-center)
    extreme=sum(count for score,count in null(n).items() if abs(2*score-center)>=deviation)
    return {'rho':rho,'p_exact':extreme/math.factorial(n),'n_lines':n}

def verify_source_scores(frame, input_path, bundle, catalog):
    sys.path.insert(0,str(REPO))
    from experiments.ipsc.predict import query_catalog
    with np.load(bundle/'reference.npz',allow_pickle=False) as z:ref={k:z[k] for k in z.files}
    _,entities,_,base=query_catalog(ref,catalog)
    with np.load(input_path,allow_pickle=False) as z:
        lines=list(z['contexts'].astype(str));targets=list(z['perturbations'].astype(str))
        effect=z['effects__lognorm_nt'].astype(np.float64)
        good=z['valid'].astype(bool)&(z['n_cells']>=10)&(z['n_guides']>=2)&(z['n_controls']>=50)[:,None]&np.isfinite(effect).all(2)
    ci=np.array([lines.index(x) for x in LINES]);ei={key:i for i,key in enumerate(entities)};ti={target:i for i,target in enumerate(targets)}
    values=np.full((len(frame),len(LINES)),np.nan)
    for i,row in enumerate(frame.itertuples()):
        source=base[ei[row.chemical_key]]
        top=np.argsort(-np.abs(source.astype(np.float64)),kind='stable')[:20]
        direction=np.where(source[top]>=0,1.,-1.)
        selected=[ti[t] for t in row.exact_measured_targets.split('|') if t in ti]
        for j,c in enumerate(ci):
            eligible=[t for t in selected if good[c,t]]
            if eligible:values[i,j]=(effect[c,eligible][:,top]*direction).mean(1).mean()
    supplied=frame[['measured_signedTop20_'+line for line in LINES]].to_numpy(float)
    if not np.array_equal(np.isfinite(values),np.isfinite(supplied)):
        raise ValueError('Recomputed genetic-score missingness differs from the fixed audit')
    error=float(np.nanmax(np.abs(values-supplied)))
    if error>1e-10:raise ValueError(f'Source-directed score mismatch: {error}')
    return {'verified':True,'max_absolute_difference':error,'missingness_equal':True}

def analyze(args):
    frame=pd.read_csv(args.source)
    if len(frame)!=235 or not frame.chemical_key.is_unique:raise ValueError('Expected the fixed 235 distinct entities')
    P=frame[['predicted_RMS_'+line for line in LINES]].to_numpy(float)
    M=frame[['measured_signedTop20_'+line for line in LINES]].to_numpy(float)
    result=pd.DataFrame([stats(p,m) for p,m in zip(P,M)])
    error=float(np.max(np.abs(result.rho.to_numpy()-frame.rho.to_numpy())))
    if error>1e-12:raise ValueError('Drug correlations do not reproduce the fixed audit')
    positive=result.rho.gt(0)
    mapping_remove=(~positive)&frame.step1_excluded
    knockdown_remove=(~positive)&(~mapping_remove)&(~frame.step2eligible)
    retained=~(mapping_remove|knockdown_remove)
    for recomputed,column in [(mapping_remove,'v2_step1_remove'),(knockdown_remove,'v2_step2_remove'),(retained,'v2_retained')]:
        if not np.array_equal(recomputed,frame[column]):raise ValueError('The adopted retention flags do not reproduce')
    assert (int(positive.sum()),int(mapping_remove.sum()),int(knockdown_remove.sum()),int(retained.sum()))==(158,7,37,191)
    args.output.mkdir(parents=True,exist_ok=True)
    checked=frame.copy();checked['recomputed_rho']=result.rho;checked['recomputed_p_exact']=result.p_exact
    checked.to_csv(args.output/'all235_recomputed.csv',index=False)
    kept=frame[retained].reset_index(drop=True);kept.to_csv(args.output/'retained191.csv',index=False)
    class_rows=[];member_rows=[]
    for moa,block in kept.groupby('fig7f_moa',sort=False):
        if len(block)<5:continue
        x=block[['predicted_RMS_'+line for line in LINES]].to_numpy(float)
        y=block[['measured_signedTop20_'+line for line in LINES]].to_numpy(float)
        common=(np.isfinite(x)&np.isfinite(y)).all(0)
        # The source routine computes available-member means, then restricts to
        # the common lines; on those lines every member is present.
        a=x.mean(0)
        count=np.isfinite(y).sum(0)
        b=np.divide(np.nansum(y,axis=0),count,out=np.full(len(LINES),np.nan),where=count>0)
        row={'moa':moa,'n_drugs':len(block),**stats(a[common],b[common]),'included_lines':';'.join(np.asarray(LINES)[common])}
        class_rows.append(row)
        for i,rec in enumerate(block.itertuples()):
            member_rows.append({'chemical_key':rec.chemical_key,'drug_name':rec.drug_name,'moa':moa,**stats(x[i,common],y[i,common])})
    classes=pd.DataFrame(class_rows).sort_values('rho',ascending=False)
    members=pd.DataFrame(member_rows)
    assert len(classes)==8 and len(members)==104
    classes.to_csv(args.output/'moa8_recomputed.csv',index=False);members.to_csv(args.output/'moa_members_recomputed.csv',index=False)
    frozen=pd.read_csv(REPO/'figures/figure7/source_data/moa/plot_moa8_consistent_members_common_lines.csv').set_index('moa')
    class_error=float(np.max(np.abs(classes.set_index('moa').rho-frozen.rho)))
    if class_error>1e-12:raise ValueError('Class-pooled correlations differ from the adopted figure')
    exact_error=float(np.max(np.abs(classes.set_index('moa').p_exact-frozen.p_exact)))
    if exact_error>1e-12:raise ValueError('Exact class P values differ from the adopted figure')
    report={'status':'pass','n_evaluable_entities':235,'positive':158,'nonpositive':77,
            'mapping_exclusions_among_nonpositive':7,'additional_knockdown_exclusions_among_nonpositive':37,
            'retained_entities':191,'retained_positive':158,'retained_nonpositive':33,
            'positive_fraction_in_retained':158/191,'outcome_conditioned_retention':True,
            'fixed_lines':LINES,'n_moas':8,'n_class_member_drugs':104,
            'max_drug_rho_difference':error,'max_class_rho_difference':class_error,'max_class_exact_P_difference':exact_error,
            'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
            'interpretation':'Exploratory correspondence in a selected set; genetic measurements reused from model inputs; not independent drug-treatment validation'}
    if args.input and args.bundle and args.catalog:
        report['source_directed_score']=verify_source_scores(frame,args.input,args.bundle,args.catalog)
    (args.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=REPO/'figures/figure7/source_data/all235_evaluation_and_retention.csv')
    p.add_argument('--output',type=Path,default=REPO/'results/ipsc/correspondence')
    p.add_argument('--input',type=Path)
    p.add_argument('--bundle',type=Path)
    p.add_argument('--catalog',type=Path)
    analyze(p.parse_args())
