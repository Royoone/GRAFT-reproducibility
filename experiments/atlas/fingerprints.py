"""Original manuscript post-analysis statistics; unchanged tests and grouping."""
from pathlib import Path
import numpy as np,pandas as pd

def export_fingerprints(output):
    O=Path(output)
    z=np.load(O/'atlas13_entities.npz');X=z['response'];Xobs=z['observed'];a=pd.read_csv(O/'chemical_catalog.csv');common=z['observed_mask'][:5].all(0)
    ma=a.PI3K.to_numpy()&~a.mTOR.to_numpy();mb=a.mTOR.to_numpy()&~a.PI3K.to_numpy();rows=[]
    for mode,dat,ok in [('Atlas13',X.mean(0),np.ones(len(a),bool)),('Measured5',Xobs[:5].mean(0),common)]:
     aa=dat[ma&ok].mean(0);bb=dat[mb&ok].mean(0)
     for g,x,y in zip(z['genes'],aa,bb):rows.append(dict(mode=mode,gene=g,pi3k=x,mtor=y))
    pd.DataFrame(rows).to_csv(O/'pi3k_mtor_disjoint_gene_profiles.csv',index=False)
