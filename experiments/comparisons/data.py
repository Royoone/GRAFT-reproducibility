"""Prepared response tables and fixed query identities for comparisons."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiments.backbone.data import FIVE

EXTERNAL=('HepG2','MCF10A','Jurkat','K562_L1000','HAP1')


def load(data_root, cohort, device='cpu'):
    root=Path(data_root); rows={}; genes=None
    for cell in FIVE+(EXTERNAL if cohort=='external' else ()):
        path=root/('sources' if cell in FIVE else 'external')/f'{cell}.npz'
        with np.load(path,allow_pickle=False) as z:
            panel=z['genes'].astype(str)
            if genes is None:genes=panel
            np.testing.assert_array_equal(genes,panel)
            rows[cell]={'genes':genes}
            for kind in ['genetic','chemical']:
                rows[cell][kind]={'names':z[kind+'_names'].astype(str),
                    'x':torch.tensor(z[kind+'_embeddings'],device=device),
                    'y':torch.tensor(z[kind+'_native'],device=device)}
            if 'query_ids' in z:rows[cell]['query_ids']=z['query_ids'].astype(str)
    common=set.intersection(*(set(rows[c]['chemical']['names']) for c in FIVE))
    import hashlib
    q=np.asarray(sorted(common,key=lambda n:hashlib.sha256(('cancer_fraction_screen_v1:'+str(n)).encode()).hexdigest())[:len(common)//5])
    queries={c:(q if c in FIVE else rows[c]['query_ids']) for c in (FIVE if cohort=='internal' else EXTERNAL)}
    return rows,genes,queries


def array_view(rows):
    return {c:{'names':d['chemical']['names'],'y':d['chemical']['y'].cpu().numpy(),
        'chem_emb':d['chemical']['x'].cpu().numpy(),'gen_names':d['genetic']['names'],
        'gen_y':d['genetic']['y'].cpu().numpy(),'gen_emb':d['genetic']['x'].cpu().numpy()} for c,d in rows.items()}


def smiles(path):
    frame=pd.read_csv(path)
    return {str(r.Compound):r.SMILES for r in frame.itertuples() if isinstance(r.SMILES,str) and r.SMILES}
