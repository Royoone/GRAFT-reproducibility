"""Frozen transcriptional-response scoring contract for Figures 2 and 3."""
from pathlib import Path
import argparse,json,hashlib,warnings
import numpy as np
from scipy.stats import rankdata
EXCLUDED=('BRD-K43389675','BRD-K49371609','BRD-K17516382')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def arrsha(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def corr(a,b,constant_pred_zero=True):
 ac=a-a.mean(1,keepdims=True);bc=b-b.mean(1,keepdims=True)
 na=np.sqrt((ac*ac).sum(1));nb=np.sqrt((bc*bc).sum(1))
 v=(ac*bc).sum(1)/np.maximum(na*nb,1e-12)
 v[(na==0)|(nb==0)]=np.nan
 if constant_pred_zero:v[(np.ptp(a,axis=1)>0)&(np.ptp(b,axis=1)==0)]=0
 return v


def score(y,p,names):
 y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
 if y.shape!=p.shape or len(names)!=len(y):raise ValueError('Shape mismatch')
 if not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('Nonfinite prediction or truth')
 if len(set(names))!=len(names):raise ValueError('Duplicate query names')
 n,g=y.shape;per={};flat={}
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning)
  for k in (20,50,100):
   ix=np.argpartition(np.abs(y),-k,axis=1)[:,-k:]
   jx=np.argpartition(np.abs(p),-k,axis=1)[:,-k:]
   yy=np.take_along_axis(y,ix,axis=1);pp=np.take_along_axis(p,ix,axis=1)
   r=corr(yy,pp);rho=corr(rankdata(yy,method='average',axis=1),rankdata(pp,method='average',axis=1))
   # Historical exclusion also affects rho20 because both use the same mask.
   if k==20:r[np.isin(names,EXCLUDED)]=np.nan;rho[np.isin(names,EXCLUDED)]=np.nan
   vals={'r':r,'rho':rho,'mse':((pp-yy)**2).mean(1),'mae':np.abs(pp-yy).mean(1)}
   vals['recall']=np.array([np.intersect1d(a,b,assume_unique=True).size/k for a,b in zip(ix,jx)])
   for key,sgn in [('pos_p',1),('neg_p',-1)]:
    a=np.argpartition(y*sgn,-k,axis=1)[:,-k:];b=np.argpartition(p*sgn,-k,axis=1)[:,-k:]
    vals[key]=np.array([np.intersect1d(aa,bb,assume_unique=True).size/k for aa,bb in zip(a,b)])
   for key,v in vals.items():per[f'{key}{k}']=v;flat[f'{key}{k}']=float(np.nanmean(v))
   flat[f'n_r{k}']=int(np.isfinite(r).sum());flat[f'rmse{k}']=float(np.sqrt(flat[f'mse{k}']))
   if k==20:per['sign20']=np.mean(np.sign(yy)==np.sign(pp),axis=1)
  # Matches GRAFT supplementary compute.py for full-signature correlation.
  per['all_r']=corr(y,p,constant_pred_zero=False)
  for key in ('all_r','sign20'):flat[key]=float(np.nanmean(per[key]))
 return flat,per

