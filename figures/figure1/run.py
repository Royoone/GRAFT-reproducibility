"""Render measured cross-cell Top20 correlation matrices F1c and F1d."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT=Path(__file__).resolve().parents[2]
DATA=Path(__file__).resolve().parent/'source_data'
CELLS=['A375','A549','HT29','MCF7','PC3']
INK='#24272b'
BLUE=LinearSegmentedColormap.from_list('blue',['#ffffff','#d6ddec','#879ac5'])
RED=LinearSegmentedColormap.from_list('red',['#ffffff','#ffd1cc','#fa786e'])
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
 'font.size':7.5,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
 'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'lines.linewidth':1.15,
 'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2.5,'ytick.major.size':2.5,
 'text.color':INK,'axes.labelcolor':INK,'axes.edgecolor':INK,'savefig.facecolor':'white',
 'legend.frameon':False,'figure.constrained_layout.h_pad':.09,'figure.constrained_layout.w_pad':.07})


def heat(ax,mat,rows,cols,cmap,vmin=None,vmax=None,fmt='.2f',fs=7,annot=True):
 m=np.asarray(mat,float);im=ax.pcolormesh(np.ma.masked_invalid(m),cmap=cmap,vmin=vmin,vmax=vmax,edgecolors='white',linewidth=.8)
 ax.set_xlim(0,m.shape[1]);ax.set_ylim(m.shape[0],0)
 ax.set_xticks(np.arange(len(cols))+.5,cols);ax.set_yticks(np.arange(len(rows))+.5,rows)
 ax.tick_params(length=0,pad=3);ax.spines[['left','bottom']].set_visible(False)
 if annot:
  for (i,j),v in np.ndenumerate(m):
   if np.isfinite(v):
    rgb=im.cmap(im.norm(v))[:3];lightness=np.dot(rgb,[.2126,.7152,.0722])
    ax.text(j+.5,i+.5,format(v,fmt),ha='center',va='center',fontsize=fs,color='white' if lightness<.61 else INK)
 im._range_extend=('both' if np.nanmin(m)<vmin and np.nanmax(m)>vmax else 'min' if np.nanmin(m)<vmin else 'max' if np.nanmax(m)>vmax else 'neither') if vmin is not None and vmax is not None else 'neither'
 return im


def colorbar(fig,im,ax):
 extend=getattr(im,'_range_extend','neither')
 c=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.045,pad=.10,aspect=32,shrink=.85,extend=extend)
 ticks=[im.norm.vmin,(im.norm.vmin+im.norm.vmax)/2,im.norm.vmax];labels=[f'{x:g}' for x in ticks]
 if extend in ['min','both']:labels[0]='≤'+labels[0]
 if extend in ['max','both']:labels[-1]='≥'+labels[-1]
 c.set_ticks(ticks,labels=labels);c.set_label('Pearson r',fontsize=6.5);c.ax.tick_params(labelsize=6.3);c.outline.set_visible(False)
 return c


def draw(ax,modality,arrays,summary):
 vals=arrays[f'{modality}/pair_directed_r20'];mat=np.full((5,5),np.nan)
 for (source,target),values in zip(arrays[f'{modality}/pair_index'],vals):
  mat[CELLS.index(source),CELLS.index(target)]=np.nanmean(values)
 im=heat(ax,mat,CELLS,CELLS,cmap=BLUE if modality=='genetic' else RED,vmin=.18,vmax=.36)
 ax.set_xlabel('Target cell');ax.set_ylabel('Source cell')
 q=summary[(summary.modality==modality)&(summary.k==20)].iloc[0]
 ax.set_title(f'{modality.capitalize()} · Top20\nMean {q["mean"]:.3f} [{q.lo:.3f}, {q.hi:.3f}]',fontsize=8.2,weight='bold')
 ax.set_aspect('equal')
 mean=float(np.nanmean(np.nanmean(vals,axis=0)))
 np.testing.assert_allclose(mean,float(q['mean']),rtol=0,atol=1e-14)
 return im,mat,dict(mean=mean,lo=float(q.lo),hi=float(q.hi),n=int(q.n))


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',type=Path,default=ROOT/'results/figures/figure1')
 args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 summary=pd.read_csv(DATA/'01_transfer_summary.csv');checks={};matrix_rows=[]
 with np.load(DATA/'arrays.npz',allow_pickle=False) as arrays:
  assert set(arrays.files)=={m+'/'+field for m in ['genetic','chemical'] for field in ['pair_directed_r20','pair_index']}
  for name,modality in [('F1c','genetic'),('F1d','chemical')]:
   f,ax=plt.subplots(figsize=(2.75,3));im,mat,checks[modality]=draw(ax,modality,arrays,summary)
   cb=colorbar(f,im,ax);f.subplots_adjust(bottom=.27);ax.set_xlabel('Target cell',labelpad=3);cb.ax.set_position([.19,.09,.64,.025])
   for extension in ['pdf','svg','png']:f.savefig(args.output/(name+'.'+extension),bbox_inches='tight',dpi=350)
   plt.close(f)
   matrix_rows.extend(dict(modality=modality,source=s,target=t,r20=float(mat[i,j])) for i,s in enumerate(CELLS) for j,t in enumerate(CELLS) if i!=j)
 pd.DataFrame(matrix_rows).to_csv(args.output/'matrix_values.csv',index=False)
 checks.update(panels=['F1c','F1d'],source_array_sha256=hashlib.sha256((DATA/'arrays.npz').read_bytes()).hexdigest(),
               statistical_unit='Perturbation; average available directed cross-cell correlations before averaging perturbations',
               interval='Frozen 95% perturbation-bootstrap interval; no resampling during rendering')
 (args.output/'numeric_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
 print(args.output)


if __name__=='__main__':main()
