"""Render FS2a/b: fixed genetic-input budgets and retained state dimensions."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
FIVE=['A375','A549','HT29','MCF7','PC3']
CC=['#527caf','#52a291','#957ca9','#c39a56','#c76c72']
P='#8066a6';GRAY='#aeb2b9'
mpl.rcParams.update({'font.family':'Arial','font.size':7.5,'axes.labelsize':8,'axes.titlesize':8,
 'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':.65,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none'})


def save(fig,name,output):
 for extension in ['pdf','svg','png']:fig.savefig(output/(name+'.'+extension),bbox_inches='tight',dpi=350)
 plt.close(fig)


def draw_dimensions(internal,output):
 f,ax=plt.subplots(figsize=(3.35,2.35));curves=[]
 for task,co,ls,label in [('z_topabs',P,'-','Largest |state|'),('z_random',GRAY,'--','Random dimensions')]:
  q=internal[internal.task==task].assign(k=lambda x:x.label.astype(int)).groupby(['k','cell']).r20.mean().unstack();x=q.index.to_numpy();mu=q.mean(1).to_numpy();se=q.sem(1).to_numpy()
  ax.fill_between(x,mu-se,mu+se,color=co,alpha=.13,lw=0);ax.plot(x,mu,c=co,ls=ls,marker='o',ms=3,label=label)
  curves.extend(dict(task=task,k=int(k),mean=float(m),sem=float(s)) for k,m,s in zip(x,mu,se))
 ax.set(xlabel='Retained state dimensions',ylabel='Top20 Pearson r',xticks=[0,4,8,16,24,32]);ax.legend(fontsize=6.5)
 save(f,'FS2b',output)
 return curves


def draw_budgets(internal,output):
 f,axs=plt.subplots(4,5,figsize=(9.1,6.6),gridspec_kw={'hspace':.65,'wspace':.30});curves=[]
 for row,(task,metric) in enumerate([('g_budget','r20'),('g_budget','mse20'),('state','r20'),('state','recall20')]):
  for j,(cell,co) in enumerate(zip(FIVE,CC)):
   ax=axs[row,j];q=internal[internal.cell==cell]
   if task=='g_budget':
    labs=['0','1','2','5','10','20','50','100','200','full'];q=q[q.task==task].groupby(['seed','label'])[metric].mean().unstack().reindex(columns=labs)
    for _,v in q.iterrows():ax.plot(range(10),v,c=co,alpha=.25,lw=.65)
    ax.plot(range(10),q.mean(),c=co,marker='o',ms=2.5);ax.set_xticks([0,2,4,6,9],['0','2','10','50','Full']);ax.set_xlabel('Genetic inputs',fontsize=6.5)
    curves.extend(dict(cell=cell,task=task,metric=metric,label=label,mean=float(value),sd=None) for label,value in q.mean().items())
   else:
    for task2,col,ls in [('z_topabs',co,'-'),('z_random',GRAY,'--')]:
     v=q[q.task==task2].assign(k=lambda x:x.label.astype(int)).groupby('k')[metric].agg(['mean','std']);ax.errorbar(v.index,v['mean'],yerr=v['std'],c=col,ls=ls,marker='o',ms=2,capsize=1)
     curves.extend(dict(cell=cell,task=task2,metric=metric,label=str(k),mean=float(r['mean']),sd=float(r['std'])) for k,r in v.iterrows())
    ax.set_xticks([0,8,16,24,32]);ax.set_xlabel('State dimensions',fontsize=6.5)
   if j==0:ax.set_ylabel({'r20':'Top20 Pearson r','mse20':'Top20 MSE','recall20':'Recall20'}[metric])
   if row==0:ax.set_title(cell,c=co,fontweight='bold')
   ax.tick_params(labelsize=6);ax.locator_params(axis='y',nbins=4)
 save(f,'FS2a',output)
 return curves


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--data',type=Path,default=ROOT/'data/processed/diagnostics/reference/diagnostics.csv')
 parser.add_argument('--output',type=Path,default=ROOT/'results/figures/supplementary')
 args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 d=pd.read_csv(args.data,dtype={'label':str});internal=d[d.cell.isin(FIVE)]
 assert set(internal.cell)==set(FIVE) and set(internal.seed)=={52,229,564}
 dimensions=draw_dimensions(internal,args.output);budgets=draw_budgets(internal,args.output)
 pd.DataFrame(dimensions).to_csv(args.output/'FS2b_curves.csv',index=False)
 pd.DataFrame(budgets).to_csv(args.output/'FS2a_curves.csv',index=False)
 checks=dict(panels=['FS2a','FS2b'],seeds=[52,229,564],contexts=FIVE,
             source_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
             FS2a='Budget draws averaged within seed; faint curves are individual seeds. State-dimension error bars: mean ± sample SD across three seeds.',
             FS2b='Average seeds within each context, then mean ± SEM across five contexts.')
 (args.output/'numeric_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
 print(args.output)


if __name__=='__main__':main()
