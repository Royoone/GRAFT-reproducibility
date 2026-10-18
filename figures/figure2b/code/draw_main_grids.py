#!/usr/bin/env python3
"""Render Figure 2b from complete, independently scored internal run tables.

No network access or model execution. Does not select the best configuration
from target performance; the requested method list is an explicit input.
"""
from pathlib import Path
import json,hashlib,os,tempfile
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'graft_comparison_mpl'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pymupdf as fitz
from score_predictions import M2,INTERNAL

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
 'font.size':6,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white',
 'axes.linewidth':.4,'text.color':'#24272B','figure.facecolor':'white'})
INK='#24272B';RED='#C24F4F';BG='#EEF0F2'

LABELS={'GRAFT': 'GRAFT', 'source_mean': 'Source mean', 'xpert_dual_drug_residual': 'XPert', 'deepce_attention': 'DeepCE', 'ciger_attention': 'CIGER (hinge)', 'ciger_ranknet': 'CIGER (RankNet)', 'prnet_repro': 'PRnet', 'transigen_repro': 'TranSiGen', 'chemcpa_repro': 'chemCPA', 'morgan256': 'Morgan-256', 'morgan2048': 'Morgan-2048', 'mlp_unipert': 'UniPert MLP', 'g2cp_author_pooled_staged_d0': 'G2CP', 'hypermap_fullsource_refit_d0': 'HyperMap', 'hypermap_fullsource_refit_d20': 'HyperMap', 'g2cp_adapter_d20': 'G2CP-adapter', 'g2cp_adapter_d10pct': 'G2CP-adapter'}
def label(m):
 if m not in LABELS:raise ValueError("No approved short label: "+m)
 return LABELS[m]

def budget(m):
 s=m.lower()
 return '10%' if 'd10pct' in s else '20' if 'd20' in s else '0'

def norm_values(df,cols):
 rows=[]
 for c in cols:
  lo=float(df[c].min());hi=float(df[c].max())
  for d in df.to_dict('records'):
   v=(d[c]-lo)/(hi-lo) if hi>lo else .5
   if c.startswith(('mse','mae')):v=1-v
   rows.append(dict(method=d['method'],cell=d['cell'],seed=d['seed'],metric=c,raw=d[c],normalized=v,minimum=lo,maximum=hi))
 return pd.DataFrame(rows)

def draw(df,cohort,out,name,methods=None,seeds=(52,229,564),note=''):
 if cohort!='internal' or name!='F2b':raise ValueError('This renderer implements only Figure 2b')
 cols=M2;cells=INTERNAL
 b=df[(df.cohort==cohort)&df.seed.isin(seeds)].copy()
 if methods is not None:b=b[b.method.isin(methods)]
 else:methods=sorted(b.method.unique())
 if set(b.method)!=set(methods):raise ValueError('Requested methods missing from run data')
 for m in methods:
  s=b[b.method==m]
  if len(s)!=len(cells)*len(seeds) or set(map(tuple,s[['cell','seed']].to_numpy()))!=set((c,z) for c in cells for z in seeds):
   raise ValueError('Incomplete/duplicate cells or seeds for '+m)
 if not np.isfinite(b[cols].to_numpy()).all():raise ValueError('Nonfinite plotted metric')
 value=b.groupby(['method','cell'])[cols].mean().groupby('method').mean()
 ranks=pd.DataFrame({c:value[c].rank(method='average',ascending=c.startswith(('mse','mae'))) for c in cols})
 summary=pd.DataFrame({'mean_rank':ranks.mean(1),'first_place_count':ranks.eq(1).sum(1),'r20':value.r20})
 order=summary.sort_values(['mean_rank','r20'],ascending=[True,False]).index.tolist()
 norm=norm_values(b,cols);n=len(order)
 metrics=['r','rho','recall','pos_p','neg_p','mse','mae'];displays=['Pearson','Spearman','Recall','Up P','Down P','MSE','MAE']
 groups=[(f'Top{k}',color,cm,[(f'{m}{k}',d) for m,d in zip(metrics,displays)]) for k,color,cm in [(20,'#079FCF','Blues_r'),(50,'#F36D32','OrRd_r'),(100,'#F4B800','YlOrBr_r')]]
 width=183.;row=4.4;top=28.;body=n*row;bottom=31.;height=top+body+bottom
 fig=plt.figure(figsize=(width/25.4,height/25.4));ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,width);ax.set_ylim(height,0);ax.axis('off')
 def text(x,y,s,**kw):return ax.text(x,y,s,va='center',**kw)
 def box(x,y,w,h,c,ec='none',lw=.35):ax.add_patch(Rectangle((x,y),w,h,facecolor=c,edgecolor=ec,lw=lw))
 text(2.5,5,'b',fontsize=11,weight='bold')
 text(9,5,'Internal benchmark',fontsize=9,weight='bold')
 text(9,10.5,f'5 target backgrounds  |  {len(seeds)} seed'+('s' if len(seeds)>1 else '')+f'  |  {n} configurations  |  {len(cols)} metrics',fontsize=6.5,color='#59616B')
 if len(seeds)==1:text(width-2,5,'SINGLE-SEED PILOT',ha='right',fontsize=7,weight='bold',color='#98591F')
 method_x=2.;method_w=30.;budget_x=33.;budget_w=8.;start=43.;cw=117./len(cols);overall=161.5
 box(method_x,16,method_w,5.5,'#727780');text(method_x+method_w/2,18.75,'Method',ha='center',fontsize=6.4,weight='bold',color='white')
 box(budget_x,16,budget_w,5.5,'#839398');text(budget_x+4,18.75,'Target D',ha='center',fontsize=5.5,weight='bold',color='white')
 text(budget_x+4,24.5,'budget',ha='center',fontsize=5.5)
 fields=[];index=0
 for title,color,cm,spec in groups:
  x=start+index*cw;w=len(spec)*cw-.6
  box(x,16,w,5.5,color);text(x+w/2,18.75,title,ha='center',fontsize=7 if len(spec)>1 else 5.2,weight='bold',color=INK if title=='Top100' else 'white')
  subtitle='Correlation · retrieval · error'
  text(x+w/2,24,subtitle,ha='center',fontsize=5.1)
  for metric,display in spec:fields.append((index,metric,display,cm));index+=1
 box(overall,16,19,5.5,'#4DA534');text(overall+9.5,18.75,'Overall',ha='center',fontsize=7,weight='bold',color='white')
 text(overall+6,24,'Mean rank',ha='center',fontsize=5.5);text(overall+16,24,'No. 1',ha='center',fontsize=5.5)
 for i,m in enumerate(order):
  y=top+i*row
  if i%2==0:box(2,y-row*.45,178.5,row*.9,BG)
  if m.startswith('GRAFT'):box(2,y-row*.45,.5,row*.9,RED)
  text(3,y,label(m),fontsize=6.6,weight='bold' if m.startswith('GRAFT') else 'normal')
  text(budget_x+4,y,budget(m),ha='center',fontsize=6,color='#98591F' if budget(m)!='0' else INK)
  for j,c,display,cm in fields:
   rk=float(ranks.loc[m,c]);co=plt.colormaps[cm](.13+.79*(rk-1)/max(n-1,1))
   values=norm[(norm.method==m)&(norm.metric==c)].sort_values(['cell','seed']).normalized.to_numpy()
   x=start+j*cw+.65;w=cw-1.1
   box(x,y-1.3,w,2.6,'white','#CFD5D8',.2);box(x,y-1.3,w*values.mean(),2.6,co,INK,.3)
   jit=(np.tile([-.7,0,.7],5)+np.repeat([-.15,-.075,0,.075,.15],3)) if len(values)==15 else np.linspace(-.8,.8,len(values))
   ax.scatter(x+w*values,y+jit,s=1.5 if len(values)>5 else 2.0,c=INK,alpha=.72,linewidths=0,zorder=5)
   if rk<=3:text(start+j*cw+.25,y,f'{rk:g}',ha='center',fontsize=5,color=RED)
  mr=float(summary.loc[m,'mean_rank']);co=plt.colormaps['Greens_r'](.13+.79*(mr-1)/max(n-1,1));box(overall+1,y-1.5,10,3,co,INK)
  text(overall+6,y,f'{mr:.2f}',ha='center',fontsize=6.3,color='white' if np.dot(co[:3],[.21,.72,.07])<.55 else INK)
  text(overall+16,y,str(int(summary.loc[m,'first_place_count'])),ha='center',fontsize=6.3)
 for j,c,d,cm in fields:text(start+j*cw+cw*.6,top+body-.5,d,ha='right',rotation=55,rotation_mode='anchor',fontsize=5.5)
 for x in [32.5,42,161]:ax.plot([x,x],[top-2,top+body-row/2],color='#D7DBE0',lw=.35,zorder=0)
 y=top+body+13.0
 text(3,y,'Bars: global min–max normalized scores (higher is better); error metrics are reversed.',fontsize=5.6)
 text(3,y+3.8,'Dots: 5 contexts × 3 computational seeds.',fontsize=5.6)
 text(3,y+7.6,f'Overall: mean of {len(cols)} raw-metric ranks; No. 1: first-place metrics. Red numbers indicate the top three.',fontsize=5.6)
 text(3,y+11.4,'Named baselines are task-adapted implementations. Target D: measured target drug training budget.',fontsize=5.6)
 text(3,y+15.2,note,fontsize=5.4)
 out.mkdir(parents=True,exist_ok=True)
 for ext in ['pdf','svg','png']:fig.savefig(out/(name+'.'+ext),dpi=300 if ext=='png' else None)
 fig.savefig(out/(name+'.tif'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
 plt.close(fig)
 sd=out/'source_data';sd.mkdir(exist_ok=True)
 value.loc[order,cols].to_csv(sd/(name+'_metric_values.csv'));ranks.loc[order,cols].to_csv(sd/(name+'_metric_ranks.csv'));summary.loc[order].to_csv(sd/(name+'_overall_ranking.csv'));norm.to_csv(sd/(name+'_normalized_points.csv'),index=False);b.to_csv(sd/(name+'_runs.csv'),index=False)
 doc=fitz.open(out/(name+'.pdf'));pg=doc[0];spans=[s for b in pg.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
 assert len(doc)==1 and not pg.get_images();assert all(f[2]!='Type3' for f in pg.get_fonts())
 overflow=[s['text'] for s in spans if not pg.rect.contains(fitz.Rect(s['bbox']))]
 if overflow:raise ValueError('PDF text outside page: '+str(overflow))
 pg.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(out/(name+'_pdf_preview.png'))
 svg=(out/(name+'.svg')).read_text();assert '<text ' in svg and '<image ' not in svg
 audit={'file':name,'cohort':cohort,'methods':order,'seeds':list(seeds),'cells':list(cells),'rank_ties':'average','first_place':'rank exactly 1','n_metric_ranks':len(cols),'normalization':'global per metric over cell–seed runs','raw_values_used_for_ranking':True,'width_mm':width,'height_mm':height,'minimum_font_pt':min(s['size'] for s in spans),'fonts':[f[3] for f in pg.get_fonts()],'raster_objects':0,'text_outside_page':overflow,'pdf_sha256':hashlib.sha256((out/(name+'.pdf')).read_bytes()).hexdigest(),'note':note,'display_labels':{m:label(m) for m in order},'target_drug_budgets':{m:budget(m) for m in order},'tiff_dpi':600}
 (out/(name+'_QA.json')).write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'saved':str(out/(name+'.pdf')),'configs':n,'rank_first':order[0]}))
 return audit

