"""Joint correlation/error panel from archived cell means; no model execution."""
from pathlib import Path
import os,json,argparse
os.environ.setdefault('MPLCONFIGDIR',str(Path.cwd()/'results/cache/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
(OUT/"source_data").mkdir(exist_ok=True)
D=pd.read_csv(ROOT/'source_data/frozen_ablation_cell_means.csv').set_index(['arm','cell'])
C=pd.read_csv(ROOT/'source_data/frozen_ablation_paired_contrasts.csv')
CELLS=['A375','A549','HT29','MCF7','PC3']
ARMS=['state_off','scale_only','shift_only','full']
COLORS=['#9ba6b2','#6d91ae','#b88b60','#8066a6']
MARKERS=['X','^','s','D']
NAMES=['State off','Scale only','Shift only','Full GRAFT']
mpl.rcParams.update({'font.family':'Arial','font.size':6.5,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'text.color':'#24272b','axes.labelcolor':'#24272b','xtick.major.size':2.5,'ytick.major.size':2.5,'legend.frameon':False})
def ci(v):
 v=np.asarray(v);return v.mean(),t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))
fig=plt.figure(figsize=(3.50,2.85))
fig.text(.05,.977,'Conditional modulation in GRAFT',weight='bold',fontsize=8,va='top')
fig.text(.05,.917,'Frozen weights · five target cell lines',fontsize=6.5,color='#6c6576')
ax=fig.add_axes([.15,.245,.81,.61])
# Each point retains both metric values; tiny points are individual cell means.
for arm,col,mk,name in zip(ARMS,COLORS,MARKERS,NAMES):
 v=D.loc[arm].reindex(CELLS);x,xe=ci(v.r20);y,ye=ci(v.mse20)
 ax.scatter(v.r20,v.mse20,s=11,marker=mk,facecolor=col,edgecolor='none',linewidth=0,alpha=.55,zorder=2)
 err=ax.errorbar(x,y,xerr=xe,yerr=ye,fmt=mk,color=col,markeredgecolor=col,markeredgewidth=0,markersize=5.5,elinewidth=.8,capsize=2,zorder=4)
 for cap in err[1]: cap.set_markeredgewidth(.8)
 # Direct labels with leaders make overlapping full/shift means identifiable.
 offsets={'state_off':(-1,23),'scale_only':(-3,16),'shift_only':(-58,10),'full':(-49,-22)}
 dx,dy=offsets[arm]
 ax.annotate(name,(x,y),xytext=(dx,dy),textcoords='offset points',ha='center',va='center',fontsize=6.2,color=col,weight='bold' if arm=='full' else 'normal',arrowprops=dict(arrowstyle='-',lw=.5,color=col,shrinkA=2,shrinkB=4))
ax.set(xlim=(.45,.79),ylim=(7.1,20.3),xticks=[.5,.6,.7],yticks=[8,12,16,20],xlabel='Top20 Pearson r  →',ylabel='Top20 MSE  ↓')
ax.grid(axis='y',color='#ededed',lw=.4,zorder=0)
# A paired-difference inset resolves the small scaling contribution without
# rescaling or hiding the large effects in the main absolute-metric plot.
ins=fig.add_axes([.655,.663,.265,.17],facecolor='white')
x=D.loc['full','r20'].reindex(CELLS)-D.loc['shift_only','r20'].reindex(CELLS)
y=D.loc['shift_only','mse20'].reindex(CELLS)-D.loc['full','mse20'].reindex(CELLS)
mx,hx=ci(x);my,hy=ci(y)
ins.axhline(0,c='#c9c9c9',lw=.5);ins.axvline(0,c='#c9c9c9',lw=.5)
ins.scatter(x,y,s=8,c=COLORS[-1],alpha=.55,edgecolors='none',linewidths=0,zorder=3)
ins.errorbar(mx,my,xerr=hx,yerr=hy,fmt='D',color=COLORS[-1],ms=3,elinewidth=.65,capsize=1.3,zorder=4)
ins.set(xlim=(-.0003,.0041),ylim=(-.045,.47),xticks=[0,.003],yticks=[0,.3])
ins.set_xticklabels(['0','0.003']);ins.set_yticklabels(['0','0.3'])
ins.tick_params(labelsize=5, pad=1,length=1.5)
ins.set_xlabel('Correlation gain',fontsize=5.2,labelpad=1)
ins.set_ylabel('MSE reduction',fontsize=5.2,labelpad=1)
ins.set_title('Full vs. shift only',fontsize=5.8,pad=3,weight='bold')
fig.text(.15,.124,'Large symbols: mean ± 95% CI; small symbols: cell means.',fontsize=5.5)
fig.text(.15,.077,'Full vs. shift only: Δr = +0.0031; MSE reduction = 0.306.',fontsize=5.5)
fig.text(.15,.036,'Paired Holm-adjusted P: r = 0.000025; MSE = 0.00132.',fontsize=5.5)
name='F1i'
fig.savefig(OUT/(name+'.pdf'))
qa=OUT/'QA';qa.mkdir(exist_ok=True)
fig.savefig(qa/(name+'_joint.png'),dpi=260)
plt.close(fig)
pd.DataFrame({'cell':CELLS,'correlation_gain_full_vs_shift':x.values,'MSE_reduction_full_vs_shift':y.values}).to_csv(OUT/'source_data/modulation_joint_paired_inset.csv',index=False)
print(OUT/(name+'.pdf'))
