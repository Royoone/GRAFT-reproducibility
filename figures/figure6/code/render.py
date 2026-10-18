"""F6e: paired predicted ROS-programme scores in glucocorticoid groups."""
from panel_export import *
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

HERE=Path(__file__).resolve().parents[1]
F=pd.read_csv(HERE/'source_data/GR_family_ROS_raw.csv',index_col=0)
summary=json.loads((HERE/'source_data/summary.json').read_text())
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
 'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
 'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'lines.linewidth':1,
 'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2.5,'ytick.major.size':2.5,
 'text.color':'#22282b','axes.labelcolor':'#22282b','axes.edgecolor':'#22282b'})
BLUE='#5d7c9d';ORANGE='#be8469'
rng=np.random.default_rng(20260926)
# Preserve the fixed jitter stream of the manuscript's paired panel.
for _ in F.columns:
    rng.uniform(-.19,.19,len(F))
    rng.integers(len(F),size=(5000,len(F)))
A=F.A549.to_numpy();B=F[['A375','HT29','MCF7','PC3']].mean(1).to_numpy()
probability=float(ttest_rel(A,B).pvalue)
assert len(A)==41 and int((A<B).sum())==summary['GR_A549_below_other4']
np.testing.assert_allclose(probability,summary['GR_ROS_paired_p_exploratory'],rtol=1e-10)
fig=plt.figure(figsize=(10.3,7.2));ax=fig.add_axes([.735,.565,.23,.265])
ax.set_title('A549 versus other LINCS cells',loc='left',fontweight='bold',fontsize=8,pad=9)
boxes=ax.boxplot([A,B],positions=[0,1],widths=.42,patch_artist=True,showfliers=False,
 medianprops={'color':'#252b2e'},whiskerprops={'lw':.8},capprops={'lw':.8})
for patch,color in zip(boxes['boxes'],[ORANGE,BLUE]):patch.set_facecolor(color);patch.set_alpha(.7)
for a,b in zip(A,B):ax.plot([0,1],[a,b],c='#aeb4b8',alpha=.3,lw=.4,zorder=0)
for i,v in enumerate([A,B]):ax.scatter(i+rng.uniform(-.09,.09,len(v)),v,s=5,c='#4b545a',alpha=.6,zorder=3)
ax.axhline(0,c='#a7afb3',lw=.6,ls='--')
ax.set_ylim(min(A.min(),B.min())-.15,max(A.max(),B.max())+.28)
ax.set_xticks([0,1],['A549','Other four\n(mean per group)'])
ax.set_ylabel('Predicted signed ROS-gene-set score',fontsize=7)
ax.text(.03,.96,f'{summary["GR_A549_below_other4"]}/41 groups: A549 < other cells',transform=ax.transAxes,fontsize=7,color=ORANGE)
ax.text(.02,-.37,f'Paired t-test P = {probability:.2g}',transform=ax.transAxes,fontsize=7)
export(fig,None,'F6e');plt.close(fig)
(OUT/'F6e_statistics.json').write_text(json.dumps({'groups':41,'A549_below_other_four':int((A<B).sum()),
 'paired_t_p':probability,'score':'Predicted signed ROS-gene-set score; not a measured ROS concentration'},indent=2)+'\n')
