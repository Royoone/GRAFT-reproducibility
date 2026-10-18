"""F2d: fixed drug-level source performance versus GRAFT improvement."""
from panel_export import *
import json
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parents[1]
D = pd.read_csv(HERE/'source_data/drug_gain.csv')
D = D[D.r_valid20 > 0]
CELLS = ['A375','A549','HT29','MCF7','PC3']
COLORS = ['#527CAF','#52A291','#957CA9','#C39A56','#C76C72']
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
 'font.size':7,'axes.labelsize':7.2,'axes.titlesize':8,'xtick.labelsize':6.5,
 'ytick.labelsize':6.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
 'xtick.major.width':.5,'ytick.major.width':.5,'xtick.major.size':2.5,'ytick.major.size':2.5,
 'text.color':'#24272B','axes.labelcolor':'#24272B','axes.edgecolor':'#24272B'})
fig = plt.figure(figsize=(7.2,6.8))
grid = fig.add_gridspec(1,5,left=.10,right=.98,bottom=.16,top=.345,wspace=.25)
rows = []
for j,(cell,color) in enumerate(zip(CELLS,COLORS)):
    ax=fig.add_subplot(grid[j]);z=D[D.target==cell]
    ax.set_title(cell,loc='left',fontsize=8,weight='bold',pad=9)
    ax.scatter(z.src_r20,z.delta_r20,c=color,s=3,alpha=.28,edgecolors='none',linewidths=0)
    ax.axhline(0,c='#9BA6B2',lw=.7,ls='--')
    rho=float(spearmanr(z.src_r20,z.delta_r20).statistic)
    ax.text(.04,.95,f'ρ = {rho:.2f}',transform=ax.transAxes,va='top',fontsize=6.5)
    ax.set_xlim(-.55,1.02);ax.set_ylim(-.35,1.35)
    ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1]);ax.set_xlabel('Source r20')
    if j==0:ax.set_ylabel('Δ Top20 Pearson r')
    else:ax.tick_params(labelleft=False)
    rows.append(dict(cell=cell,n_drugs=len(z),spearman_rho=rho))
assert all(r['n_drugs']==641 for r in rows)
export(fig,None,'F2d');plt.close(fig)
pd.DataFrame(rows).to_csv(OUT/'F2d_statistics.csv',index=False)
