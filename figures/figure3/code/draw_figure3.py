#!/usr/bin/env python3
"""Render the manuscript's external benchmark panels F3b and F3c only."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'source_data'
plt.rcParams.update({
    'font.family':'Arial', 'font.size':6.5, 'axes.labelsize':6.5,
    'axes.titlesize':7, 'xtick.labelsize':5.8, 'ytick.labelsize':5.8,
    'axes.linewidth':0.45, 'lines.linewidth':0.85, 'pdf.fonttype':42,
    'ps.fonttype':42, 'svg.fonttype':'none', 'savefig.facecolor':'white',
    'figure.facecolor':'white', 'text.color':'#000000', 'axes.labelcolor':'#000000',
    'xtick.color':'#000000', 'ytick.color':'#000000', 'axes.edgecolor':'#707D85',
})
INK='#000000'; MUTED='#000000'
CELL_ORDER=['HepG2','MCF10A','Jurkat','K562_L1000','HAP1']
CELL_COLORS=dict(zip(CELL_ORDER,['#5B7EA4','#AA849B','#689C8A','#C6945C','#8C94AC']))
CELL_NAMES={x:x.replace('_L1000','') for x in CELL_ORDER}
METHOD_NAMES={'B':'GRAFT','prnet':'PRnet','transigen':'TranSiGen',
 'chemcpa':'chemCPA','mlp_morgan':'Morgan MLP','deepce':'DeepCE',
 'source_mean':'Source mean','xpert_dual':'XPert','ciger':'CIGER','g2cp_d0':'G2CP-D0'}
METRICS=['r20','rho20','r50','r100','all_r','recall20','pos_p20','neg_p20','sign20','mse20','mae20','mae50']
FILES={'b':'b_benchmark.csv','b_points':'b_points.csv','b_gains':'b_gains.csv',
       'b_runs':'b_run_metrics.csv','c_rank':'c_metric_ranks.csv','c_value':'c_metric_values.csv',
       'c_norm':'c_normalized_cells.csv','c_overall':'c_overall_ranking.csv'}
TABLES={}
METHODS=[]


def load():
    global TABLES, METHODS
    manifest=json.loads((DATA/'manifest.json').read_text())
    for filename in FILES.values():
        digest=hashlib.sha256((DATA/filename).read_bytes()).hexdigest()
        if digest != manifest['outputs'][filename]['sha256']:
            raise ValueError('Source-table checksum mismatch: '+filename)
    TABLES={key:pd.read_csv(DATA/filename) for key,filename in FILES.items()}
    METHODS=TABLES['c_overall'].sort_values(['mean_rank','r20'],ascending=[True,False]).method.tolist()


def check_data():
    runs=TABLES['b_runs']
    assert len(runs)==150 and not runs.duplicated(['method','cell','seed']).any()
    assert set(runs.cell)==set(CELL_ORDER) and set(runs.seed)=={52,229,564}
    values=runs.groupby(['method','cell'])[METRICS].mean().groupby('method').mean()
    reported=TABLES['c_value'].set_index('method')
    np.testing.assert_allclose(values.loc[reported.index,METRICS],reported[METRICS],rtol=0,atol=1e-12)
    ranks=values.copy()
    for metric in METRICS:
        ranks[metric]=values[metric].rank(ascending=metric.startswith(('mse','mae')),method='average')
    reported_ranks=TABLES['c_rank'].set_index('method')
    np.testing.assert_array_equal(ranks.loc[reported_ranks.index,METRICS],reported_ranks[METRICS])
    overall=TABLES['c_overall'].set_index('method')
    np.testing.assert_allclose(ranks.mean(axis=1).loc[overall.index],overall.mean_rank,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(ranks.eq(1).sum(axis=1).loc[overall.index],overall.first_place_count)
    for method in METHODS:
        cell_means=runs[runs.method==method].groupby('cell').r20.mean()
        for panel in CELL_ORDER+['macro']:
            expected=(runs[(runs.method==method)&(runs.cell==panel)].r20 if panel!='macro' else cell_means)
            row=TABLES['b'][(TABLES['b'].method==method)&(TABLES['b'].panel==panel)].iloc[0]
            np.testing.assert_allclose([row['mean'],row.sd],[expected.mean(),expected.std(ddof=1)],rtol=0,atol=1e-12)
    macro=float(TABLES['b'].query("method=='B' and panel=='macro'")['mean'].iloc[0])
    np.testing.assert_allclose(macro,.60619973738475,rtol=0,atol=1e-12)
    np.testing.assert_allclose(overall.loc['B','mean_rank'],17/12,rtol=0,atol=1e-12)
    assert int(overall.loc['B','first_place_count'])==9
    return dict(panels=['F3b','F3c'],source_validation='passed',external_GRAFT_macro_r20=macro,
                mean_rank=float(overall.loc['B','mean_rank']),first_place_metrics=9,
                methods=10,metrics=12,cells=5,seeds=[52,229,564],
                spread='Target panels: sample SD across 3 seeds; mean panel: sample SD across 5 cell means')


def axmm(fig, r):
    x,y,w,h=r; fw,fh=fig.get_size_inches()*25.4
    return fig.add_axes([x/fw,1-(y+h)/fh,w/fw,h/fh])

def local(fig,r,dx,dy,w,h):
    return axmm(fig,[r[0]+dx,r[1]+dy,w,h])

def textmm(fig,r,x,y,s,size=6.5,**kw):
    fw,fh=fig.get_size_inches()*25.4
    return fig.text((r[0]+x)/fw,1-(r[1]+y)/fh,s,fontsize=size,va='top',**kw)

def title(fig,r,letter,txt,subtitle=None):
    textmm(fig,r,0,0,letter,12,weight='bold',color=INK)
    textmm(fig,r,5,0.6,txt,7.6,weight='bold',color=INK)
    if subtitle:textmm(fig,r,5,4.2,subtitle,5.8,color=MUTED)

def panel_b(fig,r):
    title(fig,r,'b','External prediction performance','Matched r20 axes · intervals: mean ± SD')
    contexts=CELL_ORDER+['macro'];rows=TABLES['b']
    label_ax=local(fig,r,1,14,21,47)
    label_ax.set_xlim(0,1);label_ax.set_ylim(9.5,-1.48);label_ax.set_axis_off()
    cw=(r[2]-24)/6
    for i,m in enumerate(METHODS):
        co=rows[rows.method==m].color.iloc[0]
        label_ax.text(.99,i,METHOD_NAMES[m],ha='right',va='center',fontsize=6.5,
                      color=co if m in ['B','source_mean'] else INK,
                      weight='bold' if m in ['B','source_mean'] else 'normal')
    for j,c in enumerate(contexts):
        a=local(fig,r,23+j*cw,14,cw-1.6,47)
        a.set_xlim(-.075,.92);a.set_ylim(9.5,-1.48)
        a.set_yticks([]);a.set_xticks([0,.8]);a.tick_params(length=1.5,pad=1,labelsize=5.3)
        a.spines[['left','right','top']].set_visible(False)
        a.spines['bottom'].set_color('#A3AEB4')
        for xx in [0,.4,.8]:a.axvline(xx,color='#E6EBEE',lw=.45,zorder=0)
        for i,m in enumerate(METHODS):
            z=rows[(rows.method==m)&(rows.panel==c)].iloc[0]
            co=z.color
            if i%2==0:a.axhspan(i-.45,i+.45,fc='#EEF0F2',zorder=0)
            if m=='B':a.axhspan(i-.45,i+.45,fc='#EDF2E9',zorder=0)
            a.errorbar(z['mean'],i,xerr=z.sd,fmt='o',color=co,ms=2.9 if m=='B' else 2.45,
                       capsize=1.5,lw=.75,zorder=3,mew=.4)
            if m in ['B','source_mean']:
                a.text(.92,i+.34,f'{z["mean"]:.2f}',ha='right',va='center',fontsize=5.0,
                       weight='bold' if m=='B' else 'normal',color=co,
                       zorder=4)
        a.text(.41,-1.29,CELL_NAMES.get(c,'Mean'),ha='center',va='center',fontsize=6.2,weight='bold')
        a.plot([-.075,.92],[-.89,-.89],color=CELL_COLORS.get(c,INK),lw=1,clip_on=False)
    gains=TABLES['b_gains'].set_index('panel')
    lower=local(fig,r,23,64,r[2]-24,3)
    lower.set_xlim(-.5,5.5);lower.set_ylim(0,1);lower.set_axis_off()
    lower.text(-.61,.6,'vs source mean',ha='right',va='center',fontsize=5.5,color=MUTED)
    for j,c in enumerate(contexts):
        lower.text(j,.6,f'+{gains.loc[c,"relative_gain_percent"]:.0f}%',ha='center',va='center',
                   fontsize=6.8,color='#6A9D59',weight='bold')
    textmm(fig,r,23,70,'Top20 r · target SD: 3 seeds · Mean SD: 5 cells',5.4,color=MUTED)

def panel_c(fig,r):
    title(fig,r,'c','Multi-metric ranking','External backgrounds · 10 methods · 12 metrics')
    d=TABLES['c_norm'];rank=TABLES['c_rank'].set_index('method');ov=TABLES['c_overall'].set_index('method')
    # Same visual encoding and palette as the existing Fig2b ranking table.
    groups=[('Top20','#079FCF','Blues_r',[
       ('r20','Pearson'),('rho20','Spearman'),('recall20','Recall'),('pos_p20','Up P'),
       ('neg_p20','Down P'),('sign20','Sign'),('mse20','MSE'),('mae20','MAE')]),
       ('Top50','#F36D32','OrRd_r',[('r50','Pearson'),('mae50','MAE')]),
       ('Top100','#F4B800','YlOrBr_r',[('r100','Pearson')]),
       ('All','#8A849D','Purples_r',[('all_r','Pearson')])]
    a=local(fig,r,1,8,r[2]-2,44)
    a.set_xlim(-3.4,14.6);a.set_ylim(9.7,-3.3);a.set_axis_off()
    def rect(x,y,w,h,fc,ec='none',lw=.35):
        a.add_patch(Rectangle((x,y),w,h,fc=fc,ec=ec,lw=lw))
    rect(-3.35,-2.98,2.94,.83,'#767D85')
    a.text(-1.88,-2.56,'Method',ha='center',va='center',fontsize=6.3,color='white',weight='bold')
    fields=[];colindex=0
    for label,col,cm,spec in groups:
        nn=len(spec);rect(colindex-.38,-3.12,nn-.10,1.0,col)
        a.text(colindex+(nn-1)/2,-2.62,'Top\n100' if label=='Top100' else label,
               ha='center',va='center',fontsize=5.0 if label=='Top100' else 6.2,
               color=INK if label=='Top100' else 'white',weight='bold',linespacing=.95)
        rect(colindex-.38,-2.01,nn-.10,.52,colors.to_rgba(col,.18))
        a.text(colindex+(nn-1)/2,-1.75,'Correlation · retrieval · sign · error' if nn==8 else ('Corr · error' if nn==2 else 'Corr'),
               fontsize=4.4,ha='center',va='center')
        for metric,display in spec:fields.append((colindex,metric,display,cm));colindex+=1
    rect(11.8,-2.98,2.71,.83,'#4DA534')
    a.text(13.16,-2.56,'Overall',ha='center',va='center',fontsize=6.2,color='white',weight='bold')
    a.text(12.45,-1.75,'Mean rank',ha='center',va='center',fontsize=4.9)
    a.text(13.91,-1.75,'No. 1',ha='center',va='center',fontsize=4.9)
    for i,m in enumerate(METHODS):
        if i%2==0:rect(-3.35,i-.46,17.86,.90,'#EEF0F2')
        if m=='B':rect(-3.35,i-.46,.055,.90,'#C24F4F')
        a.text(-3.21,i,METHOD_NAMES[m],ha='left',va='center',fontsize=6.0,
               weight='bold' if m=='B' else 'normal',color=INK)
        for j,metric,display,cm in fields:
            z=d[(d.method==m)&(d.metric==metric)].normalized.to_numpy()
            rk=float(rank.loc[m,metric]);co=plt.colormaps[cm](.13+.79*(rk-1)/9)
            xx=j-.31;ww=.67
            rect(xx,i-.28,ww,.56,'white','#CFD5D8',.25)
            rect(xx,i-.28,ww*z.mean(),.56,co,INK,.35)
            a.scatter(xx+ww*z,i+np.linspace(-.2,.2,5),s=1.9,color=INK,alpha=.85,linewidths=0,zorder=4)
            if rk<=3:a.text(j-.425,i,str(int(rk)),ha='center',va='center',fontsize=4.5,color='#C24F4F',zorder=5)
        mr=float(ov.loc[m,'mean_rank']);co=plt.colormaps['Greens_r'](.13+.79*(mr-1)/9)
        rect(11.92,i-.32,1.05,.64,co,INK,.3)
        a.text(12.445,i,f'{mr:.2f}',ha='center',va='center',fontsize=5.9,
               color='white' if np.dot(co[:3],[.21,.72,.07])<.56 else INK,weight='bold' if m=='B' else 'normal')
        a.text(13.91,i,str(int(ov.loc[m,'first_place_count'])),ha='center',va='center',fontsize=5.9,
               weight='bold' if m=='B' else 'normal')
    for j,metric,display,cm in fields:
        a.text(j,9.85,display,rotation=55,ha='right',va='top',fontsize=5.2)
    textmm(fig,r,2,61.5,'Normalized bars: higher is better · dots: 5 cells · red: top 3',5.3,color=MUTED)
    la=local(fig,r,87,53,15,5);la.set_axis_off();la.set_xlim(0,10);la.set_ylim(0,1)
    for j,(_,col,cm,_) in enumerate(groups):
        la.imshow(np.linspace(.13,.92,256)[None,:],extent=[0,10,j*.14,j*.14+.1],
                  cmap=plt.colormaps[cm],vmin=0,vmax=1,aspect='auto')
    la.text(0,-.03,'1 (best)',fontsize=4.7,va='top')
    la.text(10,-.03,'10',fontsize=4.7,ha='right',va='top')
    la.text(0,.81,'Rank',fontsize=4.7,va='center')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    load()
    checks=check_data()
    for letter,width,height,draw in [('b',102,74,panel_b),('c',103,63,panel_c)]:
        fig=plt.figure(figsize=((width+6)/25.4,(height+5)/25.4))
        draw(fig,(3,2,width,height))
        for extension in ('pdf','svg'):
            fig.savefig(output/f'F3{letter}.{extension}')
        fig.savefig(output/f'F3{letter}.png',dpi=400)
        plt.close(fig)
    (output/'numeric_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print('Exported F3b and F3c to',output)


if __name__=='__main__':main()
