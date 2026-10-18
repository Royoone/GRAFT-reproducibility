"""Redraw the retained quantitative manuscript panels from frozen inputs."""
from pathlib import Path
import argparse,hashlib,json,os,tempfile
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"graft-figure-cache"))
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.patches import Rectangle,Patch
from matplotlib.ticker import MaxNLocator
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'source_data'
mpl.rcParams['pdf.fonttype']=42;mpl.rcParams['svg.fonttype']='none'

def export_native(f,output,name,legend=None):
    for artist in list(f.texts)+list(f.legends):artist.set_visible(artist is legend)
    f.canvas.draw();renderer=f.canvas.get_renderer()
    boxes=[a.get_tightbbox(renderer) for a in f.axes]
    if legend is not None:boxes.append(legend.get_window_extent(renderer))
    bbox=Bbox.union(boxes).transformed(f.dpi_scale_trans.inverted())
    bbox=Bbox.from_extents(bbox.x0-.07,bbox.y0-.07,bbox.x1+.07,bbox.y1+.07)
    for ext in ['pdf','svg','png']:f.savefig(output/(name+'.'+ext),bbox_inches=bbox,dpi=350,facecolor='white')
    plt.close(f)

def verify_inputs():
    manifest=json.loads((ROOT/'input_manifest.json').read_text())
    for row in manifest['files']:
        path=ROOT/row['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Immutable input changed: '+row['path'])
    return len(manifest['files'])

INTERNAL=['A375','A549','HT29','MCF7','PC3']
EXTERNAL=['HepG2','MCF10A','Jurkat','K562_L1000','HAP1']
BLUE, GREEN, GRAY, INK = '#6488B5', '#6a9d59', '#9ba6b2', '#24272B'
CC = ['#527caf', '#52a291', '#957ca9', '#c39a56', '#c76c72']
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 8, 'xtick.labelsize': 6.3,
    'ytick.labelsize': 6.3, 'legend.fontsize': 6.3, 'pdf.fonttype': 42,
    'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .6, 'xtick.major.width': .5, 'ytick.major.width': .5,
    'xtick.major.size': 2.4, 'ytick.major.size': 2.4, 'text.color': INK,
    'axes.labelcolor': INK, 'axes.edgecolor': INK, 'legend.frameon': False,
    'lines.markeredgewidth': 0})


NAMES = {'D0':'GRAFT-D0','ordinary':'Plain ridge','ridge':'Gated ridge',
         'ridge_remix':'GRAFT-Adapt','kernel':'Gated kernel'}
FAMILIES = [('r','Pearson r ↑',BLUE,'Blues_r'),
            ('mse','MSE ↓','#c6a052','YlOrBr_r'),
            ('mae','MAE ↓','#b48b61','YlOrBr_r')]


def draw_ranking(output):
    raw=pd.read_csv(DATA/'primary_metric_summary.csv')
    selection=json.loads((DATA/'ranking_contract.json').read_text())
    families=[family for family in FAMILIES if family[0] in selection['families']]
    windows=selection['windows'];metrics=selection['metrics']
    f=plt.figure(figsize=(7.2,7.25))
    norm_rows, rank_rows, overall_rows = [], [], []
    for ci,(cohort,cells) in enumerate([('LINCS',INTERNAL)]):
        d=raw[(raw.cohort==cohort)&raw.metric.isin(metrics)]
        methods=['D0','ordinary','ridge','ridge_remix','kernel']
        assert len(d)==5*5*len(metrics)
        means=d.groupby(['method','metric'])['mean'].mean().unstack().loc[methods,metrics]
        ranks=means.copy()
        for m in metrics:ranks[m]=means[m].rank(ascending=m.startswith(('mse','rmse','mae')),method='average')
        overall=pd.DataFrame({'mean_rank':ranks.mean(axis=1),'first_place_count':ranks.eq(1).sum(axis=1),
                              'r20':means.r20})
        order=overall.sort_values(['mean_rank','r20'],ascending=[True,False]).index.tolist()
        y=.588-ci*.346
        f.text(.045,y+.276,chr(97+ci),size=11,weight='bold')
        f.text(.08,y+.276,cohort+' · five-cell macro comparison',size=8,weight='bold')
        ax=f.add_axes([.033,y,.938,.252]);ax.set(xlim=(0,174),ylim=(5,-2.15));ax.axis('off')
        def rect(x,y,w,h,col,ec='none',lw=.4):
            ax.add_patch(Rectangle((x,y),w,h,fc=col,ec=ec,lw=lw))
        rect(0,-2,29,.76,'#737982')
        ax.text(14.5,-1.62,'Method',size=7,weight='bold',ha='center',va='center',color='white')
        start=31; group_width=(117.2-1.3*(len(families)-1))/len(families);step=group_width/len(windows)
        for gi,(prefix,title,color,cmap) in enumerate(families):
            x=start+gi*(group_width+1.3)
            rect(x,-2,group_width,.76,color)
            ax.text(x+group_width/2,-1.62,title,size=6.3,weight='bold',ha='center',va='center',color='white')
            for ki,k in enumerate(windows):
                ax.text(x+ki*step+step/2,-.77,f'Top{k}',size=6,ha='center',va='center')
        overall_x=150.8
        rect(overall_x,-2,23,.76,GREEN)
        ax.text(overall_x+11.5,-1.62,'Overall',size=7,weight='bold',ha='center',va='center',color='white')
        ax.text(overall_x+7.6,-.77,'Rank ↓',size=6,ha='center',va='center')
        ax.text(170,-.77,'No. 1',size=6,ha='center',va='center')
        for i,method in enumerate(order):
            if i%2==0:rect(0,i-.43,174,.86,'#eef0f2')
            if method=='ridge_remix':rect(0,i-.43,.7,.86,GREEN)
            ax.text(1.5,i,NAMES[method],size=6.5,va='center',weight='bold' if method=='ridge_remix' else 'normal')
            for gi,(prefix,title,color,cmap) in enumerate(families):
                x0=start+gi*(group_width+1.3)
                for ki,k in enumerate(windows):
                    metric=f'{prefix}{k}';all_values=d.loc[d.metric==metric,'mean']
                    lo,hi=all_values.min(),all_values.max()
                    values=d[(d.metric==metric)&(d.method==method)].set_index('cell').loc[cells,'mean'].to_numpy()
                    normalized=(values-lo)/(hi-lo) if hi>lo else np.full(5,.5)
                    if prefix in ['mse','rmse','mae']:normalized=1-normalized
                    assert np.isclose(values.mean(),means.loc[method,metric],atol=1e-12)
                    rank=float(ranks.loc[method,metric]);col=mpl.colormaps[cmap](.15+.76*(rank-1)/4)
                    x=x0+ki*step+1.15;w=step-1.55
                    rect(x,i-.27,w*normalized.mean(),.54,col,INK)
                    ax.scatter(x+w*normalized,i+np.linspace(-.16,.16,5),s=2.2,c=INK,alpha=.82,linewidths=0,zorder=4)
                    if rank<=3:ax.text(x0+ki*step+.5,i,f'{rank:g}',size=6,color='#b64d4d',ha='center',va='center')
                    rank_rows.append(dict(cohort=cohort,method=method,metric=metric,raw_macro=means.loc[method,metric],rank=rank))
                    for cell,v,n in zip(cells,values,normalized):
                        norm_rows.append(dict(cohort=cohort,cell=cell,method=method,metric=metric,raw=v,normalized=n))
            mr=float(overall.loc[method,'mean_rank']);col=mpl.colormaps['Greens_r'](.15+.76*(mr-1)/4)
            rect(overall_x+1,i-.31,13.2,.62,col,INK)
            lum=np.dot(col[:3],[.21,.72,.07])
            ax.text(overall_x+7.6,i,f'{mr:.2f}',size=6.5,ha='center',va='center',color='white' if lum<.56 else INK)
            ax.text(170,i,str(int(overall.loc[method,'first_place_count'])),size=6.5,ha='center',va='center')
            overall_rows.append(dict(cohort=cohort,method=method,n_metrics=len(metrics),mean_rank=mr,
                                     first_place_count=int(overall.loc[method,'first_place_count'])))
        for x in [29.8,149.8]:ax.plot([x,x],[-.4,4.45],c='#d4d8dc',lw=.45)
    dest=output/'source_data';dest.mkdir(exist_ok=True)
    pd.DataFrame(norm_rows).to_csv(dest/'F4b_normalized_cell_metrics.csv',index=False)
    pd.DataFrame(rank_rows).to_csv(dest/'F4b_metric_values_and_ranks.csv',index=False)
    pd.DataFrame(overall_rows).to_csv(dest/'F4b_overall_ranking.csv',index=False)
    export_native(f,output,'F4b')
    return overall_rows

def draw_budget(output):
    budget=pd.read_csv(DATA/'budget_summary.csv')
    colors=dict(zip(INTERNAL,['#527caf','#52a291','#957ca9','#c39a56','#c76c72']))
    f,ax=plt.subplots(figsize=(3.6,2.6))
    for cell in INTERNAL:
        q=budget[(budget.cell==cell)&(budget.k<=15)].sort_values('k')
        ax.plot(q.k,q.gain,c=colors[cell],marker='o',ms=3,lw=1,label=cell)
        ax.fill_between(q.k,q.draw_lo,q.draw_hi,color=colors[cell],alpha=.12,lw=0)
    ax.axhline(0,c='#aaa',lw=.6,ls='--')
    ax.set(xticks=[0,1,2,5,10,15],xlabel='Anchor count, k',ylabel='Δ Top20 r')
    ax.legend(fontsize=6,ncol=3,loc='upper left');ax.set_title('Anchor-count response',loc='left',weight='bold')
    for ext in ['pdf','svg','png']:f.savefig(output/('F4c.'+ext),bbox_inches='tight',dpi=350)
    plt.close(f)
    budget.to_csv(output/'source_data/F4c_plotted_budget_values.csv',index=False)
    return budget

def zoom_x(ax, lower, upper, left_pad=.10, right_pad=.12, ticks=4):
    """Crop a performance axis without excluding any supplied data/intervals."""
    span=max(float(upper)-float(lower),1e-4)
    lo=float(lower)-span*left_pad;hi=float(upper)+span*right_pad
    if lower>0:lo=max(lo,float(lower)*.5)
    ax.set_xlim(lo,hi)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=ticks))
    if lo>0:
        # Paired diagonal marks flag a nonzero bar origin at the lower-left.
        for x in [-.004,.012]:
            ax.plot([x-.006,x+.006],[-.012,.012],transform=ax.transAxes,
                    color=INK,lw=.7,clip_on=False,zorder=20)
        ax.set_xlabel(ax.get_xlabel()+' (zoomed)')
    ax._zoom_extent=dict(data_lower=float(lower),data_upper=float(upper),axis_lower=lo,axis_upper=hi,
                         zero_excluded=lo>0)
    return lo,hi

def draw_mechanism(output):
    d = pd.read_csv(DATA/'mechanism_paired_tests.csv')
    assert (d.p_holm > 0).all(), 'Do not display numerical underflow as P=0'
    d['formatted_p'] = [f'{p:.1e}' if p < .001 else f'{p:.3f}' for p in d.p_holm]
    d['sparse_annotation'] = (d.moa_relation == 'Annotation unavailable') & (d.n_drugs < 10)
    d['display_p'] = d.formatted_p + np.where(d.sparse_annotation, '†', '')
    d['comparison'] = 'GRAFT-Adapt vs GRAFT-D0 within the same MoA stratum'
    dest = output/'source_data'; dest.mkdir(exist_ok=True)
    d.to_csv(dest/'F4f_mechanism_statistics.csv', index=False)
    f = plt.figure(figsize=(7.2,8.6))
    f.text(.045,.94,'Ten anchors | Paired response gain relative to GRAFT-D0',size=8,color='#59616b')
    for row, cells in enumerate([INTERNAL, EXTERNAL]):
        ax = f.add_axes([.12+row*.46, .57, .23, .30]); ax.set_title('LINCS' if row == 0 else 'External',loc='left',weight='bold',pad=10)
        # Statistics share the bar's data y coordinate, preserving row alignment.
        for x, title in [(1.10, 'n')]:
            ax.text(x, 1.035, title, transform=ax.transAxes, ha='right',
                    va='bottom', fontsize=6.3, weight='bold')
        ax.text(1.43,1.035,'P (Holm)\nvs D0',transform=ax.transAxes,ha='center',va='bottom',fontsize=6.3,weight='bold')
        group_limits=[]
        for j, (group, col) in enumerate(zip(['Shared MoA', 'No shared annotated MoA', 'Annotation unavailable'], [GREEN, BLUE, GRAY])):
            for i, cell in enumerate(cells):
                s = d[(d.cell == cell) & (d.moa_relation == group)]
                if len(s) and s.iloc[0].n_drugs:
                    s = s.iloc[0];y=i+(j-1)*.24;group_limits.extend([s.lo,s.hi])
                    ax.barh(y,s['mean'],height=.20,color=col,edgecolor=INK,lw=.4,zorder=2)
                    ax.errorbar(s['mean'],y,xerr=[[s['mean']-s.lo],[s.hi-s['mean']]],
                                color=INK,fmt='o',ms=1.9,lw=.6,capsize=1.3,zorder=4)
                    for x, text in [(1.10, str(int(s.n_drugs)))]:
                        ax.text(x, y, text, transform=ax.get_yaxis_transform(),
                                ha='right', va='center', fontsize=6.0, clip_on=False)
                    ax.text(1.43,y,s.display_p,transform=ax.get_yaxis_transform(),
                            ha='center',va='center',fontsize=6.0,clip_on=False)
        for boundary in np.arange(.5, 4.6, 1):
            ax.plot([0, 1.73], [boundary, boundary], transform=ax.get_yaxis_transform(),
                    color='#e4e7eb', lw=.4, clip_on=False, zorder=0)
        ax.axvline(0, c=GRAY, lw=.6, ls='--'); ax.set(yticks=range(5), yticklabels=[c.replace('_L1000','') for c in cells], ylim=(4.6, -.6), xlabel='Δr20 (drug 95% CI)')
        zoom_x(ax,min(group_limits),max(group_limits))
    legend=f.legend(handles=[Patch(facecolor=c,edgecolor=INK,lw=.4,label=g) for c,g in zip([GREEN,BLUE,GRAY],['Shared MoA','No shared annotated MoA','Annotation unavailable'])],loc='center',bbox_to_anchor=(.54,.50),ncol=3)
    export_native(f,output,'F4f',legend)
    return d


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True);(a.output/'source_data').mkdir(exist_ok=True)
    n=verify_inputs();ranks=draw_ranking(a.output);budget=draw_budget(a.output);mechanism=draw_mechanism(a.output)
    top=next(r for r in ranks if r['method']=='ridge_remix')
    assert top['mean_rank']==1 and top['first_place_count']==6
    gain=float(budget[(budget.cell=='A375')&(budget.k==10)].gain.iloc[0])
    assert abs(gain-0.01656919477283215)<1e-12
    (a.output/'verification.json').write_text(json.dumps({'panels':['F4b','F4c','F4f'],'immutable_input_files_verified':n,'GRAFT_Adapt_mean_rank':top['mean_rank'],'first_place_count':top['first_place_count'],'A375_k10_gain':gain,'ranking_metric_set_exploratory':True,'mechanism_strata_rows':len(mechanism),'mechanism_P_values':'Original Holm-adjusted paired Wilcoxon values over 30 cell-stratum tests'},indent=2)+'\n')
if __name__=='__main__':main()
