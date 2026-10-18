"""Redraw the two retained quantitative manuscript panels from frozen inputs."""
from pathlib import Path
import argparse,hashlib,json,os,tempfile
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"graft-figure-cache"))
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Bbox
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'source_data'
mpl.rcParams['pdf.fonttype']=42;mpl.rcParams['svg.fonttype']='none'

def export_native(f,output,name):
    for artist in list(f.texts)+list(f.legends):artist.set_visible(False)
    f.canvas.draw();renderer=f.canvas.get_renderer()
    bbox=Bbox.union([a.get_tightbbox(renderer) for a in f.axes]).transformed(f.dpi_scale_trans.inverted())
    bbox=Bbox.from_extents(bbox.x0-.12,bbox.y0-.12,bbox.x1+.12,bbox.y1+.12)
    for ext in ['pdf','svg','png']:f.savefig(output/(name+'.'+ext),bbox_inches=bbox,dpi=350,facecolor='white')
    plt.close(f)

def verify_inputs():
    manifest=json.loads((ROOT/'input_manifest.json').read_text())
    for row in manifest['files']:
        path=ROOT/row['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Immutable input changed: '+row['path'])
    return len(manifest['files'])

from collections import Counter
INK='#22282b'
MC=dict(zip(['MEK','PI3K','mTOR','HDAC','Topoisomerase','Proteasome','CDK','EGFR','Tubulin','HSP inhibition','Glucocorticoid','Estrogen agonist','Estrogen antagonist','Retinoid','Dopamine antagonist','Serotonin antagonist'],['#299d8f','#629bb9','#8d80b4','#d7788c','#df8861','#77a665','#b45e6e','#62b0af','#bfa45a','#9e7657','#bb9846','#bd7db0','#766b99','#739d94','#7892a0','#baafa0']))
MOA=list(MC)
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'lines.linewidth':1,'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2.5,'ytick.major.size':2.5,'text.color':INK,'axes.labelcolor':INK,'axes.edgecolor':INK,'savefig.facecolor':'white','legend.frameon':False})
def mapaxes(ax,U):
 lo=U[['UMAP1','UMAP2']].min().to_numpy();hi=U[['UMAP1','UMAP2']].max().to_numpy();pad=(hi-lo)*.025
 ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]);ax.set_ylim(lo[1]-pad[1],hi[1]+pad[1]);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
 for sp in ax.spines.values():sp.set_visible(False)

def heatmap_zscore(data,axis=1,key=None,exclude_diagonal=False,log=False):
 """Display-only population z-score. axis=1: within row; axis=0: within column.
 Raw and displayed matrices are exported; masked diagonal is excluded from moments.
 """
 raw=data.copy() if isinstance(data,pd.DataFrame) else pd.DataFrame(data)
 x=raw.to_numpy(dtype=float).copy()
 if log:
  if np.any(x[np.isfinite(x)]<=0):raise ValueError('Log scaling requires positive values')
  x=np.log(x)
 if exclude_diagonal:np.fill_diagonal(x,np.nan)
 mu=np.nanmean(x,axis=axis,keepdims=True);sd=np.nanstd(x,axis=axis,keepdims=True,ddof=0)
 z=np.divide(x-mu,sd,out=np.zeros_like(x),where=sd>1e-12);z[~np.isfinite(x)]=np.nan
 out=pd.DataFrame(z,index=raw.index,columns=raw.columns)
 if key:
  d=S/'heatmap_scaling';d.mkdir(exist_ok=True)
  raw.to_csv(d/(key+'_raw.csv'));out.to_csv(d/(key+'_zscore.csv'))
  (d/(key+'_scale.json')).write_text(json.dumps(dict(axis='row' if axis==1 else 'column',ddof=0,exclude_diagonal=exclude_diagonal,log_before_zscore=log,colour_limit=[-2.5,2.5],raw_shape=list(x.shape),zero_sd_policy='zero',missing='white / excluded'),indent=2))
 return out
EXTRA = {
    'cyclooxygenase inhibitor': ('COX inhibitor', '#cc7850'),
    'adrenergic receptor antagonist': ('Adrenergic antagonist', '#668fb1'),
    'acetylcholine receptor antagonist': ('Acetylcholine antagonist', '#8e6c99'),
    'adrenergic receptor agonist': ('Adrenergic agonist', '#b08c69'),
    'histamine receptor antagonist': ('Histamine antagonist', '#bd7295'),
    'phosphodiesterase inhibitor': ('PDE inhibitor', '#4a9181'),
    'glutamate receptor antagonist': ('Glutamate antagonist', '#728351'),
    'calcium channel blocker': ('Calcium channel blocker', '#ab796d'),
    'serotonin receptor agonist': ('Serotonin agonist', '#678ca5'),
    'vegfr inhibitor': ('VEGFR inhibitor', '#af9551'),
    'atpase inhibitor': ('ATPase inhibitor', '#798b73'),
    'sodium channel blocker': ('Sodium channel blocker', '#997eaa'),
    'dopamine receptor agonist': ('Dopamine agonist', '#599aa0'),
    'opioid receptor antagonist': ('Opioid antagonist', '#b16e6c'),
    'opioid receptor agonist': ('Opioid agonist', '#819a5b'),
    'pdgfr inhibitor': ('PDGFR inhibitor', '#9a825b'),
}


def plot_expanded_moa(ax, U, catalog):
    tokens = catalog.moa.fillna('').map(lambda s: set(t.strip() for t in s.split('|') if t.strip()))
    remaining = catalog.display_moa.eq('Other / unannotated')
    counts = Counter(t for ts in tokens[remaining] for t in ts)
    assert set(EXTRA) == {t for t, _ in counts.most_common(16)}
    labels = catalog.display_moa.copy()
    matched = pd.Series('', index=catalog.index)
    for term, (name, color) in EXTRA.items():
        select = labels.eq('Other / unannotated') & tokens.map(lambda ts: term in ts)
        labels.loc[select] = name
        matched.loc[select] = term
    palette = dict(MC)
    palette.update({name: color for name, color in EXTRA.values()})
    point_labels = labels.iloc[U.chemical_index.to_numpy()].to_numpy()
    colored = np.isin(point_labels, list(palette))
    gray = ~colored
    # Distinct artists and explicit zorder guarantee color remains above gray.
    background = ax.scatter(U.UMAP1[gray], U.UMAP2[gray], s=.24,
                            c='#a5adb2', alpha=.25, lw=0, rasterized=True, zorder=1)
    order = np.random.default_rng(20260926).permutation(np.flatnonzero(colored))
    foreground = ax.scatter(U.UMAP1.iloc[order], U.UMAP2.iloc[order], s=1.35,
                            c=[palette[point_labels[i]] for i in order], alpha=.82,
                            lw=0, rasterized=True, zorder=3)
    mapaxes(ax, U)
    annotations = catalog[['chemical_key', 'drug_id', 'name', 'moa', 'display_moa']].copy()
    annotations['fig5_03_display_moa'] = labels
    annotations['new_matched_term'] = matched
    annotations.to_csv(S/'fig5_03_expanded_moa_annotations.csv', index=False)
    rows=[]
    for name,color in palette.items():
        term=next((t for t,v in EXTRA.items() if v[0]==name), '')
        rows.append(dict(moa=name,color=color,n_displayed_chemicals=int(labels.eq(name).sum()),
                         n_displayed_points=int((point_labels==name).sum()),exact_added_term=term,
                         original_16=name in MC))
    pd.DataFrame(rows).to_csv(S/'fig5_03_expanded_moa_counts.csv', index=False)
    info=dict(n_classes=len(palette),n_points=len(U),n_colored_points=int(colored.sum()),
              n_gray_points=int(gray.sum()),n_colored_chemicals=int(labels.isin(palette).sum()),
              previous_colored_chemicals=int(catalog.display_moa.isin(MC).sum()),
              n_unannotated_chemicals=int(tokens.map(len).eq(0).sum()),
              background_color="#a5adb2",background_alpha=.25,background_zorder=background.get_zorder(),foreground_zorder=foreground.get_zorder(),
              gray_points_drawn_only_once=True,all_points_retained=True,
              input_coordinates_sha256=hashlib.sha256(U[['UMAP1','UMAP2']].to_numpy(dtype='float32').tobytes()).hexdigest(),
              rule='Retain existing 16 labels; add the 16 most frequent exact MoA terms among remaining entries. Ties follow original annotation order. Multi-label display priority: original classes first, then listed added terms by descending initial count. Full annotations and all statistical memberships unchanged.')
    assert len(background.get_offsets())+len(foreground.get_offsets())==len(U)
    (S/'fig5_03_expanded_moa_audit.json').write_text(json.dumps(info,indent=2))
    return palette, info
HCM=LinearSegmentedColormap.from_list('heatmap_blue_white_red',['#3f7f93','#ffffff','#da3b46'],N=257)
HCM.set_bad('#ffffff')

def draw_umap(output):
    global S
    S=output/'source_data'
    U=pd.read_csv(DATA/'umap_coordinates.csv.gz');catalog=pd.read_csv(DATA/'chemical_catalog.csv')
    f=plt.figure(figsize=(7.2,9.2));ax=f.add_axes([.025,.42,.585,.48])
    palette,info=plot_expanded_moa(ax,U,catalog)
    legend=f.add_axes([.625,.435,.355,.475]);legend.set_xlim(0,1);legend.set_ylim(-.4,34.2);legend.axis('off')
    legend.text(.055,33.5,'MoA',fontsize=6.5,weight='bold',va='center');legend.text(.99,33.5,'Points / drugs',fontsize=6.5,weight='bold',ha='right',va='center')
    counts=pd.read_csv(S/'fig5_03_expanded_moa_counts.csv').set_index('moa')
    for i,(name,color) in enumerate(palette.items()):
        y=32-i;row=counts.loc[name];legend.plot(.015,y,'o',ms=3.4,c=color,clip_on=False)
        legend.text(.055,y,name,fontsize=6.0,va='center');legend.text(.99,y,f"{int(row.n_displayed_points):,} / {int(row.n_displayed_chemicals):,}",fontsize=6.0,va='center',ha='right')
    legend.plot(.015,0,'o',ms=3.4,c='#a5adb2',clip_on=False);legend.text(.055,0,'Other / unannotated',fontsize=6.0,va='center')
    legend.text(.99,0,f"{info['n_gray_points']:,} / {len(catalog)-info['n_colored_chemicals']:,}",fontsize=6.0,va='center',ha='right')
    export_native(f,output,'F5b');return info

def draw_fingerprint(output):
    global S
    S=output/'source_data'
    raw=pd.read_csv(DATA/'disjoint_class_correlations.csv')
    C=raw.pivot(index='moa1',columns='moa2',values='r').loc[MOA,MOA]
    f=plt.figure(figsize=(7.2,7.6));ax=f.add_axes([.24,.145,.38,.255])
    ax.set_title('Relative fingerprint similarity\nPI3K–mTOR: raw r = '+f'{C.loc["PI3K","mTOR"]:.3f}',loc='left',fontweight='bold',fontsize=7.8,pad=9)
    z=heatmap_zscore(C,axis=1,key='Fig5_07b_similarity',exclude_diagonal=True)
    im=ax.imshow(z,cmap=HCM,vmin=-2.5,vmax=2.5)
    ax.set_yticks(range(16),[f'{i+1}  {name}' for i,name in enumerate(MOA)],fontsize=6);ax.set_xticks(range(16),range(1,17),fontsize=6);ax.tick_params(length=0)
    for i,j in [(1,2),(2,1)]:ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,ec=INK,lw=.8))
    cax=f.add_axes([.29,.08,.27,.01]);cb=f.colorbar(im,cax=cax,orientation='horizontal',label='Pearson r: row z-score (off-diagonal)',ticks=[-2,0,2],extend='both');cb.ax.xaxis.set_label_position('top')
    export_native(f,output,'F5k');return float(C.loc['PI3K','mTOR'])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True);(a.output/'source_data').mkdir(exist_ok=True)
    n=verify_inputs();info=draw_umap(a.output);r=draw_fingerprint(a.output)
    assert info['n_points']==100815 and abs(r-0.9661206340765746)<1e-12
    (a.output/'verification.json').write_text(json.dumps({'panels':['F5b','F5k'],'immutable_input_files_verified':n,'n_points':info['n_points'],'PI3K_mTOR_raw_r':r,'embedding_refitted':False},indent=2)+'\n')
if __name__=='__main__':main()
