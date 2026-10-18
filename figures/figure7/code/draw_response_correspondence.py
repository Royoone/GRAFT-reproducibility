"""Render the manuscript's drug-level and MoA correspondence panels, F7f/g/h."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'source_data'
DEFAULT_OUTPUT = ROOT.parents[1] / 'results/figures/figure7'
INK = '#283139'
MUTED = '#68747c'
LINES = ['eipl_3','fiaj_3','iudw_1','jejf_3','kolf_3','oikd_5','paab_3','pipw_4','tolg_6','zapk_3']
COL = ['#567fab','#96b3d2','#389284','#86bdb3','#baa2c8','#bc945a','#849b54','#c56d78','#d2b394','#aa775a']
TC = {'EZH2':'#617faa','HDAC3':'#aa775f'}


def main(output):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA / 'all235_evaluation_and_retention.csv')
    retained = frame[frame.v2_retained].reset_index(drop=True)
    positive = retained.rho.gt(0)
    assert (len(frame), len(retained), int(positive.sum())) == (235, 191, 158)
    edges = np.arange(-1, 1.01, .1)
    hist = pd.DataFrame({'bin_left':edges[:-1], 'bin_right':edges[1:],
                         'positive':np.histogram(retained.loc[positive, 'rho'], edges)[0],
                         'nonpositive':np.histogram(retained.loc[~positive, 'rho'], edges)[0]})
    moa = pd.read_csv(DATA / 'moa/plot_moa8_consistent_members_common_lines.csv')
    mem = pd.read_csv(DATA / 'moa/plot_moa8_member_drugs_common_lines.csv')
    assert len(moa) == 8 and len(mem) == 104
    assert set(moa.moa) == set(mem.moa)
    assert np.array_equal(mem.groupby('moa').size().reindex(moa.moa).to_numpy(), moa.n_drugs.to_numpy())
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
                         'font.size':6,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                         'text.color':INK,'axes.labelcolor':INK,'xtick.labelsize':5.4,'ytick.labelsize':5.4})
    W, H = 240, 94
    fig = plt.figure(figsize=(W/25.4, H/25.4))

    def ax(x,y,w,h):
        return fig.add_axes([x/W,1-(y+h)/H,w/W,h/H])

    def text(x,y,t,size=6,**kw):
        kw.setdefault('va','top')
        return fig.text(x/W,1-y/H,t,fontsize=size,**kw)

    def header(x,w,letter,title):
        text(x,1,letter,12,weight='bold')
        text(x+7,2,title,7.1,weight='bold')
        fig.lines.append(Line2D([(x+7)/W,(x+w)/W],[1-9/H]*2,
                                transform=fig.transFigure,color='#acb7c0',lw=.5))

    def tidy(a):
        a.spines[['top','right']].set_visible(False)
        for sp in a.spines.values():
            sp.set_linewidth(.45)
            sp.set_color('#7f8992')
        a.tick_params(length=2,width=.45,pad=2,labelsize=5.4)

    header(1,41,'f','Drug-level agreement')
    a=ax(9,20,32,42)
    centres=(hist.bin_left+hist.bin_right)/2
    for col,color in [('nonpositive','#bac3cb'),('positive','#617faa')]:
     a.bar(centres,hist[col],width=.093,color=color,lw=0)
    a.axvline(0,color='#79838b',ls=(0,(2,2)),lw=.55)
    a.set(xlim=(-1.03,1.03),xticks=[-1,0,1],xlabel='Spearman ρ',ylabel='Drug count')
    a.xaxis.label.set_size(6);a.yaxis.label.set_size(6);tidy(a)
    text(9,71,f'{int(retained.positive.sum())}/{len(retained)} ({100*retained.positive.mean():.2f}%) positive\nchemical entities',5.6,color=MUTED,linespacing=1.4)

    header(49,107,'g','MoA and compound agreement')
    labels={'Serotonin receptor agonist':'5-HT agonist','Dopamine antagonist':'Dopamine antagonist','Topoisomerase':'Topoisomerase','HDAC':'HDAC','Serotonin antagonist':'5-HT antagonist','Dopamine receptor agonist':'Dopamine agonist','CDK':'CDK','Thymidylate synthase inhibitor':'TYMS inhibitor'}
    a=ax(57,20,98,42);a.set(xlim=(-.6,7.6),ylim=(-1.03,1.28),yticks=[-1,-.5,0,.5,1],ylabel='Spearman ρ')
    a.yaxis.label.set_size(6)
    a.axhspan(-1.03,0,color='#f6f7f8',zorder=0);a.axhline(0,color='#ccd2d7',lw=.55,zorder=1)
    for k,row in enumerate(moa.itertuples()):
     sub=mem[mem.moa.eq(row.moa)]
     a.scatter(k+sub.jitter*1.8,sub.rho,s=6,c='#aeb8c0',alpha=.82,lw=0,zorder=2)
     a.plot([k,k],[0,row.rho],color='#849bb7',lw=1.0,zorder=3)
     a.scatter(k,row.rho,s=22,c='#567ba8',ec='white',lw=.4,zorder=4)
     a.text(k,1.19,f'{row.rho:.2f}\nP={row.p_exact:.3f}',fontsize=4.8,ha='center',va='top',linespacing=1.3)
     for rr in sub[sub.drug_name.isin(['romidepsin','dacinostat'])].itertuples():
      a.scatter(k+rr.jitter*1.8,rr.rho,s=9,c='#b58164',lw=.2,ec='white',zorder=5)
    a.set_xticks(range(8),[f'{labels[r.moa]}\n{r.n_drugs} drugs · {r.n_lines} lines' for r in moa.itertuples()],rotation=90,fontsize=5.3)
    tidy(a)
    leg=ax(58,12,98,5);leg.axis('off');leg.set(xlim=(0,1),ylim=(0,1))
    for x,sz,col,txt in [(.02,20,'#567ba8','MoA-pooled'),(.38,6,'#aeb8c0','Individual drug'),(.77,9,'#b58164','Romi. / Daci.')]:
     leg.scatter(x,.5,s=sz,c=col,lw=0);leg.text(x+.035,.5,txt,fontsize=5.2,va='center')


    header(163,75,'h','Representative drug–target pairs')
    points=pd.read_csv(DATA/'examples/recommended_four_plot_points.csv')
    stats=pd.read_csv(DATA/'examples/recommended_four_drugs.csv').set_index('drug_name')
    for ii,drug in enumerate(['CPI-360','EI1','romidepsin','dacinostat']):
     x=174+(ii%2)*33;y=19+(ii//2)*29
     target='EZH2' if ii<2 else 'HDAC3'
     text(x+13.5,y-4,f'{drug} · {target}',5.7,ha='center',color=TC[target],weight='bold')
     a=ax(x,y,27,20)
     pp=points[points.drug_name.eq(drug)].set_index('clone').loc[LINES]
     a.plot([1,10],[1,10],ls=(0,(3,3)),lw=.4,color='#c6cdd4',zorder=0)
     for jj,line in enumerate(LINES):
      rr=pp.loc[line];a.scatter(rr.predicted_rank,rr.measured_rank,s=9,c=COL[jj],ec='white',lw=.2)
     a.set(xlim=(.5,10.5),ylim=(10.5,.5),xticks=[1,5,10],yticks=[1,5,10]);tidy(a)
     rr=stats.loc[drug]
     a.text(.05,.05,f'ρ = {rr.rho:.2f}\nP = {rr.p_exact:.3f}',transform=a.transAxes,fontsize=5,va='bottom',linespacing=1.2)
    text(165,43.5,'Measured target-response rank',5.8,rotation=90,ha='center',va='center')
    text(204,73,'Predicted response RMS rank',5.8,ha='center')
    cl=ax(173,79,65,9);cl.axis('off');cl.set(xlim=(0,5),ylim=(2,0))
    for j,col in enumerate(COL):
     x=j%5+.1;y=j//5+.5
     cl.scatter(x,y,s=12,c=col,lw=0)
     cl.text(x+.18,y,f'C{j+1}',fontsize=5.8,va='center',color=col)
    assert len(points)==40 and len(stats)==4
    assert set(stats.index)=={'CPI-360','EI1','romidepsin','dacinostat'}
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for label in fig.findobj(mpl.text.Text):
        if label.get_visible() and label.get_text():
            b = label.get_window_extent(renderer)
            if b.x0 < -.6 or b.y0 < -.6 or b.x1 > fig.bbox.width+.6 or b.y1 > fig.bbox.height+.6:
                outside.append(label.get_text())
    if outside:
        raise ValueError('Text outside correspondence panels: ' + str(outside))
    buffer = io.BytesIO()
    fig.savefig(buffer, format='pdf')
    plt.close(fig)
    panels = {'F7f':(0,0,47,94), 'F7g':(48,0,160,94), 'F7h':(161,0,240,94)}
    exports = {}
    with fitz.open(stream=buffer.getvalue(),filetype='pdf') as drawing:
        for name, box in panels.items():
            crop = fitz.Rect(*(v*72/25.4 for v in box))
            panel = fitz.open()
            page = panel.new_page(width=crop.width,height=crop.height)
            page.show_pdf_page(page.rect,drawing,0,clip=crop)
            path = output / (name+'.pdf')
            panel.save(path)
            (output / (name+'.svg')).write_text(page.get_svg_image(text_as_path=False))
            page.get_pixmap(matrix=fitz.Matrix(1.7,1.7),alpha=False).save(output / (name+'.png'))
            exports[name] = {'pdf_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                             'width_mm':box[2]-box[0], 'height_mm':box[3]-box[1],
                             'raster_objects':len(page.get_images())}
            panel.close()
    hist.to_csv(output / 'F7f_histogram.csv',index=False)
    checks = {'status':'pass','panels':['F7f','F7g','F7h'],'n_evaluable_entities':len(frame),
              'retained_entities':len(retained),'retained_positive':int(positive.sum()),
              'n_moas':len(moa),'n_class_member_drugs':len(mem),
              'case_drugs':['CPI-360','EI1','romidepsin','dacinostat'],'n_case_points':len(points),
              'p_values':'exact two-sided, unadjusted','MoA_common_lines':True,
              'outcome_conditioned_retention':True,'text_outside_page':outside,'exports':exports}
    (output / 'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,
                        default=Path(os.environ.get('GRAFT_FIGURE_OUTPUT',str(DEFAULT_OUTPUT))))
    args = parser.parse_args()
    os.environ.setdefault('MPLCONFIGDIR',str(args.output/'.mplconfig'))
    main(args.output)
