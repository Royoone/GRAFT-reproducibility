"""Render only the selected quantitative panel in this group."""
from pathlib import Path
import os,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
env=os.environ.copy();env['GRAFT_FIGURE_OUTPUT']=str(ROOT/'results/figures/figure2')
env.setdefault('MPLCONFIGDIR',str(ROOT/'results/cache/matplotlib'))
subprocess.run([sys.executable,str(HERE/'code/render.py')],cwd=ROOT,env=env,check=True)
