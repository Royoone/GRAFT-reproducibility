"""Reproduce the key iPSC correspondence panels, F7f, F7g and F7h."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-audit',action='store_true')
    parser.add_argument('--output',type=Path,default=REPO/'results/figures/figure7',
                        help='Output directory for F7f/g/h and numerical audits')
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    env = os.environ.copy()
    env.setdefault('MPLCONFIGDIR',str(output/'.mplconfig'))
    if not args.skip_audit:
        command = [sys.executable,str(REPO/'experiments/ipsc/analyze_correspondence.py'),
                   '--output',str(output/'reanalysis')]
        prepared = REPO/'data/processed/ipsc/graft_input_976.npz'
        bundle = REPO/'data/model_bundle'
        catalog = REPO/'data/processed/ipsc/record_catalog.csv'
        if prepared.is_file() and (bundle/'reference.npz').is_file() and catalog.is_file():
            command += ['--input',str(prepared),'--bundle',str(bundle),'--catalog',str(catalog)]
        subprocess.run(command,check=True,env=env)
    subprocess.run([sys.executable,str(ROOT/'code/draw_response_correspondence.py'),
                    '--output',str(output)],check=True,env=env)
    for name in ['F7f','F7g','F7h']:
        if not (output/(name+'.pdf')).is_file():
            raise RuntimeError('Missing panel: '+name)
    manifest = {'figure':7,'panels':{'F7f':'F7f.pdf','F7g':'F7g.pdf','F7h':'F7h.pdf'},
                'scope':'Drug-level, MoA and four-case correspondence in the fixed iPSC analysis',
                'scientific_inputs':'source_data/; source_provenance.json'}
    (output/'panel_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Figure 7: F7f, F7g and F7h generated in',output)


if __name__ == '__main__':
    main()
