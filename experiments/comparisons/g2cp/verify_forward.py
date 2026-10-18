"""Compare the adopted vectorized G2CP computation with the author forward."""
from pathlib import Path
from types import SimpleNamespace
import argparse,copy,json,sys
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[3];sys.path.insert(0,str(REPO))
from experiments.comparisons.g2cp.author.model import GEARS_Model
from experiments.comparisons.g2cp import helpers as H
from experiments.comparisons.g2cp.fast_forward import enable_fast_forward,fast_batch_for


def verify():
    torch.set_num_threads(2);H.seed_all(52);g=32;b=4
    edge=torch.stack([torch.arange(g),torch.roll(torch.arange(g),1)])
    args=SimpleNamespace(device='cpu',graph_batch_mode='corrected')
    original=H.make_model(args,GEARS_Model,edge,torch.ones(g),np.asarray([f'g{i}' for i in range(g)]),8)
    fast=copy.deepcopy(original);enable_fast_forward(fast)
    x=np.random.default_rng(9).normal(size=(b,256)).astype(np.float32)
    source=H.batch_for(x,g,'cpu');packed=fast_batch_for(x,g,'cpu')
    report={}
    for training in [False,True]:
        a=copy.deepcopy(original);c=copy.deepcopy(fast);a.train(training);c.train(training)
        y=a(source);z=c(packed);label='training' if training else 'evaluation'
        report[label+'_max_output_error']=float((y-z).abs().max().detach())
        y.square().mean().backward();z.square().mean().backward();errors=[]
        for (name,pa),(other,pb) in zip(a.named_parameters(),c.named_parameters()):
            assert name==other
            if pa.grad is not None and pb.grad is not None:errors.append(float((pa.grad-pb.grad).abs().max()))
            else:assert (pa.grad is None)==(pb.grad is None)
        report[label+'_max_parameter_gradient_error']=max(errors)
        stats=[float((a.state_dict()[name]-c.state_dict()[name]).abs().max()) for name in a.state_dict() if 'running_' in name]
        report[label+'_max_BatchNorm_stat_error']=max(stats)
    report['status']='pass' if all(value<3e-4 for value in report.values()) else 'fail'
    assert report['status']=='pass',report
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=REPO/'results/comparisons/G2CP/forward_verification.json');a=p.parse_args()
    report=verify();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
