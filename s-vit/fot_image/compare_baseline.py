"""Plaintext original FoT with independent validation; original training code unchanged."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F

from training.core import ZOConfig, PlainAdam, basis, direction
from training.data import PartyData
from training.model import fixture, load_backbone, PlainOps, forward


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture', action='store_true')
    p.add_argument('--checkpoint'); p.add_argument('--fixed-student'); p.add_argument('--mpcvit-root')
    p.add_argument('--dataset-roots', nargs=3); p.add_argument('--manifests', nargs=3)
    p.add_argument('--eval-root'); p.add_argument('--eval-manifest')
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--steps', type=int, default=100); p.add_argument('--eval-every', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=1); p.add_argument('--fixed-batch', action='store_true')
    p.add_argument('--rank', type=int, default=4); p.add_argument('--sigma', type=float, default=.1)
    p.add_argument('--lr', type=float, default=.03); p.add_argument('--epsilon', type=float, default=.01)
    p.add_argument('--seed', type=int, default=20261107); p.add_argument('--output', required=True)
    a = p.parse_args()
    if min(a.steps,a.eval_every,a.batch_size)<1: p.error('positive steps, evaluation interval and batch size required')
    if not a.fixture and not all((a.checkpoint,a.fixed_student,a.mpcvit_root,a.dataset_roots,a.manifests,a.eval_root,a.eval_manifest)):
        p.error('real experiment requires backbone, train and independent evaluation manifests')
    if a.device=='cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable')
    out = Path(a.output).resolve(); out.mkdir(parents=True,exist_ok=False,mode=0o700)
    (out/'options.json').write_text(json.dumps(vars(a),indent=2)+'\n')
    torch.set_num_threads(1)
    spec, params = fixture() if a.fixture else load_backbone(a.checkpoint,a.mpcvit_root,a.fixed_student)
    device = torch.device(a.device)
    ops = PlainOps({k:v.to(device) for k,v in params.items()})
    parties = [PartyData(i,spec,None if a.fixture else a.dataset_roots[i],None if a.fixture else a.manifests[i]) for i in range(3)]
    validation = parties if a.fixture else [PartyData(0,spec,a.eval_root,a.eval_manifest)]
    cfg = ZOConfig(rank=a.rank,sigma=a.sigma,lr=a.lr,epsilon=a.epsilon,seed=a.seed).validate()
    adam = PlainAdam(cfg)
    b = basis(spec['patches']*spec['dim'],cfg).to(device)
    def eval_model():
        loss_sum = correct = count = 0
        prompt = b.matmul(adam.z.to(device))
        for ds in validation:
            for i in range(len(ds)):
                x,y = ds.batch(i,1)
                logits = forward(spec,ops,x.to(device),prompt)
                loss_sum += F.cross_entropy(logits,y.to(device),reduction='sum').item()
                correct += (logits.argmax(-1).cpu()==y).sum().item()
                count += y.numel()
        return {'loss':loss_sum/count,'top1':correct/count,'samples':count}
    initial = eval_model()
    print(json.dumps({'event':'evaluation','step':0,**initial}),flush=True)
    rows = []
    for step in range(1,a.steps+1):
        owner = (step-1)%3
        x,y = parties[owner].batch((step-1)//3,a.batch_size,a.fixed_batch)
        x,y = x.to(device),y.to(device)
        u = direction(step,cfg).to(device)
        z = adam.z.to(device)
        plus = F.cross_entropy(forward(spec,ops,x,b.matmul(z+a.sigma*u)),y).item()
        minus = F.cross_entropy(forward(spec,ops,x,b.matmul(z-a.sigma*u)),y).item()
        adam.update(plus-minus,u)
        row = {'step':step,'owner':owner,'batch_size':a.batch_size}
        if step%a.eval_every==0 or step==a.steps:
            row['evaluation'] = eval_model()
            torch.save({'z':adam.z,'m':adam.m,'v':adam.v,'step':step,'config':vars(a)},out/f'step-{step:06d}.pt')
        rows.append(row)
        print(json.dumps({'event':'step_complete',**row}),flush=True)
    result = {'status':'completed','backend':'plaintext','method':'original_fot_adam',
              'initial':initial,'steps':rows,'batch_size':a.batch_size,
              'scope':'public fixture' if a.fixture else 'explicit independent manifest'}
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'event':'complete','report':str(out/'report.json')}),flush=True)


if __name__=='__main__': main()
