"""Evaluate an authorized PLAINTEXT prompt on an explicit independent manifest."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from .core import ZOConfig, basis
from .data import PartyData
from .model import load_backbone, fixture, PlainOps, forward


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompt', type=Path, required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--fixed-student')
    p.add_argument('--mpcvit-root')
    p.add_argument('--dataset-root')
    p.add_argument('--manifest')
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    artifact = torch.load(args.prompt, map_location='cpu', weights_only=True)
    if artifact.get('format') != 'zo_prompt_plain_v1':
        p.error('only plaintext prompt artifacts supported; secret shares must not be silently reconstructed')
    if args.batch_size < 1 or args.output.exists():
        p.error('positive batch size and new output file required')
    spec, params = fixture() if artifact['spec']['fixture'] else load_backbone(args.checkpoint,args.mpcvit_root,args.fixed_student)
    if spec != artifact['spec']:
        raise ValueError('backbone/config differs from training artifact')
    cfg = ZOConfig(**artifact['config']).validate()
    device = torch.device(args.device)
    ops = PlainOps({k:v.to(device) for k,v in params.items()})
    prompt = basis(spec['patches']*spec['dim'],cfg).matmul(artifact['z']).to(device)
    data = PartyData(0,spec,args.dataset_root,args.manifest)
    loss, correct, count = 0.,0,0
    for start in range(0,len(data),args.batch_size):
        # PartyData's indexing API uses step*size. Read the final partial batch
        # as individual rows to avoid wraparound/duplicate evaluation images.
        rows=[data.batch(i,1) for i in range(start,min(start+args.batch_size,len(data)))]
        x=torch.cat([a for a,_ in rows]).to(device)
        y=torch.cat([b for _,b in rows]).to(device)
        logits=forward(spec,ops,x,prompt)
        loss += F.cross_entropy(logits,y,reduction='sum').item()
        correct += (logits.argmax(-1)==y).sum().item()
        count += y.numel()
    report={'samples':count,'ce':loss/count,'top1':correct/count,
            'scope':'public fixture diagnostic' if spec['fixture'] else 'plaintext explicit manifest evaluation',
            'manifest':args.manifest,'prompt':str(args.prompt)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
