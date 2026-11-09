"""Prepare immutable, disjoint CIFAR-10 public/party/validation/test indices."""
import argparse
import json
from pathlib import Path
from .data import read_official_archive, make_splits, CIFAR10_CLASSES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',required=True); p.add_argument('--output',required=True)
    p.add_argument('--public-per-class',type=int,default=500); p.add_argument('--val-per-class',type=int,default=500)
    p.add_argument('--seed',type=int,default=20261109)
    a=p.parse_args()
    if a.public_per_class<1 or a.val_per_class<1 or a.seed<0: p.error('invalid split arguments')
    archive=read_official_archive(a.archive,include_test=False)
    groups=make_splits(archive['train'][1],a.public_per_class,a.val_per_class,a.seed)
    # Official archive checksum fixes the test split at 10,000 examples;
    # do not decode its images during split construction or model selection.
    groups['test']=list(range(10000))
    report={'format':'cifar10_transfer_splits_v1','archive_sha256':archive['archive_sha256'],
            'classes':list(CIFAR10_CLASSES),'seed':a.seed,'indices':groups,
            'note':'public, party and validation indices address official train; test indices address official test'}
    out=Path(a.output).resolve()
    if out.exists(): p.error('output exists; choose a new path')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(out),'counts':{k:len(v) for k,v in groups.items()},'archive_sha256':archive['archive_sha256']}),flush=True)


if __name__=='__main__': main()
