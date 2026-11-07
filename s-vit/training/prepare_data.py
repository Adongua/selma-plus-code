"""Create disjoint public Tiny-ImageNet diagnostic manifests (never moves data)."""
import argparse
import json
from pathlib import Path
import random


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--classes', type=int, default=4)
    p.add_argument('--per-party-per-class', type=int, default=1)
    p.add_argument('--eval-per-class', type=int, default=1)
    p.add_argument('--seed', type=int, default=20261107)
    args = p.parse_args()
    classes = sorted((args.dataset_root/'wnids.txt').read_text().splitlines())
    if len(classes) != 200 or len(set(classes)) != 200:
        p.error('expected 200 distinct wnids')
    if not 1 <= args.classes <= 200 or min(args.per_party_per_class,args.eval_per_class) < 1:
        p.error('invalid split sizes')
    parties, evaluation = [[],[],[]], []
    for label, wnid in enumerate(classes[:args.classes]):
        paths = sorted((args.dataset_root/'train'/wnid/'images').glob('*.JPEG'))
        random.Random(f'{args.seed}:{wnid}').shuffle(paths)
        needed = 3*args.per_party_per_class + args.eval_per_class
        if len(paths) < needed:
            raise ValueError(f'insufficient images for {wnid}')
        rows = [{'path': path.relative_to(args.dataset_root).as_posix(), 'label': label} for path in paths[:needed]]
        for owner in range(3):
            parties[owner].extend(rows[owner*args.per_party_per_class:(owner+1)*args.per_party_per_class])
        evaluation.extend(rows[3*args.per_party_per_class:])
    args.output.mkdir(parents=True, exist_ok=False)
    for owner, rows in enumerate(parties):
        (args.output/f'party-{owner}.json').write_text(json.dumps(rows,indent=2)+'\n')
    (args.output/'eval.json').write_text(json.dumps(evaluation,indent=2)+'\n')
    (args.output/'split.json').write_text(json.dumps({'class_order':classes, 'seed':args.seed,
        'party_samples':[len(x) for x in parties], 'eval_samples':len(evaluation),
        'scope':'public diagnostic split; may have been seen by pretrained backbone; not official validation'},indent=2)+'\n')
    print(json.dumps({'manifests':str(args.output),'party_samples':[len(x) for x in parties]}))


if __name__ == '__main__':
    main()
