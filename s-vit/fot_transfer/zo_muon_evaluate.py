"""Evaluate a validation-selected ZO-Muon checkpoint on CIFAR-10 test data."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import install_head, load_data, load_head, source_model
from .train import emit
from .zo_muon_vit import TrainableViT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'splits', 'checkpoint', 'fixed-student', 'mpcvit-root', 'head', 'selected-checkpoint', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--allow-plaintext-gelu-out-of-range', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_samples < 0:
        parser.error('invalid sample or batch size')
    output = Path(args.output)
    if output.exists():
        parser.error('output already exists')
    torch.set_num_threads(1)
    device = torch.device(args.device)
    _, split, data = load_data(args.archive, args.splits, include_test=True)
    head = load_head(args, split)
    if bool(head['allow_plaintext_gelu_out_of_range']) != args.allow_plaintext_gelu_out_of_range:
        raise ValueError('head and run GeLU range policies differ')
    spec, params = install_head(*source_model(args), head)
    model = TrainableViT(spec, {k: v.detach().to(device) for k, v in params.items()}, args.allow_plaintext_gelu_out_of_range)
    selected = torch.load(args.selected_checkpoint, map_location='cpu', weights_only=True)
    if selected['names'] != model.names or len(selected['parameters']) != len(model.values):
        raise ValueError('checkpoint parameter names do not match model')
    for value, target in zip(selected['parameters'], model.values):
        if value.shape != target.shape or not torch.isfinite(value).all():
            raise ValueError('checkpoint contains invalid parameter')
    test = data['test']
    limit = min(args.max_samples, len(test)) if args.max_samples else len(test)

    def evaluate(label):
        count = correct = loss = 0
        model.audit = {'gelu_out_of_range_calls': 0, 'gelu_out_of_range_elements': 0, 'gelu_max_abs': 0.0}
        with torch.inference_mode():
            for x, y in test.chunks(args.batch_size):
                take = min(len(y), limit - count)
                x, y = x[:take].to(device), y[:take].to(device)
                logits = model(x, None)
                loss += F.cross_entropy(logits, y, reduction='sum').item()
                correct += (logits.argmax(-1) == y).sum().item()
                count += take
                if count % (args.batch_size * 50) == 0 or count == limit:
                    emit('evaluation_progress', model=label, samples=count, total=limit)
                if count == limit:
                    break
        return {'samples': count, 'loss': loss / count, 'top1': correct / count, 'range_audit': dict(model.audit)}

    baseline = evaluate('initial')
    with torch.no_grad():
        for value, target in zip(selected['parameters'], model.values):
            target.copy_(value.to(device=target.device, dtype=target.dtype))
    trained = evaluate('trained')
    report = {'scope': 'official CIFAR-10 test', 'initial': baseline, 'trained': trained,
              'selected_checkpoint': str(Path(args.selected_checkpoint).resolve()),
              'selected_step': selected.get('step'), 'selected_epoch': selected.get('epoch'),
              'note': 'Legacy checkpoints contain paths but no content hashes; verify checkpoint provenance.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    emit('complete', report=str(output), initial=baseline, trained=trained)


if __name__ == '__main__':
    main()
