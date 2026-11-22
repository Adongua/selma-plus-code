"""Supervised zeroth-order prompt-token experiment on CIFAR-10."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import install_head, load_data, load_head, source_model
from .data import CifarData
from .epoch_compare import epoch_batches
from .fozo_prompt import prompt_forward
from .train import emit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'splits', 'checkpoint', 'fixed-student', 'mpcvit-root', 'head', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--eval-batch-size', type=int, default=16)
    parser.add_argument('--max-batches-per-epoch', type=int, default=0)
    parser.add_argument('--validation-max-samples', type=int, default=0)
    parser.add_argument('--prompt-count', type=int, default=3)
    parser.add_argument('--directions', type=int, default=1)
    parser.add_argument('--sigma', type=float, default=.01)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=20261107)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--allow-plaintext-gelu-out-of-range', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.prompt_count, args.directions, args.log_every) < 1 or min(args.sigma, args.lr) <= 0:
        parser.error('invalid training configuration')
    root = Path(args.output).resolve()
    if root.exists():
        parser.error('output already exists')
    torch.set_num_threads(1)
    device = torch.device(args.device)
    archive, split, data = load_data(args.archive, args.splits)
    if args.validation_max_samples:
        data['val'] = CifarData(archive, 'train', data['val'].indices[:args.validation_max_samples])
    head = load_head(args, split)
    if bool(head['allow_plaintext_gelu_out_of_range']) != args.allow_plaintext_gelu_out_of_range:
        raise ValueError('head and run GeLU range policies differ')
    spec, params = install_head(*source_model(args), head)
    params = {name: value.detach().to(device) for name, value in params.items()}
    generator = torch.Generator().manual_seed(args.seed)
    # Small initial prompts allow evaluation of insertion effects independently.
    prompt = torch.randn(args.prompt_count, spec['dim'], generator=generator, dtype=torch.float64).to(device) * .001
    root.mkdir(parents=True)
    audits = {'gelu_out_of_range_calls': 0, 'gelu_out_of_range_elements': 0, 'gelu_max_abs': 0.0}

    def predict(x, value):
        logits, audit = prompt_forward(spec, params, x, value, args.allow_plaintext_gelu_out_of_range)
        for key in ('gelu_out_of_range_calls', 'gelu_out_of_range_elements'):
            audits[key] += audit[key]
        audits['gelu_max_abs'] = max(audits['gelu_max_abs'], audit['gelu_max_abs'])
        return logits

    def evaluate(value):
        count = correct = loss = 0
        for x, y in data['val'].chunks(args.eval_batch_size):
            x, y = x.to(device), y.to(device)
            logits = predict(x, value)
            loss += F.cross_entropy(logits, y, reduction='sum').item()
            correct += (logits.argmax(-1) == y).sum().item()
            count += len(y)
        return {'samples': count, 'loss': loss / count, 'top1': correct / count}

    with torch.no_grad():
        baseline = evaluate(prompt[:0])
        initial = evaluate(prompt)
        emit('training_started', no_prompt=baseline, initial=initial)
        records, step, queries = [], 0, 0
        for epoch in range(1, args.epochs + 1):
            samples = loss_sum = 0
            for batch, (owner, x, y) in enumerate(epoch_batches(data, epoch, args.batch_size, args.seed)):
                if args.max_batches_per_epoch and batch >= args.max_batches_per_epoch:
                    break
                x, y = x.to(device), y.to(device)
                estimate = torch.zeros_like(prompt)
                batch_loss = 0
                for _ in range(args.directions):
                    direction = torch.randn(prompt.shape, generator=generator, dtype=prompt.dtype).to(device)
                    plus = F.cross_entropy(predict(x, prompt + args.sigma * direction), y)
                    minus = F.cross_entropy(predict(x, prompt - args.sigma * direction), y)
                    estimate += ((plus - minus) / (2 * args.sigma)) * direction
                    batch_loss += ((plus + minus) / 2).item()
                prompt -= args.lr * estimate / args.directions
                if not torch.isfinite(prompt).all():
                    raise ArithmeticError('nonfinite prompt')
                step += 1
                queries += 2 * args.directions
                samples += len(y)
                loss_sum += batch_loss / args.directions * len(y)
                if step % args.log_every == 0:
                    emit('training_progress', epoch=epoch, step=step, owner=owner, samples=samples, queries=queries, running_perturbed_loss=loss_sum / samples)
            validation = evaluate(prompt)
            row = {'epoch': epoch, 'step': step, 'samples': samples, 'queries': queries,
                   'training_perturbed_loss': loss_sum / samples, 'validation': validation}
            records.append(row)
            torch.save({'prompt': prompt.cpu(), 'options': vars(args), 'epoch': epoch}, root / f'epoch-{epoch:04d}.pt')
            report = {'status': 'completed' if epoch == args.epochs else 'running', 'no_prompt': baseline,
                      'initial_validation': initial, 'epochs': records, 'options': vars(args),
                      'trainable_parameters': prompt.numel(), 'range_audit': audits}
            (root / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            emit('epoch_complete', **row)


if __name__ == '__main__':
    main()
