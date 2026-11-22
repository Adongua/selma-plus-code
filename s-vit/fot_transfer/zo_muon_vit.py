"""Full-parameter ZO-Muon adaptation of the fixed hAttention ViT in plaintext."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from fot_image.plaintext_ops import AccuracyPlainOps
from training.model import forward
from .common import install_head, load_data, load_head, source_model
from .data import CifarData
from .epoch_compare import epoch_batches
from .train import emit


class TrainableViT(torch.nn.Module):
    def __init__(self, spec, params, allow_out_of_range):
        super().__init__()
        self.spec = spec
        self.names = list(params)
        self.values = torch.nn.ParameterList([torch.nn.Parameter(params[name]) for name in self.names])
        self.allow_out_of_range = allow_out_of_range
        self.audit = {'gelu_out_of_range_calls': 0, 'gelu_out_of_range_elements': 0, 'gelu_max_abs': 0.0}

    def forward(self, images, labels):
        params = dict(zip(self.names, self.values))
        ops = AccuracyPlainOps(params, self.allow_out_of_range)
        prompt = images.new_zeros((self.spec['patches'], self.spec['dim']))
        logits = forward(self.spec, ops, images, prompt)
        audit = ops.range_report()
        for key in ('gelu_out_of_range_calls', 'gelu_out_of_range_elements'):
            self.audit[key] += audit[key]
        self.audit['gelu_max_abs'] = max(self.audit['gelu_max_abs'], audit['gelu_max_abs'])
        if labels is None:
            return logits
        return F.cross_entropy(logits, labels)


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
    parser.add_argument('--rank', type=int, default=64)
    parser.add_argument('--directions', type=int, default=16)
    parser.add_argument('--step-interval', type=int, default=100)
    parser.add_argument('--sigma', type=float, default=.001)
    parser.add_argument('--lr', type=float, default=.01)
    parser.add_argument('--seed', type=int, default=20261107)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--eval-every-steps', type=int, default=0)
    parser.add_argument('--train-eval-samples', type=int, default=0)
    parser.add_argument('--allow-plaintext-gelu-out-of-range', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.rank, args.directions, args.step_interval, args.log_every) < 1 or min(args.sigma, args.lr) <= 0 or min(args.max_batches_per_epoch, args.validation_max_samples, args.eval_every_steps, args.train_eval_samples) < 0:
        parser.error('invalid training configuration')
    root = Path(args.output).resolve()
    if root.exists():
        parser.error('output already exists')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    device = torch.device(args.device)
    archive, split, data = load_data(args.archive, args.splits)
    if args.validation_max_samples:
        if args.validation_max_samples > len(data['val']):
            parser.error('validation sample limit exceeds split')
        data['val'] = CifarData(archive, 'train', data['val'].indices[:args.validation_max_samples])
    head = load_head(args, split)
    if bool(head['allow_plaintext_gelu_out_of_range']) != args.allow_plaintext_gelu_out_of_range:
        raise ValueError('head and run GeLU range policies differ')
    spec, params = install_head(*source_model(args), head)
    model = TrainableViT(spec, {k: v.detach().to(device) for k, v in params.items()}, args.allow_plaintext_gelu_out_of_range)
    upstream = Path(__file__).resolve().parents[2] / 'ZO-Muon' / 'vit'
    sys.path.insert(0, str(upstream))
    from zo_optimizer import LowDimOptimizer
    options = SimpleNamespace(zo_eps=args.sigma, rank_r=args.rank, step_interval=args.step_interval,
                              multiple_sample=True, num_samples=args.directions, n_gpu=1 if args.device == 'cuda' else 0,
                              learning_rate=args.lr, zo_optimizer='muon', weight_decay=0)
    optimizer = LowDimOptimizer(model, options)
    root.mkdir(parents=True)

    train_subset = []
    if args.train_eval_samples:
        for _, x, y in epoch_batches(data, 1, args.batch_size, args.seed):
            remaining = args.train_eval_samples - sum(len(labels) for _, labels in train_subset)
            train_subset.append((x[:remaining], y[:remaining]))
            if remaining <= len(y):
                break
        if sum(len(y) for _, y in train_subset) != args.train_eval_samples:
            parser.error('training sample limit exceeds split')

    def evaluate(batches):
        count = correct = loss = 0
        with torch.inference_mode():
            for x, y in batches:
                x, y = x.to(device), y.to(device)
                logits = model(x, None)
                loss += F.cross_entropy(logits, y, reduction='sum').item()
                correct += (logits.argmax(-1) == y).sum().item()
                count += len(y)
        return {'samples': count, 'loss': loss / count, 'top1': correct / count}

    def validation():
        return evaluate(data['val'].chunks(args.eval_batch_size))

    initial = validation()
    initial_training = evaluate(train_subset) if train_subset else None
    emit('training_started', initial=initial, initial_training_subset=initial_training, trainable_parameters=sum(p.numel() for p in model.parameters()))
    records, observations, step = [], [], 0

    def write_report(status):
        report = {'status': status, 'initial_validation': initial, 'initial_training_subset': initial_training,
                  'epochs': records, 'observations': observations, 'options': vars(args), 'range_audit': model.audit,
                  'trainable_parameters': sum(p.numel() for p in model.parameters())}
        (root / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    def save_checkpoint(path, epoch):
        torch.save({'names': model.names, 'parameters': [p.detach().cpu() for p in model.values],
                    'options': vars(args), 'epoch': epoch, 'step': step}, root / path)
    for epoch in range(1, args.epochs + 1):
        samples = loss_sum = 0
        for batch, (owner, x, y) in enumerate(epoch_batches(data, epoch, args.batch_size, args.seed)):
            if args.max_batches_per_epoch and batch >= args.max_batches_per_epoch:
                break
            x, y = x.to(device), y.to(device)
            loss = optimizer.zo_step(x, y)
            optimizer.zo_update()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ArithmeticError('nonfinite model parameters')
            step += 1
            samples += len(y)
            loss_sum += loss.item() * len(y)
            if step % args.log_every == 0:
                emit('training_progress', epoch=epoch, step=step, owner=owner, samples=samples,
                     queries=step * (1 + args.directions), running_loss=loss_sum / samples)
            if args.eval_every_steps and step % args.eval_every_steps == 0:
                observation = {'epoch': epoch, 'step': step, 'queries': step * (1 + args.directions),
                               'training_subset': evaluate(train_subset) if train_subset else None,
                               'validation': validation()}
                observations.append(observation)
                save_checkpoint(f'step-{step:06d}.pt', epoch)
                write_report('running')
                emit('step_evaluation', **observation)
        val = validation()
        row = {'epoch': epoch, 'step': step, 'samples': samples, 'queries': step * (1 + args.directions),
               'training_loss': loss_sum / samples, 'training_subset': evaluate(train_subset) if train_subset else None,
               'validation': val}
        records.append(row)
        save_checkpoint(f'epoch-{epoch:04d}.pt', epoch)
        write_report('completed' if epoch == args.epochs else 'running')
        emit('epoch_complete', **row)


if __name__ == '__main__':
    main()
