"""Plaintext zeroth-order experiments for structured additive patch prompts."""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from hattention.student_checkpoint import file_sha256
from training.model import forward
from fot_image.plaintext_ops import AccuracyPlainOps
from .common import install_head, load_data, load_head, source_model
from .data import CifarData
from .epoch_compare import epoch_batches
from .train import emit


def make_prompt(theta, method, rank, grid, dim):
    if method == 'grid':
        coarse = theta.reshape(1, 4, 4, dim).permute(0, 3, 1, 2)
        return F.interpolate(coarse, size=grid, mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0).reshape(-1, dim)
    split = grid[0] * grid[1] * rank
    return theta[:split].reshape(-1, rank) @ theta[split:].reshape(rank, dim)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'splits', 'checkpoint', 'fixed-student', 'mpcvit-root', 'head', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--method', choices=('grid', 'lowrank'), required=True)
    p.add_argument('--rank', type=int, default=2)
    p.add_argument('--directions', type=int, default=1)
    p.add_argument('--block-size', type=int, default=192)
    p.add_argument('--sigma', type=float, default=.01)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--eval-batch-size', type=int, default=16)
    p.add_argument('--validation-max-samples', type=int, default=0)
    p.add_argument('--max-batches-per-epoch', type=int, default=0)
    p.add_argument('--seed', type=int, default=20261107)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--allow-plaintext-gelu-out-of-range', action='store_true')
    a = p.parse_args()
    if min(a.rank, a.directions, a.block_size, a.epochs, a.batch_size, a.eval_batch_size, a.log_every) < 1 or min(a.validation_max_samples, a.max_batches_per_epoch) < 0 or not all(math.isfinite(v) and v > 0 for v in (a.lr, a.sigma)):
        p.error('invalid configuration')
    if a.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable')
    root = Path(a.output).resolve()
    if root.exists():
        p.error('output already exists')
    torch.set_num_threads(1)
    device = torch.device(a.device)
    archive, split, data = load_data(a.archive, a.splits)
    if a.validation_max_samples:
        if a.validation_max_samples > len(data['val']):
            p.error('validation sample limit exceeds split')
        data['val'] = CifarData(archive, 'train', data['val'].indices[:a.validation_max_samples])
    head = load_head(a, split)
    if bool(head['allow_plaintext_gelu_out_of_range']) != a.allow_plaintext_gelu_out_of_range:
        raise ValueError('head and run GeLU range policies differ')
    spec, params = install_head(*source_model(a), head)
    params = {k: v.detach().to(device) for k, v in params.items()}
    ops = AccuracyPlainOps(params, a.allow_plaintext_gelu_out_of_range)
    grid, dim = tuple(spec['grid']), spec['dim']
    generator = torch.Generator().manual_seed(a.seed)
    if a.method == 'grid':
        theta = torch.zeros(16 * dim, dtype=torch.float64, device=device)
    else:
        left = torch.randn(grid[0] * grid[1], a.rank, generator=generator, dtype=torch.float64) / math.sqrt(a.rank)
        theta = torch.cat((left.flatten(), torch.zeros(a.rank * dim, dtype=torch.float64))).to(device)
    m, v = torch.zeros_like(theta), torch.zeros_like(theta)
    # Alternate low-rank factors; within each factor cycle deterministic coordinate blocks.
    split_at = grid[0] * grid[1] * a.rank if a.method == 'lowrank' else 0
    blocks = []
    spans = [(0, len(theta))] if a.method == 'grid' else [(split_at, len(theta)), (0, split_at)]
    by_factor = [[(i, min(i + a.block_size, end)) for i in range(start, end, a.block_size)] for start, end in spans]
    for index in range(max(map(len, by_factor))):
        for group in by_factor:
            if index < len(group):
                blocks.append(group[index])
    hashes = {'archive_sha256': archive['archive_sha256'], 'split_sha256': file_sha256(a.splits),
              'head_sha256': file_sha256(a.head), 'fixed_student_sha256': file_sha256(a.fixed_student),
              'source_checkpoint_sha256': file_sha256(a.checkpoint)}
    root.mkdir(parents=True, mode=0o700)
    queries = 0

    def predict(x, value):
        return forward(spec, ops, x, make_prompt(value, a.method, a.rank, grid, dim))

    def evaluate(value):
        total = correct = count = 0
        for x, y in data['val'].chunks(a.eval_batch_size):
            x, y = x.to(device), y.to(device)
            out = predict(x, value)
            if not torch.isfinite(out).all():
                raise ArithmeticError('nonfinite validation logits')
            total += F.cross_entropy(out, y, reduction='sum').item()
            correct += (out.argmax(-1) == y).sum().item()
            count += len(y)
            if count % (a.eval_batch_size * 50) == 0 or count == len(data['val']):
                emit('evaluation_progress', samples=count, total=len(data['val']))
        return {'loss': total / count, 'top1': correct / count, 'samples': count}

    with torch.no_grad():
        initial = evaluate(theta)
        rows = []
        step = 0
        for epoch in range(1, a.epochs + 1):
            total_loss = samples = 0
            for batch, (owner, x, y) in enumerate(epoch_batches(data, epoch, a.batch_size, a.seed)):
                if a.max_batches_per_epoch and batch >= a.max_batches_per_epoch:
                    break
                x, y = x.to(device), y.to(device)
                start, end = blocks[step % len(blocks)]
                estimate = torch.zeros(end - start, dtype=theta.dtype, device=device)
                loss_sum = rms_sum = 0.
                for _ in range(a.directions):
                    u = (2 * torch.randint(0, 2, (end - start,), generator=generator) - 1).to(device=device, dtype=theta.dtype)
                    delta = torch.zeros_like(theta)
                    delta[start:end] = a.sigma * u
                    plus, minus = theta + delta, theta - delta
                    lp = F.cross_entropy(predict(x, plus), y).item()
                    lm = F.cross_entropy(predict(x, minus), y).item()
                    queries += 2
                    estimate += (lp - lm) / (2 * a.sigma) * u
                    loss_sum += (lp + lm) / 2
                    rms_sum += ((make_prompt(plus, a.method, a.rank, grid, dim) - make_prompt(minus, a.method, a.rank, grid, dim)) / 2).square().mean().sqrt().item()
                estimate /= a.directions
                step += 1
                # Bias correction is per coordinate because inactive blocks do not update.
                if step == 1:
                    counts = torch.zeros_like(theta)
                counts[start:end] += 1
                m[start:end] = .9 * m[start:end] + .1 * estimate
                v[start:end] = .99 * v[start:end] + .01 * estimate.square()
                mh = m[start:end] / (1 - .9 ** counts[start:end])
                vh = v[start:end] / (1 - .99 ** counts[start:end])
                theta[start:end] -= a.lr * mh / (vh.sqrt() + .01)
                if not torch.isfinite(theta).all():
                    raise ArithmeticError('nonfinite parameters')
                samples += len(y)
                total_loss += loss_sum / a.directions * len(y)
                if step % a.log_every == 0:
                    emit('training_progress', epoch=epoch, step=step, owner=owner, samples=samples,
                         queries=queries, perturbed_loss=loss_sum / a.directions,
                         perturbation_prompt_rms=rms_sum / a.directions, coefficient_max=estimate.abs().max().item())
            validation = evaluate(theta)
            row = {'epoch': epoch, 'step': step, 'samples': samples, 'queries': queries,
                   'training_perturbed_loss': total_loss / samples, 'validation': validation,
                   'prompt_rms': make_prompt(theta, a.method, a.rank, grid, dim).square().mean().sqrt().item()}
            rows.append(row)
            torch.save({'format': 'structured_zo_v1', 'theta': theta.cpu(), 'options': vars(a), 'hashes': hashes, 'epoch': epoch}, root / f'epoch-{epoch:04d}.pt')
            report = {'status': 'completed' if epoch == a.epochs else 'running', 'initial_validation': initial,
                      'epochs': rows, 'options': vars(a), 'hashes': hashes, 'trainable_parameters': len(theta),
                      'range_audit': ops.range_report(), 'note': 'Plaintext research experiment, no MPC or security validation. Queries exclude validation.'}
            (root / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            emit('epoch_complete', **row)
    emit('complete', report=str(root / 'report.json'))


if __name__ == '__main__':
    main()
