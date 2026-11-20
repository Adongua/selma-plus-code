"""Compare low-rank V finite differences against autograd on fixed training batches."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from fot_image.plaintext_ops import AccuracyPlainOps
from training.model import forward
from .common import install_head, load_data, load_head, source_model
from .epoch_compare import epoch_batches
from .structured_zo import make_prompt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'splits', 'checkpoint', 'fixed-student', 'mpcvit-root', 'head', 'prompt-checkpoint', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--sigmas', type=float, nargs='+', default=[.01, .003, .001])
    p.add_argument('--batches', type=int, default=3)
    p.add_argument('--directions', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seed', type=int, default=20261107)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--allow-plaintext-gelu-out-of-range', action='store_true')
    a = p.parse_args()
    if min(a.batches, a.directions, a.batch_size) < 1 or any(s <= 0 for s in a.sigmas):
        p.error('counts and sigmas must be positive')
    torch.set_num_threads(1)
    device = torch.device(a.device)
    archive, split, data = load_data(a.archive, a.splits)
    head = load_head(a, split)
    if bool(head['allow_plaintext_gelu_out_of_range']) != a.allow_plaintext_gelu_out_of_range:
        raise ValueError('head and run GeLU range policies differ')
    spec, params = install_head(*source_model(a), head)
    params = {k: v.detach().to(device) for k, v in params.items()}
    ops = AccuracyPlainOps(params, a.allow_plaintext_gelu_out_of_range)
    ck = torch.load(a.prompt_checkpoint, map_location='cpu', weights_only=True)
    if ck.get('format') != 'structured_zo_v1' or ck['options']['method'] != 'lowrank':
        raise ValueError('expected a low-rank structured checkpoint')
    from hattention.student_checkpoint import file_sha256
    for key, path in [('fixed_student_sha256', a.fixed_student), ('head_sha256', a.head), ('split_sha256', a.splits), ('source_checkpoint_sha256', a.checkpoint)]:
        if ck['hashes'][key] != file_sha256(path):
            raise ValueError(f'checkpoint mismatch: {key}')
    rank = ck['options']['rank']
    grid, dim = tuple(spec['grid']), spec['dim']
    theta = ck['theta'].to(device).double().detach().requires_grad_(True)
    offset = grid[0] * grid[1] * rank
    generator = torch.Generator().manual_seed(a.seed)
    records = []
    for index, (owner, x, y) in enumerate(epoch_batches(data, 1, a.batch_size, a.seed)):
        if index >= a.batches:
            break
        x, y = x.to(device), y.to(device)
        def objective(value):
            prompt = make_prompt(value, 'lowrank', rank, grid, dim)
            return F.cross_entropy(forward(spec, ops, x, prompt), y)
        value = objective(theta)
        grad, = torch.autograd.grad(value, theta)
        directions = (2 * torch.randint(0, 2, (a.directions, theta.numel() - offset), generator=generator) - 1).to(device=device, dtype=theta.dtype)
        truth = directions @ grad[offset:]
        measurements = []
        with torch.no_grad():
            for sigma in a.sigmas:
                approx, differences = [], []
                for direction in directions:
                    delta = torch.zeros_like(theta)
                    delta[offset:] = sigma * direction
                    diff = (objective(theta + delta) - objective(theta - delta)).item()
                    differences.append(abs(diff))
                    approx.append(diff / (2 * sigma))
                estimate = torch.tensor(approx, dtype=truth.dtype, device=device)
                denom = truth.norm().item()
                measurements.append({'sigma': sigma, 'relative_l2_error': (estimate-truth).norm().item()/denom if denom else None,
                                     'mean_absolute_loss_difference': sum(differences)/len(differences),
                                     'exact_directional_rms': truth.square().mean().sqrt().item(),
                                     'finite_directional_rms': estimate.square().mean().sqrt().item()})
        row = {'batch': index + 1, 'owner': owner, 'loss': value.item(), 'measurements': measurements}
        records.append(row)
        print(json.dumps({'event': 'batch_complete', **row}), flush=True)
    report = {'scope': 'plaintext audit, full V factor; autograd used for diagnostics only', 'options': vars(a),
              'records': records, 'range_audit': ops.range_report()}
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'event': 'complete', 'report': str(path)}), flush=True)


if __name__ == '__main__':
    main()
