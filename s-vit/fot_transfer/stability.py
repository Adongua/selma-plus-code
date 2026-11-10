"""Plaintext image-prompt ZO stability experiment; leaves original training untouched."""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from hattention.student_checkpoint import file_sha256
from fot_explore.conditioning import LocalEncoder
from fot_image.decoder import FEATURE_DIM, PlainDecoderOps, image_prompt, initial_parameters, rademacher
from fot_image.plaintext_ops import AccuracyPlainOps
from .common import install_head, load_data, load_head, source_model
from .data import CifarData, balanced_subset
from .diagnose_fit import visited_indices
from .train import emit, evaluate, logits_for


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'splits', 'checkpoint', 'fixed-student', 'mpcvit-root', 'head', 'mae-path', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--eval-every', type=int, default=200)
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--eval-batch-size', type=int, default=8)
    p.add_argument('--val-samples', type=int, default=1000)
    p.add_argument('--train-samples', type=int, default=1000)
    p.add_argument('--visited-steps', type=int, default=600)
    p.add_argument('--sigma', type=float, default=.05)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--momentum', type=float, default=.9)
    p.add_argument('--prompt-scale', type=float, default=.1)
    p.add_argument('--seed', type=int, default=20261107)
    p.add_argument('--allow-plaintext-gelu-out-of-range', action='store_true')
    a = p.parse_args()
    if min(a.steps, a.eval_every, a.log_every, a.batch_size, a.eval_batch_size, a.val_samples, a.train_samples, a.visited_steps) < 1 or a.visited_steps > a.steps: p.error('invalid sample or step counts')
    if any(not math.isfinite(v) or v <= 0 for v in (a.sigma, a.lr, a.prompt_scale)) or not 0 <= a.momentum < 1: p.error('invalid hyperparameters')
    if a.device == 'cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable')
    root = Path(a.output).resolve()
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    torch.set_num_threads(1)
    device = torch.device(a.device)
    archive, split, data = load_data(a.archive, a.splits, include_test=False)
    head = load_head(a, split)
    if bool(head['allow_plaintext_gelu_out_of_range']) != a.allow_plaintext_gelu_out_of_range: raise ValueError('GeLU range policy differs from head')
    spec, params = install_head(*source_model(a), head)
    ops = AccuracyPlainOps({k: v.to(device) for k, v in params.items()}, a.allow_plaintext_gelu_out_of_range)
    encoder = LocalEncoder('mae', FEATURE_DIM, device, a.mae_path, seed=a.seed + 44)
    mae_file = next((f for f in (Path(a.mae_path)/'model.safetensors', Path(a.mae_path)/'pytorch_model.bin') if f.is_file()), None)
    if mae_file is None: raise ValueError('MAE weights not found')
    hashes = {'head_sha256': file_sha256(a.head), 'fixed_student_sha256': file_sha256(a.fixed_student),
              'archive_sha256': archive['archive_sha256'], 'split_sha256': file_sha256(a.splits),
              'mae_sha256': file_sha256(mae_file), 'mae_config_sha256': file_sha256(Path(a.mae_path)/'config.json')}
    labels = archive['train'][1]
    val = CifarData(archive, 'train', balanced_subset(data['val'].indices, labels, a.val_samples))
    train_ids = balanced_subset(visited_indices(data, a.visited_steps, a.batch_size), labels, a.train_samples)
    train = CifarData(archive, 'train', train_ids)
    dec = PlainDecoderOps()
    theta = initial_parameters(a.seed).to(device)
    velocity = torch.zeros_like(theta)
    options = vars(a)
    (root/'options.json').write_text(json.dumps(options, indent=2)+'\n')
    rows, evaluations = [], []

    def do_evaluation(step):
        emit('evaluation_started', step=step)
        training = evaluate('image-zo', spec, ops, dec, encoder, train, theta, None, a.prompt_scale, device, a.eval_batch_size)
        validation = evaluate('image-zo', spec, ops, dec, encoder, val, theta, None, a.prompt_scale, device, a.eval_batch_size)
        record = {'step': step, 'training_subset': training, 'validation': validation}
        evaluations.append(record)
        torch.save({'format': 'cifar10_stability_prompt_v1', 'method': 'image-zo', 'step': step,
                    'weights': theta.detach().cpu(), 'velocity': velocity.detach().cpu(), 'options': options, **hashes}, root/f'step-{step:06d}.pt')
        emit('evaluation_complete', **record)

    do_evaluation(0)
    for step in range(1, a.steps + 1):
        owner = (step - 1) % 3
        x, y = data[f'party-{owner}'].batch((step - 1)//3, a.batch_size)
        x, y = x.to(device), y.to(device)
        feature = encoder(x)
        u = rademacher(step, a.seed).to(device)
        center = theta + a.momentum * velocity
        with torch.no_grad():
            plus = F.cross_entropy(logits_for('image-zo', spec, ops, dec, x, feature, center + a.sigma*u, None, a.prompt_scale), y).item()
            minus = F.cross_entropy(logits_for('image-zo', spec, ops, dec, x, feature, center - a.sigma*u, None, a.prompt_scale), y).item()
            coefficient = (plus - minus)/(2*a.sigma)
            if not math.isfinite(coefficient) or abs(coefficient) > 8: raise ArithmeticError('SPSA coefficient outside [-8,8]')
            velocity = a.momentum*velocity - a.lr*coefficient*u
            theta = theta + velocity
            if not torch.isfinite(theta).all() or theta.abs().max() > 8: raise ArithmeticError('CNN parameter outside [-8,8]')
            prompt = image_prompt(feature, theta, dec, spec['image_shape']) * a.prompt_scale
            row = {'step': step, 'owner': owner, 'loss_plus': plus, 'loss_minus': minus,
                   'directional_derivative': coefficient, 'parameter_l2': theta.norm().item(),
                   'parameter_max_abs': theta.abs().max().item(), 'update_l2': velocity.norm().item(),
                   'pixel_prompt_rms': prompt.square().mean().sqrt().item(), 'pixel_prompt_max_abs': prompt.abs().max().item()}
            if any(not math.isfinite(v) for v in row.values()): raise ArithmeticError('nonfinite diagnostic')
        rows.append(row)
        if step % a.log_every == 0 or step == 1: emit('step_complete', **row)
        if step % a.eval_every == 0 or step == a.steps: do_evaluation(step)
    report = {'status': 'completed_plaintext', 'options': options, **hashes, 'evaluations': evaluations,
              'steps': rows, 'train_subset_indices': train_ids, 'range_audit': ops.range_report(),
              'note': 'Fixed train subset visited by step visited_steps; earlier checkpoints may not have seen all samples. No test data loaded. Plaintext diagnostics do not establish MPC correctness.'}
    (root/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    emit('complete', report=str(root/'report.json'))


if __name__ == '__main__': main()
