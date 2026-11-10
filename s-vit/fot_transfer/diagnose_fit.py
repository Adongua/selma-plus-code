"""Evaluate saved prompt checkpoints on a fixed, previously visited training subset."""
import argparse
import json
from pathlib import Path

import torch

from hattention.student_checkpoint import file_sha256
from training.core import ZOConfig, basis
from fot_explore.conditioning import LocalEncoder
from fot_image.decoder import FEATURE_DIM, PARAMS, PlainDecoderOps, initial_parameters
from fot_image.plaintext_ops import AccuracyPlainOps
from .common import load_data, load_head, source_model, install_head
from .data import CifarData, balanced_subset
from .train import emit, evaluate


def visited_indices(data, steps, batch_size):
    """Reconstruct samples visited by round-robin training up to `steps`."""
    result = []
    for step in range(steps):
        indices = data[f'party-{step % 3}'].indices
        start = (step // 3) * batch_size
        result.extend(indices[(start + i) % len(indices)] for i in range(batch_size))
    return list(dict.fromkeys(result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'splits', 'checkpoint', 'fixed-student', 'mpcvit-root', 'head', 'mae-path', 'baseline', 'image-zo', 'image-adam', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--steps', type=int, nargs='+', default=[0, 600, 1400, 3000])
    p.add_argument('--visited-steps', type=int, default=600)
    p.add_argument('--train-samples', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--allow-plaintext-gelu-out-of-range', action='store_true')
    a = p.parse_args()
    if min(a.visited_steps, a.train_samples, a.batch_size) < 1 or any(s < 0 for s in a.steps) or len(set(a.steps)) != len(a.steps):
        p.error('invalid steps or sample counts')
    if a.device == 'cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable')
    out = Path(a.output).resolve()
    if out.exists(): p.error('output exists; choose a new path')
    torch.set_num_threads(1)
    device = torch.device(a.device)
    archive, split, data = load_data(a.archive, a.splits, include_test=False)
    head = load_head(a, split)
    if bool(head['allow_plaintext_gelu_out_of_range']) != a.allow_plaintext_gelu_out_of_range: raise ValueError('GeLU range policy differs from head')
    spec, params = install_head(*source_model(a), head)
    params = {k: v.to(device) for k, v in params.items()}
    roots = {'baseline': a.baseline, 'image-zo': a.image_zo, 'image-adam': a.image_adam}
    reports, options = {}, {}
    signature = None
    for method, directory in roots.items():
        root = Path(directory)
        reports[method] = json.loads((root / 'report.json').read_text())
        options[method] = json.loads((root / 'options.json').read_text())
        if reports[method].get('status') != 'completed_plaintext' or reports[method].get('method') != method: raise ValueError('invalid training report')
        opt = options[method]
        current = tuple(opt[k] for k in ('seed', 'steps', 'eval_every', 'batch_size', 'val_samples'))
        if signature is not None and current != signature: raise ValueError('training schedules differ')
        signature = current
        if bool(opt['allow_plaintext_gelu_out_of_range']) != a.allow_plaintext_gelu_out_of_range: raise ValueError('GeLU range policy differs from training')
    if a.visited_steps > min(options[m]['steps'] for m in roots): raise ValueError('visited steps exceed training budget')
    indices = visited_indices(data, a.visited_steps, options['baseline']['batch_size'])
    chosen = balanced_subset(indices, archive['train'][1], a.train_samples)
    target = CifarData(archive, 'train', chosen)
    spec_basis = ZOConfig(rank=options['baseline']['rank'], seed=options['baseline']['seed'])
    b = basis(spec['patches'] * spec['dim'], spec_basis).to(device)
    encoder = LocalEncoder('mae', FEATURE_DIM, device, a.mae_path, seed=options['image-zo']['seed'] + 44)
    mae_file = next((x for x in (Path(a.mae_path)/'model.safetensors', Path(a.mae_path)/'pytorch_model.bin') if x.is_file()), None)
    if mae_file is None: raise ValueError('MAE weights not found')
    hashes = {'head_sha256': file_sha256(a.head), 'fixed_student_sha256': file_sha256(a.fixed_student),
              'archive_sha256': archive['archive_sha256'], 'split_sha256': file_sha256(a.splits)}
    mae_hash, config_hash = file_sha256(mae_file), file_sha256(Path(a.mae_path)/'config.json')
    result = {}
    for method, directory in roots.items():
        opt = options[method]
        validation = {0: reports[method]['initial']}
        validation.update({r['step']: r['evaluation'] for r in reports[method]['steps'] if 'evaluation' in r})
        ops = AccuracyPlainOps(params, a.allow_plaintext_gelu_out_of_range)
        records = []
        for step in a.steps:
            if step not in validation: raise ValueError(f'{method}: no validation metric for step {step}')
            if step == 0:
                weights = (torch.zeros(opt['rank'], dtype=torch.float64) if method == 'baseline' else initial_parameters(opt['seed'])).to(device)
                checkpoint_hash = None
            else:
                path = Path(directory)/f'step-{step:06d}.pt'
                state = torch.load(path, map_location='cpu', weights_only=True)
                if state.get('format') != 'cifar10_transfer_prompt_v1' or state.get('method') != method or state.get('step') != step: raise ValueError('checkpoint metadata mismatch')
                if any(state[k] != v for k, v in hashes.items()) or state['options'] != opt: raise ValueError('checkpoint provenance mismatch')
                if method != 'baseline' and (state['mae_sha256'] != mae_hash or state['mae_config_sha256'] != config_hash): raise ValueError('MAE differs from training')
                weights = state['weights'].double().to(device)
                if weights.numel() != (opt['rank'] if method == 'baseline' else PARAMS) or not torch.isfinite(weights).all(): raise ValueError('invalid prompt weights')
                checkpoint_hash = file_sha256(path)
            emit('diagnostic_started', method=method, step=step, samples=len(target))
            train = evaluate(method, spec, ops, PlainDecoderOps(), None if method == 'baseline' else encoder,
                             target, weights, b if method == 'baseline' else None, opt['prompt_scale'], device, a.batch_size)
            records.append({'step': step, 'training_subset': train, 'validation': validation[step], 'checkpoint_sha256': checkpoint_hash})
            emit('diagnostic_complete', method=method, **records[-1])
        result[method] = {'records': records, 'range_audit': ops.range_report()}
    report = {'status': 'evaluated_plaintext', 'train_subset_samples': len(target), 'visited_steps': a.visited_steps,
              'subset_class_counts': [sum(int(archive['train'][1][i]) == c for i in chosen) for c in range(10)],
              'train_subset_indices': chosen, 'hashes': hashes, 'methods': result,
              'note': 'Train subset was visited by all methods over the specified initial steps; early checkpoints need not have seen every sample. Validation metrics are copied from the original reports. Test is not loaded. Plaintext only.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    emit('complete', report=str(out))


if __name__ == '__main__': main()
