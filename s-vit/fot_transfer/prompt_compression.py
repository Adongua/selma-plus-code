"""Evaluate spatial and low-rank approximations of a trained patch prompt."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from hattention.student_checkpoint import file_sha256
from training.model import forward
from fot_image.plaintext_ops import AccuracyPlainOps
from .common import install_head, load_data, load_head, source_model
from .data import CifarData
from .train import emit


def prompt_variants(prompt, grid):
    height, width = grid
    if prompt.ndim != 2 or prompt.shape[0] != height * width:
        raise ValueError('prompt shape does not match patch grid')
    result = {'no_prompt': torch.zeros_like(prompt), 'full_pt': prompt}
    result['spatial_mean'] = prompt.mean(dim=0, keepdim=True).expand_as(prompt)
    image = prompt.reshape(height, width, -1).permute(2, 0, 1).unsqueeze(0)
    for size in (2, 4):
        if size > min(grid):
            continue
        reduced = F.adaptive_avg_pool2d(image, (size, size))
        restored = F.interpolate(reduced, size=grid, mode='bilinear', align_corners=False)
        result[f'grid_{size}x{size}'] = restored.squeeze(0).permute(1, 2, 0).reshape_as(prompt)
    u, s, vh = torch.linalg.svd(prompt, full_matrices=False)
    for rank in (1, 2, 4, 8):
        if rank <= len(s):
            result[f'svd_rank_{rank}'] = (u[:, :rank] * s[:rank]) @ vh[:rank]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'splits', 'checkpoint', 'fixed-student', 'mpcvit-root', 'head', 'pt-checkpoint', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-samples', type=int, default=0, help='smoke only; 0 evaluates all validation samples')
    parser.add_argument('--allow-plaintext-gelu-out-of-range', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_samples < 0:
        parser.error('invalid batch size or sample limit')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    output = Path(args.output)
    if output.exists():
        parser.error('output already exists')
    torch.set_num_threads(1)
    archive, split, data = load_data(args.archive, args.splits)
    head = load_head(args, split)
    if bool(head['allow_plaintext_gelu_out_of_range']) != args.allow_plaintext_gelu_out_of_range:
        raise ValueError('head and run GeLU range policies differ')
    hashes = {'head_sha256': file_sha256(args.head), 'fixed_student_sha256': file_sha256(args.fixed_student),
              'source_checkpoint_sha256': file_sha256(args.checkpoint), 'split_sha256': file_sha256(args.splits),
              'archive_sha256': archive['archive_sha256']}
    state = torch.load(args.pt_checkpoint, map_location='cpu', weights_only=True)
    if state.get('format') != 'epoch_compare_v1' or state.get('method') != 'pt' or state.get('hashes') != hashes:
        raise ValueError('PT checkpoint provenance mismatch')
    if state['options']['allow_plaintext_gelu_out_of_range'] != args.allow_plaintext_gelu_out_of_range:
        raise ValueError('PT checkpoint GeLU range policy differs')
    spec, params = install_head(*source_model(args), head)
    device = torch.device(args.device)
    params = {name: value.detach().to(device) for name, value in params.items()}
    prompt = state['prompt'].double().reshape(spec['patches'], spec['dim']).to(device)
    if not torch.isfinite(prompt).all():
        raise ValueError('nonfinite PT prompt')
    variants = prompt_variants(prompt, tuple(spec['grid']))
    validation = data['val']
    if args.max_samples:
        if args.max_samples > len(validation):
            parser.error('sample limit exceeds validation size')
        validation = CifarData(archive, 'train', validation.indices[:args.max_samples])
    output.mkdir(parents=True, mode=0o700)
    metrics = {}
    with torch.no_grad():
        for name, candidate in variants.items():
            ops = AccuracyPlainOps(params, args.allow_plaintext_gelu_out_of_range)
            loss = correct = count = 0
            for x, y in validation.chunks(args.batch_size):
                x, y = x.to(device), y.to(device)
                logits = forward(spec, ops, x, candidate)
                if not torch.isfinite(logits).all():
                    raise ArithmeticError('nonfinite logits')
                loss += F.cross_entropy(logits, y, reduction='sum').item()
                correct += (logits.argmax(-1) == y).sum().item()
                count += len(y)
                if count % (args.batch_size * 50) == 0 or count == len(validation):
                    emit('evaluation_progress', variant=name, samples=count, total=len(validation))
            metrics[name] = {'loss': loss / count, 'top1': correct / count, 'correct': correct,
                             'samples': count, 'prompt_rms': candidate.square().mean().sqrt().item(),
                             'reconstruction_rmse': (candidate - prompt).square().mean().sqrt().item(),
                             'range_audit': ops.range_report()}
            emit('variant_complete', variant=name, **metrics[name])
    report = {'status': 'completed', 'scope': 'plaintext validation; PT-derived diagnostic',
              'pt_epoch': state['epoch'], 'pt_checkpoint_sha256': file_sha256(args.pt_checkpoint),
              'hashes': hashes, 'options': vars(args), 'metrics': metrics}
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    emit('complete', report=str(output / 'report.json'))


if __name__ == '__main__':
    main()
