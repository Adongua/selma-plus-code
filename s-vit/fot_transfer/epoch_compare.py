"""Plaintext CIFAR transfer: matched FoT, image ZO, PT and full fine-tuning.

Training never loads the official test split. Test is a separate explicit command.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from hattention.student_checkpoint import file_sha256
from training.core import ZOConfig, PlainAdam, basis, direction
from training.model import forward
from fot_explore.conditioning import LocalEncoder
from fot_image.decoder import FEATURE_DIM, PlainDecoderOps, initial_parameters, rademacher
from fot_image.plaintext_ops import AccuracyPlainOps
from .common import load_data, load_head, source_model, install_head
from .data import CifarData
from .train import emit, logits_for


def epoch_batches(data, epoch, batch_size, seed):
    """Each party's examples appear once, with independent deterministic shuffles."""
    queues = []
    for owner in range(3):
        dataset = data[f'party-{owner}']
        gen = torch.Generator().manual_seed(seed + epoch*17 + owner)
        order = torch.randperm(len(dataset), generator=gen).tolist()
        queues.append([order[i:i+batch_size] for i in range(0, len(order), batch_size)])
    for batch_id in range(max(map(len, queues))):
        for owner in range(3):
            if batch_id >= len(queues[owner]): continue
            items = [data[f'party-{owner}'].sample(i) for i in queues[owner][batch_id]]
            yield owner, torch.stack([x for x, _ in items]), torch.tensor([y for _, y in items], dtype=torch.long)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['train', 'test'], required=True)
    p.add_argument('--method', choices=['fot', 'image-zo', 'pt', 'sft'], required=True)
    for name in ('archive', 'splits', 'checkpoint', 'fixed-student', 'mpcvit-root', 'head', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--mae-path')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--eval-batch-size', type=int, default=16)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--max-batches-per-epoch', type=int, default=0, help='smoke only; zero uses all batches')
    p.add_argument('--validation-max-samples', type=int, default=0, help='smoke only; zero uses the full validation set')
    p.add_argument('--seed', type=int, default=20261107)
    p.add_argument('--sigma', type=float, default=.1)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--epsilon', type=float, default=.01)
    p.add_argument('--momentum', type=float, default=.9)
    p.add_argument('--prompt-scale', type=float, default=.1)
    p.add_argument('--allow-plaintext-gelu-out-of-range', action='store_true')
    p.add_argument('--selected-checkpoint', help='required in test phase, selected using validation')
    a = p.parse_args()
    if min(a.epochs, a.batch_size, a.eval_batch_size, a.log_every) < 1 or min(a.max_batches_per_epoch, a.validation_max_samples) < 0 or not all(math.isfinite(x) and x > 0 for x in (a.sigma, a.lr, a.epsilon, a.prompt_scale)) or not 0 <= a.momentum < 1: p.error('invalid configuration')
    if a.method == 'image-zo' and not a.mae_path: p.error('image-zo requires --mae-path')
    if a.phase == 'test' and not a.selected_checkpoint: p.error('test phase requires --selected-checkpoint')
    if a.device == 'cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable')
    root = Path(a.output).resolve()
    if root.exists(): p.error('output already exists; choose a new directory')
    root.mkdir(parents=True, mode=0o700)
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    archive, split, data = load_data(a.archive, a.splits, include_test=a.phase == 'test')
    if a.validation_max_samples:
        if a.validation_max_samples > len(data['val']): raise ValueError('validation sample limit exceeds split')
        data['val'] = CifarData(archive, 'train', data['val'].indices[:a.validation_max_samples])
    head = load_head(a, split)
    if bool(head['allow_plaintext_gelu_out_of_range']) != a.allow_plaintext_gelu_out_of_range: raise ValueError('head and run GeLU range policies differ')
    spec, params = install_head(*source_model(a), head)
    params = {k: v.detach().to(device).requires_grad_(a.method == 'sft') for k, v in params.items()}
    ops = AccuracyPlainOps(params, a.allow_plaintext_gelu_out_of_range)
    dec = PlainDecoderOps()
    cfg = ZOConfig(rank=4, sigma=a.sigma, lr=a.lr, epsilon=a.epsilon, seed=a.seed).validate()
    b = basis(spec['patches']*spec['dim'], cfg).to(device) if a.method == 'fot' else None
    fot = PlainAdam(cfg) if a.method == 'fot' else None
    theta = initial_parameters(a.seed).to(device) if a.method == 'image-zo' else None
    velocity = torch.zeros_like(theta) if theta is not None else None
    prompt = torch.nn.Parameter(torch.zeros(spec['patches']*spec['dim'], dtype=torch.float64, device=device)) if a.method == 'pt' else None
    optimizer = (torch.optim.Adam([prompt], lr=a.lr) if prompt is not None else
                 torch.optim.Adam(list(params.values()), lr=a.lr) if a.method == 'sft' else None)
    encoder = LocalEncoder('mae', FEATURE_DIM, device, a.mae_path, seed=a.seed+44) if a.method == 'image-zo' else None
    hashes = {'head_sha256': file_sha256(a.head), 'fixed_student_sha256': file_sha256(a.fixed_student),
              'source_checkpoint_sha256': file_sha256(a.checkpoint), 'split_sha256': file_sha256(a.splits),
              'archive_sha256': archive['archive_sha256']}
    if encoder:
        mae_file = next((f for f in (Path(a.mae_path)/'model.safetensors', Path(a.mae_path)/'pytorch_model.bin') if f.is_file()), None)
        if mae_file is None: raise ValueError('MAE weights missing')
        hashes.update(mae_sha256=file_sha256(mae_file), mae_config_sha256=file_sha256(Path(a.mae_path)/'config.json'))

    def predict(x, feature=None, weights=None):
        if a.method == 'image-zo': return logits_for('image-zo', spec, ops, dec, x, feature, theta if weights is None else weights, None, a.prompt_scale)
        if a.method == 'fot': value = b.matmul(fot.z.to(device) if weights is None else weights)
        elif a.method == 'pt': value = prompt
        else: value = torch.zeros(spec['patches']*spec['dim'], dtype=torch.float64, device=device)
        return forward(spec, ops, x, value)

    def evaluate(dataset, phase, epoch):
        n = correct = 0; total = 0.
        with torch.no_grad():
            for x, y in dataset.chunks(a.eval_batch_size):
                x, y = x.to(device), y.to(device)
                logits = predict(x, encoder(x) if encoder else None)
                if not torch.isfinite(logits).all(): raise ArithmeticError('nonfinite logits')
                total += F.cross_entropy(logits, y, reduction='sum').item(); correct += (logits.argmax(-1)==y).sum().item(); n += y.numel()
                if n % (a.eval_batch_size*50) == 0 or n == len(dataset): emit('evaluation_progress', phase=phase, epoch=epoch, samples=n, total=len(dataset))
        return {'loss': total/n, 'top1': correct/n, 'samples': n}

    if a.phase == 'test':
        state = torch.load(a.selected_checkpoint, map_location='cpu', weights_only=True)
        if state['format'] != 'epoch_compare_v1' or state['method'] != a.method or state['hashes'] != hashes: raise ValueError('checkpoint provenance mismatch')
        saved = state['options']
        for key in ('seed', 'prompt_scale', 'allow_plaintext_gelu_out_of_range'):
            if saved[key] != getattr(a, key): raise ValueError('checkpoint option mismatch: '+key)
        if a.method == 'fot': fot.z = state['fot_z'].cpu().double()
        elif a.method == 'image-zo': theta = state['theta'].to(device)
        elif a.method == 'pt': prompt.data.copy_(state['prompt'].to(device))
        else:
            for name, value in state['params'].items(): params[name].data.copy_(value.to(device))
        result = {'phase': 'test', 'method': a.method, 'checkpoint': str(Path(a.selected_checkpoint).resolve()),
                  'checkpoint_sha256': file_sha256(a.selected_checkpoint), 'epoch': state['epoch'],
                  'test': evaluate(data['test'], 'test', state['epoch']), 'hashes': hashes,
                  'range_audit': ops.range_report(), 'note': 'Plaintext; previously inspected test results are not a fresh independent final evaluation.'}
        (root/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        emit('complete', **result)
        return

    emit('training_started', method=a.method, party_sizes=[len(data[f'party-{i}']) for i in range(3)], epochs=a.epochs)
    initial = evaluate(data['val'], 'validation', 0)
    rows = []; step = 0; best = None
    for epoch in range(1, a.epochs+1):
        n = 0; sum_loss = 0.
        for batch_index, (owner, x, y) in enumerate(epoch_batches(data, epoch, a.batch_size, a.seed)):
            if a.max_batches_per_epoch and batch_index >= a.max_batches_per_epoch: break
            x, y = x.to(device), y.to(device); step += 1
            feature = encoder(x) if encoder else None
            if a.method in ('fot', 'image-zo'):
                u = direction(step, cfg).to(device) if a.method == 'fot' else rademacher(step, a.seed).to(device)
                center = fot.z.to(device) if fot else theta + a.momentum*velocity
                with torch.no_grad():
                    plus = F.cross_entropy(predict(x, feature, center+a.sigma*u), y).item()
                    minus = F.cross_entropy(predict(x, feature, center-a.sigma*u), y).item()
                if fot: fot.update(plus-minus, u)
                else:
                    c = (plus-minus)/(2*a.sigma)
                    if not math.isfinite(c) or abs(c)>8: raise ArithmeticError('ZO coefficient outside [-8,8]')
                    velocity = a.momentum*velocity - a.lr*c*u; theta = theta+velocity
                    if not torch.isfinite(theta).all() or theta.abs().max()>8: raise ArithmeticError('prompt parameters outside [-8,8]')
                # This diagnostic mean uses perturbed losses, not loss at the updated model.
                loss_value = (plus+minus)/2
            else:
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(predict(x), y)
                if not torch.isfinite(loss): raise ArithmeticError('nonfinite training loss')
                loss.backward()
                if not all(v.grad is None or torch.isfinite(v.grad).all() for group in optimizer.param_groups for v in group['params']): raise ArithmeticError('nonfinite gradient')
                optimizer.step(); loss_value = loss.item()
            n += y.numel(); sum_loss += loss_value*y.numel()
            if step % a.log_every == 0: emit('training_progress', epoch=epoch, step=step, owner=owner, samples=n, total=sum(len(data[f'party-{i}']) for i in range(3)), running_loss=sum_loss/n)
        validation = evaluate(data['val'], 'validation', epoch)
        row = {'epoch': epoch, 'step': step, 'training_loss': sum_loss/n, 'training_samples': n, 'validation': validation}
        rows.append(row)
        state = {'format': 'epoch_compare_v1', 'method': a.method, 'epoch': epoch, 'step': step, 'options': vars(a), 'hashes': hashes, 'validation': validation}
        if fot: state.update(fot_z=fot.z.cpu(), fot_state=fot.state())
        if theta is not None: state.update(theta=theta.cpu(), velocity=velocity.cpu())
        if prompt is not None: state['prompt'] = prompt.detach().cpu()
        if a.method == 'sft': state['params'] = {k: v.detach().cpu() for k, v in params.items()}
        if optimizer: state['optimizer'] = optimizer.state_dict()
        path = root/f'epoch-{epoch:04d}.pt'; torch.save(state, path)
        if best is None or (-validation['top1'], validation['loss'], epoch) < (-best['top1'], best['loss'], best['epoch']):
            best = {'epoch': epoch, 'top1': validation['top1'], 'loss': validation['loss'], 'path': str(path), 'sha256': file_sha256(path)}
        emit('epoch_complete', **row, best=best)
        result = {'phase': 'train', 'status': 'running', 'method': a.method, 'options': vars(a), 'hashes': hashes, 'initial_validation': initial, 'epochs': rows, 'best': best, 'range_audit': ops.range_report(), 'note': 'Training loss is mean of plus/minus query losses for ZO and pre-update CE for PT/SFT; not directly identical metrics. Plaintext only.'}
        (root/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    result['status'] = 'completed'; (root/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    emit('complete', report=str(root/'report.json'), best=best)


if __name__ == '__main__': main()
