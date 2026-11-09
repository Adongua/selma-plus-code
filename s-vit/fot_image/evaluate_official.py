"""Compare frozen hAttention and saved prompts on the same Tiny-ImageNet validation images."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.core import ZOConfig, basis
from training.model import fixture, load_backbone, forward
from fot_explore.conditioning import LocalEncoder
from hattention.search_data import SearchDataset, collate_search
from hattention.search_mixer import balanced_items
from hattention.student_checkpoint import file_sha256
from hattention.validate_teacher import validation_items
from .decoder import FEATURE_DIM, PARAMS, PlainDecoderOps, image_prompt, initial_parameters, perturb_image
from .plaintext_ops import AccuracyPlainOps


def load_prompt(path, kind, source_path, device):
    artifact = torch.load(path, map_location='cpu', weights_only=True)
    options = artifact.get('config' if kind == 'baseline' else 'options')
    field = 'z' if kind == 'baseline' else 'theta'
    if not isinstance(options, dict) or field not in artifact:
        raise ValueError(f'invalid {kind} checkpoint: {path}')
    if options.get('fixed_student') and Path(options['fixed_student']).resolve() != Path(source_path).resolve():
        raise ValueError(f'{path} was trained with another fixed-student path')
    value = artifact[field]
    if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
        raise ValueError(f'non-finite or missing prompt in {path}')
    if kind == 'baseline' and value.numel() != int(options['rank']):
        raise ValueError(f'invalid latent dimension in {path}')
    if kind == 'image' and value.numel() != PARAMS:
        raise ValueError(f'invalid CNN parameter dimension in {path}')
    return {'kind':kind, 'value':value.to(device).double(), 'options':options,
            'path':str(Path(path).resolve()), 'sha256':file_sha256(path),
            'step':int(artifact.get('step', 0))}


def initial_image_variant(source, device):
    """Recreate the original CNN; trained values in source are never reused."""
    seed = int(source['options']['seed'])
    return {'kind':'image', 'value':initial_parameters(seed).to(device),
            'prompt_scale':float(source['options']['prompt_scale'])}


def evaluate_batches(spec, params, batches, variants, encoder, device, allow_gelu, progress_every=0):
    ops = {name:AccuracyPlainOps(params, allow_gelu) for name in variants}
    decoder = PlainDecoderOps()
    results = {name:{'samples':0, 'correct_top1':0, 'correct_top5':0, 'loss_sum':0.,
                     'reference_disagreements':0, 'reference_correct_variant_wrong':0,
                     'reference_wrong_variant_correct':0} for name in variants}
    last_progress = 0
    with torch.no_grad():
        for images, labels in batches:
            x, y = images.to(device).double(), labels.to(device)
            if x.shape[0] != y.shape[0]: raise ValueError('inconsistent batch shape')
            feature = encoder(x) if encoder else None
            zero = torch.zeros(spec['patches']*spec['dim'], dtype=torch.float64, device=device)
            predictions = {}
            for name, variant in variants.items():
                prompt = zero
                inputs = x
                if variant['kind'] == 'baseline':
                    prompt = variant['basis'].matmul(variant['value'])
                elif variant['kind'] == 'image':
                    image_delta = image_prompt(feature, variant['value'], decoder, spec['image_shape'])
                    inputs = perturb_image(x, image_delta, variant['prompt_scale'], spec['fixture'])
                logits = forward(spec, ops[name], inputs, prompt)
                if not torch.isfinite(logits).all(): raise ArithmeticError(f'nonfinite logits: {name}')
                prediction = logits.argmax(-1)
                row = results[name]
                row['samples'] += y.numel()
                row['loss_sum'] += F.cross_entropy(logits, y, reduction='sum').item()
                row['correct_top1'] += (prediction == y).sum().item()
                row['correct_top5'] += (logits.topk(min(5, logits.shape[-1]), dim=-1).indices == y[:,None]).any(-1).sum().item()
                predictions[name] = prediction
            ref = predictions['no_prompt']
            for name, pred in predictions.items():
                results[name]['reference_disagreements'] += (pred != ref).sum().item()
                results[name]['reference_correct_variant_wrong'] += ((ref == y) & (pred != y)).sum().item()
                results[name]['reference_wrong_variant_correct'] += ((ref != y) & (pred == y)).sum().item()
            count = results['no_prompt']['samples']
            if progress_every and count-last_progress >= progress_every:
                print(json.dumps({'event':'evaluation_progress', 'samples':count}),flush=True)
                last_progress = count
    for name, row in results.items():
        if not row['samples']: raise ValueError('empty evaluation')
        row['top1'] = row['correct_top1']/row['samples']
        row['top5'] = row['correct_top5']/row['samples']
        row['loss'] = row.pop('loss_sum')/row['samples']
        row['plaintext_range_audit'] = ops[name].range_report()
        if variants[name].get('path'):
            row.update({key:variants[name][key] for key in ('path','sha256','step')})
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True); p.add_argument('--fixed-student', required=True)
    p.add_argument('--mpcvit-root', required=True); p.add_argument('--dataset-root', required=True)
    p.add_argument('--mae-path', required=True)
    p.add_argument('--baseline-checkpoints', nargs='+', default=[])
    p.add_argument('--image-checkpoints', nargs='+', default=[])
    p.add_argument('--image-initial-from', help='recreate initial CNN using the seed and settings stored in this image checkpoint')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--max-samples', type=int, default=0, help='0: all official validation images; otherwise balanced subset')
    p.add_argument('--seed', type=int, default=20261030)
    p.add_argument('--progress-every', type=int, default=100)
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--allow-plaintext-gelu-out-of-range', action='store_true')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if a.batch_size<1 or a.max_samples<0 or a.seed<0 or a.progress_every<0: p.error('invalid batch, sample, seed or progress setting')
    if a.device=='cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable')
    output=Path(a.output).resolve()
    if output.exists(): p.error('output exists; choose a new path')
    root=Path(a.dataset_root).resolve()
    classes=sorted((root/'wnids.txt').read_text().splitlines())
    if len(classes)!=200 or len(set(classes))!=200: raise ValueError('expected 200 unique class IDs')
    artifact=torch.load(a.fixed_student,map_location='cpu',weights_only=True)
    saved_classes=artifact.get('distillation_settings',{}).get('class_order')
    if saved_classes is not None and saved_classes!=classes: raise ValueError('dataset class order differs from distillation')
    items=validation_items(root,classes)
    if a.max_samples>len(items): p.error('max samples exceeds validation size')
    if a.max_samples: items=balanced_items(items,a.max_samples,a.seed)
    digest=hashlib.sha256()
    for path,label in items: digest.update(f'{label}\t{Path(path).relative_to(root)}\n'.encode())
    device=torch.device(a.device)
    torch.set_num_threads(1)
    spec, params=load_backbone(a.checkpoint,a.mpcvit_root,a.fixed_student)
    params={key:value.to(device) for key,value in params.items()}
    variants={'no_prompt':{'kind':'none'}}
    for path in a.baseline_checkpoints:
        v=load_prompt(path,'baseline',a.fixed_student,device)
        config=ZOConfig(rank=int(v['options']['rank']),seed=int(v['options']['seed']))
        v['basis']=basis(spec['patches']*spec['dim'],config).to(device)
        name=f'baseline_step_{v["step"]}'
        if name in variants: raise ValueError(f'duplicate checkpoint step: {name}')
        variants[name]=v
    configs=[]
    for path in a.image_checkpoints:
        v=load_prompt(path,'image',a.fixed_student,device)
        v['prompt_scale']=float(v['options']['prompt_scale'])
        configs.append((int(v['options']['seed']),v['options']['encoder']))
        name=f'image_step_{v["step"]}'
        if name in variants: raise ValueError(f'duplicate checkpoint step: {name}')
        variants[name]=v
    initial_source = None
    if a.image_initial_from:
        source=load_prompt(a.image_initial_from,'image',a.fixed_student,device)
        seed=int(source['options']['seed'])
        configs.append((seed,source['options']['encoder']))
        name='image_initial'
        variants[name]=initial_image_variant(source, device)
        initial_source={'checkpoint':source['path'], 'checkpoint_sha256':source['sha256'],
                        'seed':seed, 'prompt_scale':variants[name]['prompt_scale'],
                        'note':'Initial parameters regenerated from the current code and saved seed; not loaded from the trained theta.'}
    if configs and (len(set(configs))!=1 or configs[0][1]!='mae'):
        raise ValueError('CNN checkpoints require same MAE projection configuration')
    encoder=LocalEncoder('mae',FEATURE_DIM,device,a.mae_path,seed=configs[0][0]+44) if configs else None
    loader=DataLoader(SearchDataset(root,items),batch_size=a.batch_size,shuffle=False,collate_fn=collate_search,num_workers=0)
    batches=((x,y) for x,_,y in loader)
    print(json.dumps({'event':'evaluation_started','samples':len(items),'batch_size':a.batch_size,'variants':list(variants)}),flush=True)
    metrics=evaluate_batches(spec,params,batches,variants,encoder,device,a.allow_plaintext_gelu_out_of_range,a.progress_every)
    report={'status':'evaluated_plaintext_not_security_validated','scope':'official validation' if not a.max_samples else 'balanced official validation subset',
            'samples':len(items),'manifest_sha256':digest.hexdigest(), 'batch_size':a.batch_size,
            'fixed_student_sha256':file_sha256(a.fixed_student),'source_checkpoint_sha256':file_sha256(a.checkpoint),
            'mae_path':str(Path(a.mae_path).resolve()),'image_initialization_source':initial_source,'metrics':metrics,
            'note':'All variants use the same plaintext training forward including g2 GeLU; not directly identical to evaluate_fixed; no secure accuracy claim. Official validation was previously used for model selection. Checkpoint provenance relies on recorded paths, not a saved weight hash.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'event':'evaluation_complete','report':str(output),'metrics':metrics}),flush=True)


if __name__=='__main__': main()
