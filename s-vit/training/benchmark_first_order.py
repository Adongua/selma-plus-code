"""3PC standard ViT SFT/PT step benchmark with private local labels.

Uses CrypTen autograd and a bounded, fixed-iteration secret Adam. Online
figures are instrumented inline estimates, not precomputed offline replay.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .core import ZOConfig
from .run import public_owner_error, write_json
from .standard_data import StandardPartyData
from .standard_vit import extract_teacher, fixture, forward


def load_source(options):
    if options['model_source'] == 'standard-vitlite-random':
        from .standard_vitlite import fixture as lite_fixture, random_backbone
        spec, params = (lite_fixture() if options['fixture'] else
                        random_backbone(options['mpcvit_root'], options['model_seed']))
        return spec, params, list(range(spec['classes'])), {
            'kind': 'public_fixture' if options['fixture'] else 'random_vitlite_standard_softmax',
            'initialization_seed': options['model_seed'], 'weights_pretrained': False,
            'attention': 'standard_softmax', 'accuracy_scope': 'cost benchmark only'}
    if options['fixture']:
        spec, params = fixture()
        return spec, params, list(range(spec['classes'])), {'kind': 'public_fixture'}
    from hattention.teacher_guidance import load_teacher
    teacher, meta = load_teacher(options['teacher_dir'], options['dataset_roots'][0], mapping=options['mapping'])
    spec, params = extract_teacher(teacher.teacher)
    return spec, params, teacher.permutation.tolist(), meta


def worker(rank, rendezvous, options, config):
    os.environ.update(RANK=str(rank), WORLD_SIZE='3', RENDEZVOUS=rendezvous, DISTRIBUTED_BACKEND='gloo')
    device = torch.device(f'cuda:{rank}' if options['device'] == 'cuda' else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    import crypten
    from hattention.experimental_crypten_truncation import experimental_truncation
    from .first_order_adam import SecretVectorAdam
    from .secure import open_to
    from .step_benchmark import StepBenchmark, aggregate_stages
    from .trace import ProgressTrace
    from .standard_vitlite import forward as lite_forward
    from .data import PartyData
    forward_model = lite_forward if options['model_source'] == 'standard-vitlite-random' else forward
    crypten.cfg.encoder.precision_bits = options['precision_bits']
    crypten.cfg.functions.sqrt_nr_iters = options['sqrt_iters_ln']
    crypten.init(device=device)
    crypten.cfg.communicator.verbose = True
    meter = None
    event_lock = threading.Lock()
    def event(name, **values):
        if rank == 0:
            line = json.dumps({'event': name, **values}, allow_nan=False)
            with event_lock:
                print(line, flush=True)
                with (Path(options['output'])/'progress.jsonl').open('a', encoding='utf-8') as output:
                    output.write(line+'\n')
    def emit(record):
        event(record.pop('event'), **record)
    progress = ProgressTrace(rank == 0, device, options['heartbeat_seconds'], emit)
    try:
        source = public_owner_error(0, lambda: load_source(options))
        package = [None]
        if rank == 0:
            spec, values, permutation, meta = source
            package[0] = {'spec': spec, 'shapes': {name: list(value.shape) for name, value in values.items()},
                          'permutation': permutation, 'metadata': meta}
        dist.broadcast_object_list(package, src=0)
        spec, shapes = package[0]['spec'], package[0]['shapes']
        permutation = torch.tensor(package[0]['permutation'], dtype=torch.long, device=device)
        data = None
        for owner in range(3):
            local = public_owner_error(owner, lambda: (
                PartyData(owner, spec, (options['dataset_roots'] or [None]*3)[owner],
                          (options['manifests'] or [None]*3)[owner])
                if options['model_source'] == 'standard-vitlite-random' else
                StandardPartyData(owner, spec, (options['dataset_roots'] or [None]*3)[owner],
                                  (options['manifests'] or [None]*3)[owner], options['teacher_dir'])))
            if owner == rank:
                data = local
        event('model_sharing_started', tensors=len(shapes))
        params = {}
        for index, (name, shape) in enumerate(shapes.items(), 1):
            value = values[name].to(device) if rank == 0 else torch.zeros(shape, device=device)
            params[name] = crypten.cryptensor(value, src=0, device=device,
                                             precision=options['precision_bits'],
                                             requires_grad=options['method'] == 'sft')
            if index % 10 == 0 or index == len(shapes):
                event('model_sharing_progress', completed=index, total=len(shapes))
        if rank == 0:
            reference_values = None if options['skip_numerical_audit'] else values
            del source, values
        prompt = (crypten.cryptensor(torch.zeros(spec['patches'], spec['dim'], device=device), src=0,
                                     precision=options['precision_bits'], requires_grad=True)
                  if options['method'] == 'pt' else None)
        names = list(params) if options['method'] == 'sft' else ['prompt']
        shapes_train = [shapes[name] for name in names] if prompt is None else [[spec['patches'], spec['dim']]]
        parameter_count = sum(int(torch.tensor(s).prod().item()) for s in shapes_train)
        event('model_sharing_complete', method=options['method'], trainable_parameters=parameter_count)
        meter = StepBenchmark(crypten, device, SimpleNamespace(sent_bytes=0))
        meter.install()
        optimizer_span = progress.optimizer_span if options['optimizer_progress'] else lambda site: nullcontext()
        optimizer = SecretVectorAdam(config, allow_numerical_failure=options['cost_only'], span=optimizer_span)
        records = []
        with experimental_truncation('carry_save'):
            if options['skip_numerical_audit']:
                audit = [{'passed': None, 'skipped': True, 'scope': 'not performed'}]
                event('numerical_audit_skipped', reason='explicit cost-only option')
            else:
                # The owner validates one public/authorized fixture against the
                # corresponding plaintext formula before any timing is accepted.
                event('numerical_audit_started')
                audit_item = public_owner_error(0, lambda: data.batch(0, options['batch_size'], True))
                audit_images = (audit_item[0] if rank == 0 else
                                torch.zeros(options['batch_size'], *spec['image_shape']))
                audit_secret = crypten.cryptensor(audit_images.to(device), src=0,
                                                  precision=options['precision_bits'], device=device)
                progress.location = 'numerical_audit'
                with progress.heartbeat(), crypten.no_grad():
                    audit_output = forward_model(spec, params, audit_secret, prompt,
                                           softmax_scale=options['softmax_scale'], gelu_bound=options['gelu_bound'],
                                           gelu_method=options['gelu_method'])
                audit_opened = open_to(audit_output, 0)
                audit = [None]
                if rank == 0:
                    with torch.no_grad():
                        plain_prompt = torch.zeros(spec['patches'], spec['dim'], dtype=torch.float64) if prompt is not None else None
                        reference = forward_model(spec, reference_values, audit_images.double(), plain_prompt,
                                            softmax_scale=options['softmax_scale'], gelu_bound=options['gelu_bound'],
                                            gelu_method=options['gelu_method'])
                    difference = (audit_opened.cpu() - reference).abs()
                    finite = bool(torch.isfinite(difference).all())
                    max_error = float(difference.max()) if finite else None
                    details = {'finite': finite, 'max_abs_error': max_error,
                               'passed': finite and max_error <= options['max_audit_error'],
                               'scope': 'one owner batch, plaintext model reference; not a security proof',
                               'gelu_method': options['gelu_method']}
                    private_root = Path(options['output'])/'party-0'
                    private_root.mkdir(mode=0o700)
                    write_json(private_root/'numerical-audit.json', details)
                    audit[0] = {'passed': details['passed'], 'scope': details['scope']}
                    if options['fixture']:
                        audit[0]['max_abs_error'] = max_error
                    del reference_values
                dist.broadcast_object_list(audit, src=0)
                event('numerical_audit_complete', **audit[0])
                if not audit[0]['passed'] and not options['cost_only']:
                    if rank == 0:
                        write_json(Path(options['output'])/'audit.json', audit[0])
                    raise ArithmeticError('standard ViT numerical audit failed; benchmark was not measured')
                if not audit[0]['passed']:
                    event('numerical_audit_failed_continuing_cost_only')
            for step in range(1, options['steps']+1):
                owner = (step-1) % 3
                event('step_started', step=step, owner=owner)
                progress.context = {'step': step, 'owner': owner}
                progress.location = 'forward'
                before = meter.start()
                item = public_owner_error(owner, lambda: data.batch((step-1)//3, options['batch_size'], options['fixed_batch']))
                if rank == owner:
                    images, labels = item
                else:
                    images = torch.zeros(options['batch_size'], *spec['image_shape'])
                    labels = None
                images = crypten.cryptensor(images.to(device), src=owner, precision=options['precision_bits'], device=device)
                with progress.heartbeat():
                    logits = forward_model(spec, params, images, prompt,
                                     softmax_scale=options['softmax_scale'], gelu_bound=options['gelu_bound'],
                                     gelu_method=options['gelu_method'])
                if not logits.requires_grad:
                    raise RuntimeError('logits detached from the gradient graph')
                opened = open_to(logits, owner)
                def local_gradient():
                    if not torch.isfinite(opened).all():
                        raise ArithmeticError('nonfinite owner logits')
                    ordered = opened.index_select(1, permutation)
                    probabilities = ordered.softmax(dim=1)
                    probabilities[torch.arange(labels.numel(), device=device), labels.to(device)] -= 1
                    probabilities /= labels.numel()
                    result = torch.zeros_like(opened)
                    result.index_add_(1, permutation, probabilities)
                    if not torch.isfinite(result).all():
                        raise ArithmeticError('nonfinite owner logit gradient')
                    return result
                gradient = public_owner_error(owner, local_gradient)
                if rank != owner:
                    gradient = torch.zeros(options['batch_size'], spec['classes'], device=device)
                shared_gradient = crypten.cryptensor(gradient, src=owner, precision=options['precision_bits'], device=device)
                forward_cost = meter.finish(before)
                event('forward_complete', step=step)

                progress.location = 'backward'
                before = meter.start()
                with progress.heartbeat():
                    logits.backward(shared_gradient)
                backward_cost = meter.finish(before)
                event('backward_complete', step=step)

                progress.location = 'optimizer'
                before = meter.start()
                with progress.heartbeat(), crypten.no_grad():
                    selected = [params[name] for name in names] if prompt is None else [prompt]
                    if any(value.grad is None for value in selected):
                        missing = [name for name, value in zip(names, selected) if value.grad is None]
                        raise RuntimeError('missing parameter gradients: ' + ', '.join(missing))
                    with optimizer_span('pack_parameters'):
                        flat_grad = crypten.cat([value.grad.reshape(-1) for value in selected])
                        flat_values = crypten.cat([value.detach().reshape(-1) for value in selected])
                    updated = optimizer.update(flat_values, flat_grad)
                    with optimizer_span('unpack_parameters'):
                        offset = 0
                        for name, shape in zip(names, shapes_train):
                            count = int(torch.tensor(shape).prod().item())
                            value = updated.narrow(0, offset, count).reshape(shape).detach()
                            value.requires_grad = True
                            if name == 'prompt':
                                prompt = value
                            else:
                                params[name] = value
                            offset += count
                optimizer_cost = meter.finish(before)
                failures = [failure for failure in optimizer.numerical_failures if failure['step'] == step]
                if failures:
                    event('optimizer_numerical_failure_continuing_cost_only', step=step, sites=failures)
                costs = aggregate_stages({'forward': forward_cost, 'backward': backward_cost,
                                          'optimizer': optimizer_cost})
                records.append({'step': step, 'owner': owner, 'benchmark': costs})
                event('step_complete', step=step, owner=owner, benchmark=costs)
        if rank == 0:
            status = ('benchmark_completed_cost_only_not_numerically_validated' if options['cost_only'] else
                      'benchmark_completed_one_batch_audit_not_security_validated')
            report = {'status': status,
                      'method': 'MPC-' + options['method'].upper(), 'model': spec,
                      'source': package[0]['metadata'], 'trainable_parameters': parameter_count, 'audit': audit[0],
                      'options': options, 'config': asdict(config), 'steps': records,
                      'optimizer_numerical_failures': optimizer.numerical_failures,
                      'numeric_note': f"Q20 scaled softmax, CrypTen LayerNorm, {options['gelu_method']} GeLU; "
                                      + ('audit skipped; ' if options['skip_numerical_audit'] else 'one-batch forward audit; ')
                                      + 'gradients require further full-model validation',
                      'timing_note': 'inline provider subtraction estimate; synchronized instrumentation changes timing',
                      'communication_note': 'CrypTen bidirectional counter sum / 2; public control metadata excluded',
                      'privacy_note': 'owner receives own logits and shares its logit gradient; no privacy proof'}
            write_json(Path(options['output'])/'report.json', report)
            event('benchmark_complete', report=str(Path(options['output'])/'report.json'), status=report['status'])
    finally:
        if meter:
            meter.close()
        crypten.uninit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=['sft', 'pt'])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--fixture', action='store_true')
    parser.add_argument('--model-source', choices=['teacher', 'standard-vitlite-random'], default='teacher')
    parser.add_argument('--mpcvit-root', help='source tree for random standard ViTLite')
    parser.add_argument('--model-seed', type=int, default=20261107)
    parser.add_argument('--teacher-dir')
    parser.add_argument('--dataset-roots', nargs=3)
    parser.add_argument('--manifests', nargs=3)
    parser.add_argument('--mapping', choices=['first-name', 'strict-id'], default='first-name')
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--fixed-batch', action='store_true')
    parser.add_argument('--precision-bits', type=int, default=20)
    parser.add_argument('--softmax-scale', type=int, default=256)
    parser.add_argument('--sqrt-iters-ln', type=int, default=7)
    parser.add_argument('--gelu-bound', type=float, default=12.)
    parser.add_argument('--gelu-method', choices=['erf', 'chebyshev16'], default='erf')
    parser.add_argument('--cost-only', action='store_true',
                        help='continue after audit and Adam numerical check failures; results are diagnostic only')
    parser.add_argument('--skip-numerical-audit', action='store_true',
                        help='skip initial reference forward (requires --cost-only)')
    parser.add_argument('--optimizer-progress', action='store_true',
                        help='log synchronized rank-0 Adam stage wall times including preprocessing')
    parser.add_argument('--max-audit-error', type=float, default=.25,
                        help='maximum absolute logit error on the first owner batch')
    parser.add_argument('--heartbeat-seconds', type=float, default=30.)
    parser.add_argument('--lr', type=float, default=.03)
    parser.add_argument('--epsilon', type=float, default=.01)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.skip_numerical_audit and not args.cost_only:
        parser.error('--skip-numerical-audit requires --cost-only')
    if min(args.steps, args.batch_size, args.sqrt_iters_ln) < 1 or args.precision_bits != 20:
        parser.error('positive steps/batch/iterations and Q20 precision required')
    if (args.softmax_scale < 1 or args.softmax_scale & (args.softmax_scale-1)
            or any(not math.isfinite(x) or x <= 0 for x in
                   (args.gelu_bound, args.max_audit_error, args.heartbeat_seconds))):
        parser.error('positive power-of-two softmax scale and positive GeLU bound required')
    if not args.fixture and not all((args.dataset_roots, args.manifests)):
        parser.error('three dataset-roots and three manifests required')
    if not args.fixture and args.model_source == 'teacher' and not args.teacher_dir:
        parser.error('teacher-dir required for teacher model')
    if not args.fixture and args.model_source == 'standard-vitlite-random' and not args.mpcvit_root:
        parser.error('mpcvit-root required for random ViTLite')
    if args.device == 'cuda' and torch.cuda.device_count() < 3:
        parser.error('three visible CUDA devices required')
    config = ZOConfig(lr=args.lr, epsilon=args.epsilon).validate()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    options = vars(args).copy()
    options['output'] = str(root)
    write_json(root/'run.json', {'options': options, 'config': asdict(config)})
    with tempfile.TemporaryDirectory(prefix='rendezvous-', dir=root) as tmp:
        mp.spawn(worker, args=(Path(tmp, 'store').as_uri(), options, config), nprocs=3, join=True)


if __name__ == '__main__':
    main()
