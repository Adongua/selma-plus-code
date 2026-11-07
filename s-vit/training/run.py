"""CLI: python -m training.run --backend plaintext|3pc --fixture ..."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from .core import ZOConfig, PlainAdam, basis, direction, weights
from .data import PartyData
from .model import fixture, load_backbone, PlainOps, forward


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def load(options):
    return fixture() if options['fixture'] else load_backbone(options['checkpoint'], options['mpcvit_root'], options['fixed_student'])


def datasets(options, spec, parties):
    return {i: PartyData(i, spec, (options['dataset_roots'] or [None]*3)[i],
                        (options['manifests'] or [None]*3)[i]) for i in parties}


def report(options, config, spec, records, initial, final, top1=None):
    loss_checked = initial is not None and final is not None
    decreased = (final < initial-options['min_loss_drop']) if loss_checked else None
    return {'status': ('smoke_passed_not_security_validated' if decreased else
                       'loss_decrease_not_observed' if loss_checked else 'numerical_run_completed_loss_not_disclosed'),
            'backend': options['backend'], 'mode': options['mode'], 'config': asdict(config),
            'model': spec, 'weights': weights(options['weights']), 'steps': records,
            'initial_loss': initial, 'final_loss': final, 'loss_decreased': decreased,
            'top1_diagnostic_only': top1, 'prompt_injection': 'additive_patch_embeddings_no_extra_tokens',
            'public_fixture': options['fixture'], 'diagnostic_loss_disclosure': options['diagnostic_loss'] or options['fixture'],
            'public_numerical_failure_flags': True,
            'secret_forward_range_checks': options['backend'] == '3pc' and options['check_forward_ranges'],
            'ring3_truncation_preprocessing': options['ring3_truncation_preprocessing'],
            'scope': 'training diagnostic; no validation accuracy or privacy proof',
            'backbone_frozen': True, 'backbone_weights_secret_in_3pc': options['backend']=='3pc',
            'update_schedule': options['update_schedule'],
            'training_weights': (weights(options['weights']) if options['update_schedule']=='aggregate' else None),
            'diagnostic_weights': weights(options['weights']),
            'optimizer': 'Adam epsilon outside sqrt; secret v_hat projected nonnegative for fixed-point rounding',
            'timing': 'wall time includes preprocessing, checks, output delivery and diagnostic queries'}


@torch.no_grad()
def run_plain(options, config):
    device = torch.device(options['device'])
    spec, params = load(options)
    params = {k: v.to(device) for k, v in params.items()}
    ops = PlainOps(params)
    data = datasets(options, spec, range(3))
    b = basis(spec['patches']*spec['dim'], config).to(device)
    w = weights(options['weights'])
    adam = PlainAdam(config)
    def loss(z, step, fixed, parties=range(3)):
        losses, correct, total = [], 0, 0
        for party in parties:
            x, y = data[party].batch(step, options['batch_size'], fixed)
            logits = forward(spec, ops, x.to(device), b.matmul(z.to(device)))
            losses.append(F.cross_entropy(logits, y.to(device)).item())
            correct += (logits.argmax(-1).cpu() == y).sum().item()
            total += y.numel()
        return (sum(w[i]*v for i, v in zip(parties, losses)) if len(losses)>1 else losses[0]), correct/total
    initial, _ = loss(adam.z, 0, True)
    records = []
    for step in range(1, options['steps']+1):
        started = time.perf_counter()
        u = direction(step, config)
        owner = (step-1)%3 if options['update_schedule']=='round-robin' else None
        parties = [owner] if owner is not None else range(3)
        batch_step = (step-1)//3 if owner is not None else step-1
        plus, _ = loss(adam.z+config.sigma*u, batch_step, options['fixed_batch'], parties)
        minus, _ = loss(adam.z-config.sigma*u, batch_step, options['fixed_batch'], parties)
        adam.update(plus-minus, u)
        row = {'step': step, 'owner': owner, 'batch_index': batch_step}
        if step % options['eval_every'] == 0 or step == options['steps']:
            row['diagnostic_loss'], row['top1'] = loss(adam.z, 0, True)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        row['wall_seconds'] = time.perf_counter()-started
        records.append(row)
        print(json.dumps({'event': 'training_step', **row}), flush=True)
    final, top1 = loss(adam.z, 0, True)
    torch.save({'format': 'zo_prompt_plain_v1', **adam.state(), 'spec': spec}, Path(options['output'])/'prompt.pt')
    result = report(options, config, spec, records, initial, final, top1)
    write_json(Path(options['output'])/'report.json', result)


def public_owner_error(owner, function):
    """All parties fail coherently before the next collective, without data logging."""
    result, error = None, None
    if dist.get_rank() == owner:
        try:
            result = function()
        except Exception as exc:
            error = type(exc).__name__ + ': owner input/model preparation failed'
    errors = [error]
    dist.broadcast_object_list(errors, src=owner)
    if errors[0]:
        raise RuntimeError(errors[0])
    return result


@torch.no_grad()
def worker(rank, rendezvous, options, config, incoming, outgoing, dealer_connections):
    from .secure import SecretOps, SecretAdam, open_to, precision, raw_share
    from .trusted import pack, receive
    from hattention.experimental_crypten_truncation import experimental_truncation
    from .trace import ProgressTrace
    from .truncation_dealer import trusted_truncations
    device = torch.device(f'cuda:{rank}' if options['device'] == 'cuda' else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE='3', RENDEZVOUS=rendezvous, DISTRIBUTED_BACKEND='gloo')
    import crypten
    crypten.cfg.encoder.precision_bits = 17
    crypten.init(device=device)
    root = Path(options['output'])
    event_lock = threading.Lock()
    def emit(record):
        if rank == 0:
            line = json.dumps(record, allow_nan=False)
            with event_lock:
                print(line, flush=True)
                with (root/'progress.jsonl').open('a', encoding='utf-8') as output:
                    output.write(line+'\n')
    trace = ProgressTrace(options['debug_progress'] and rank == 0, device, options['heartbeat_seconds'], emit)
    def event(name, **fields):
        emit({'event': name, **fields})
    try:
        event('model_loading_started')
        setup_start = time.perf_counter()
        loaded = public_owner_error(0, lambda: load(options))
        package = [None]
        if rank == 0:
            spec, values = loaded
            if any(not torch.isfinite(v).all() or v.abs().max() > 64 for v in values.values()):
                # Signal before parameter sharing rather than leave peers blocked.
                package[0] = {'error': 'invalid frozen model values'}
            else:
                package[0] = {'spec': spec, 'shapes': {k: list(v.shape) for k, v in values.items()}}
        dist.broadcast_object_list(package, src=0)
        if 'error' in package[0]:
            raise ValueError(package[0]['error'])
        spec, shapes = package[0]['spec'], package[0]['shapes']
        event('model_sharing_started', tensors=len(shapes), seconds_since_load_start=time.perf_counter()-setup_start)
        params = {}
        for key, shape in shapes.items():
            value = values[key].to(device) if rank == 0 else torch.zeros(shape, device=device, dtype=torch.float64)
            params[key] = crypten.cryptensor(value, src=0, precision=17, device=device)
        if rank == 0:
            del loaded, values
        event('model_sharing_complete', tensors=len(shapes), seconds_since_load_start=time.perf_counter()-setup_start)
        # Each process opens only its own data manifest.
        data = None
        for owner in range(3):
            local = public_owner_error(owner, lambda: datasets(options, spec, [owner])[owner])
            if owner == rank:
                data = local
        ops = SecretOps(params, device, trace, check_forward_ranges=options['check_forward_ranges'])
        event('forward_range_checks_configured', enabled=options['check_forward_ranges'])
        event('ring3_truncation_preprocessing_configured', mode=options['ring3_truncation_preprocessing'])
        b = basis(spec['patches']*spec['dim'], config).to(device)
        w = weights(options['weights'])
        diagnostic = options['fixture'] or options['diagnostic_loss']
        records, local_records = [], []

        def share_batch(step, fixed, owners):
            shared = {}
            local_y = None
            for owner in owners:
                item = public_owner_error(owner, lambda: data.batch(step, options['batch_size'], fixed))
                if rank == owner:
                    x, local_y = item
                else:
                    x = torch.zeros(options['batch_size'], *spec['image_shape'], dtype=torch.float64)
                shared[owner] = crypten.cryptensor(x.to(device), src=owner, precision=17, device=device)
            return shared, local_y

        def owner_loss(z, batches, label, query, step):
            event('prompt_projection_started', step=step, query=query)
            trace.sync()
            projection_start = time.perf_counter()
            prompt = precision(z.matmul(b.t()), 17)
            trace.sync()
            event('prompt_projection_complete', step=step, query=query,
                  seconds_rank0=time.perf_counter()-projection_start)
            local_loss = None
            for owner in batches:
                trace.context = {'step': step, 'owner': owner, 'query': query}
                event('forward_started', step=step, owner=owner, query=query)
                trace.sync()
                trace_before = trace.totals.copy()
                start = time.perf_counter()
                with trace.heartbeat():
                    logits = forward(spec, ops, batches[owner], prompt)
                opened = open_to(logits, owner)
                def compute():
                    if not torch.isfinite(opened).all():
                        raise ArithmeticError('nonfinite authorized logits')
                    return F.cross_entropy(opened, label.to(device)).item()
                value = public_owner_error(owner, compute)
                if rank == owner:
                    local_loss = value
                trace.sync()
                timing = {}
                if trace.enabled:
                    timing = {'ring3_prepare_seconds_rank0': trace.totals['ring3_prepare']-trace_before['ring3_prepare'],
                              'range_check_seconds_rank0': trace.totals['range_check']-trace_before['range_check']}
                event('forward_complete', step=step, owner=owner, query=query,
                      seconds_rank0=time.perf_counter()-start, **timing)
            return local_loss

        def aggregate_local_loss(value):
            if not diagnostic:
                return None
            # Authorized smoke/diagnostic output, independent from secret delta.
            total = torch.tensor(value*w[rank], dtype=torch.float64)
            dist.all_reduce(total)
            return total.item()

        dealer_connection = dealer_connections[rank] if dealer_connections else None
        with experimental_truncation('carry_save'), trusted_truncations(dealer_connection):
            event('initial_evaluation_started')
            initial_start = time.perf_counter()
            adam = SecretAdam(config, device) if options['mode'] == 'secret' else None
            z = adam.z if adam else receive(outgoing[rank], config, device)
            evaluation, eval_y = share_batch(0, True, range(3))
            first = owner_loss(z, evaluation, eval_y, 'initial_evaluation', 0)
            initial = aggregate_local_loss(first)
            local_records.append({'step': 0, 'loss': first})
            event('training_started', initial_loss=initial, mode=options['mode'],
                  initial_evaluation_seconds_rank0=time.perf_counter()-initial_start)
            for step in range(1, options['steps']+1):
                started = time.perf_counter()
                owner = (step-1)%3 if options['update_schedule']=='round-robin' else None
                owners = [owner] if owner is not None else range(3)
                batch_step = (step-1)//3 if owner is not None else step-1
                event('step_started', step=step, owner=owner, batch_index=batch_step)
                trace.sync()
                share_start = time.perf_counter()
                batches, labels = share_batch(batch_step, options['fixed_batch'], owners)
                trace.sync()
                event('batch_shared', step=step, owner=owner, seconds_rank0=time.perf_counter()-share_start)
                u = direction(step, config).to(device)
                plus = owner_loss(z+config.sigma*u, batches, labels, 'plus', step)
                minus = owner_loss(z-config.sigma*u, batches, labels, 'minus', step)
                event('delta_sharing_started', step=step, owner=owner)
                trace.sync()
                delta_start = time.perf_counter()
                delta = None
                for data_owner in owners:
                    value = torch.tensor(plus-minus if rank == data_owner else 0., dtype=torch.float64, device=device)
                    weight = w[data_owner] if owner is None else 1.
                    term = crypten.cryptensor(value, src=data_owner, precision=24, device=device)*weight
                    delta = term if delta is None else delta+term
                trace.sync()
                event('delta_sharing_complete', step=step, owner=owner, seconds_rank0=time.perf_counter()-delta_start)
                event('adam_started', step=step, owner=owner)
                adam_start = time.perf_counter()
                if adam:
                    z = adam.update(delta, u)
                else:
                    # Each party sends ONLY its aggregated delta SHARE to T.
                    incoming.put((rank, step, pack(raw_share(delta))))
                    z = receive(outgoing[rank], config, device)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                event('adam_complete', step=step, owner=owner, seconds_rank0=time.perf_counter()-adam_start)
                row = {'step': step, 'owner': owner, 'batch_index': batch_step}
                if step % options['eval_every'] == 0 or step == options['steps']:
                    event('evaluation_started', step=step)
                    value = owner_loss(z, evaluation, eval_y, 'evaluation', step)
                    local_records.append({'step': step, 'loss': value})
                    row['diagnostic_loss'] = aggregate_local_loss(value)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                elapsed = torch.tensor(time.perf_counter()-started, dtype=torch.float64)
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                row['wall_seconds'] = elapsed.item()
                records.append(row)
                event('training_step', **row)
            final = records[-1]['diagnostic_loss']
            state = adam.state() if adam else {'z_share': raw_share(z).cpu(), 'precision': 24, 'step': options['steps']}
            private_root = root / f'party-{rank}'
            private_root.mkdir(mode=0o700)
            torch.save({'format': 'zo_prompt_share_v1', 'party': rank, 'spec': spec,
                        'config': asdict(config), **state}, private_root/'prompt-share.pt')
            write_json(private_root/'local-loss.json', local_records)
            if rank == 0:
                write_json(root/'report.json', report(options, config, spec, records, initial, final))
    finally:
        crypten.uninit()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['plaintext', '3pc'], required=True)
    p.add_argument('--mode', choices=['secret', 'trusted'], default='secret')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--fixture', action='store_true', help='PUBLIC mini-transformer overfit diagnostic')
    p.add_argument('--checkpoint')
    p.add_argument('--fixed-student')
    p.add_argument('--mpcvit-root')
    p.add_argument('--manifests', nargs=3)
    p.add_argument('--dataset-roots', nargs=3)
    p.add_argument('--weights', nargs=3, type=float, default=[1, 1, 1])
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--fixed-batch', action='store_true', help='overfit diagnostic, not generalization evaluation')
    p.add_argument('--diagnostic-loss', action='store_true', help='explicitly authorize aggregate evaluation loss disclosure')
    p.add_argument('--update-schedule', choices=['round-robin', 'aggregate'], default='round-robin',
                   help='round-robin: one owner batch and update per step; aggregate: all owners per update')
    p.add_argument('--debug-progress', action='store_true', help='rank-0 block and Ring3 preparation timing (adds GPU synchronizations)')
    p.add_argument('--check-forward-ranges', action='store_true',
                   help='enable secret runtime checks for forward numerical ranges (extra MPC work)')
    p.add_argument('--ring3-truncation-preprocessing', choices=['distributed', 'trusted-dealer'],
                   default='distributed', help='trusted-dealer is a local performance simulation, not security validated')
    p.add_argument('--heartbeat-seconds', type=float, default=30., help='debug forward heartbeat interval')
    p.add_argument('--min-loss-drop', type=float, default=.001)
    for name in ('rank', 'sigma', 'lr', 'beta1', 'beta2', 'epsilon', 'seed', 'c_bound', 'z_bound', 'sqrt_iters', 'reciprocal_iters'):
        default = getattr(ZOConfig(), name)
        p.add_argument('--'+name.replace('_', '-'), type=type(default), default=default)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    config = ZOConfig(**{k: getattr(args, k) for k in asdict(ZOConfig())}).validate()
    weights(args.weights)
    if (min(args.steps, args.batch_size, args.eval_every) < 1 or not math.isfinite(args.min_loss_drop)
            or args.min_loss_drop < 0 or not math.isfinite(args.heartbeat_seconds) or args.heartbeat_seconds <= 0):
        p.error('positive steps, batch and evaluation interval required')
    if not args.fixture and not all((args.checkpoint, args.fixed_student, args.mpcvit_root, args.manifests, args.dataset_roots)):
        p.error('real model requires checkpoint, fixed-student, mpcvit-root, 3 manifests and 3 dataset-roots')
    if args.device == 'cuda' and torch.cuda.device_count() < (3 if args.backend == '3pc' else 1):
        p.error('insufficient visible CUDA devices')
    if args.backend != '3pc' and args.ring3_truncation_preprocessing != 'distributed':
        p.error('trusted-dealer preprocessing requires the 3pc backend')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    options = vars(args).copy()
    options['output'] = str(root)
    write_json(root/'run.json', {'options': options, 'config': asdict(config)})
    torch.set_num_threads(1)
    if args.backend == 'plaintext':
        run_plain(options, config)
    else:
        ctx = mp.get_context('spawn')
        incoming, outgoing, service = None, None, None
        dealer, dealer_clients, dealer_servers = None, None, None
        if args.mode == 'trusted':
            from .trusted import serve
            incoming, outgoing = ctx.Queue(), [ctx.Queue() for _ in range(3)]
            service_root = root/'trusted-service'
            service_root.mkdir(mode=0o700)
            service = ctx.Process(target=serve, args=(config, args.steps, incoming, outgoing, service_root/'state.pt'))
            service.start()
        try:
            if args.ring3_truncation_preprocessing == 'trusted-dealer':
                from .truncation_dealer import serve as serve_truncations
                pipes = [ctx.Pipe() for _ in range(3)]
                dealer_servers, dealer_clients = zip(*pipes)
                dealer = ctx.Process(target=serve_truncations, args=(dealer_servers,))
                dealer.start()
                for connection in dealer_servers:
                    connection.close()
                dealer_servers = None
            with tempfile.TemporaryDirectory(prefix='rendezvous-', dir=root) as tmp:
                mp.spawn(worker, args=(Path(tmp, 'store').as_uri(), options, config, incoming, outgoing, dealer_clients), nprocs=3, join=True)
            if service:
                service.join(timeout=10)
                if service.exitcode != 0:
                    raise RuntimeError('trusted service did not exit successfully')
        finally:
            if dealer:
                if dealer.is_alive():
                    dealer.terminate()
                dealer.join(timeout=5)
            for connection in (dealer_servers or ()) + (dealer_clients or ()):
                connection.close()
            if service and service.is_alive():
                service.terminate()
                service.join(timeout=5)
    result = json.loads((root/'report.json').read_text())
    print(json.dumps({'event': 'training_complete', 'report': str(root/'report.json'), 'status': result['status']}), flush=True)
    if result['loss_decreased'] is False:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
