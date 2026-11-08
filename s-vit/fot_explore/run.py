"""Independent FoT experiments: shared Adam baseline vs local-MAE conditioned SPSA-GC."""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from training.core import ZOConfig, PlainAdam, basis, direction as base_direction
from training.data import PartyData
from training.model import fixture, load_backbone, PlainOps, forward
from training.run import public_owner_error
from .conditioning import LocalEncoder, conditioned_latent, direction


def emit(event, **values):
    print(json.dumps({'event': event, **values}, allow_nan=False), flush=True)


def load_model(a):
    return fixture() if a.fixture else load_backbone(a.checkpoint, a.mpcvit_root, a.fixed_student)


def make_data(a, spec, party):
    return PartyData(party, spec, None if a.fixture else a.dataset_roots[party],
                     None if a.fixture else a.manifests[party])


def evaluate_plain(a, spec, ops, data, b, z, encoder, theta):
    loss = correct = count = 0
    for ds in data:
        for i in range(len(ds)):
            x, y = ds.batch(i, 1)
            x, y = x.to(a.device), y.to(a.device)
            latent = z if a.method == 'baseline' else conditioned_latent(theta, encoder(x), a.rank, a.feature_dim)
            logits = forward(spec, ops, x, b.matmul(latent))
            loss += F.cross_entropy(logits, y, reduction='sum').item()
            correct += (logits.argmax(-1) == y).sum().item()
            count += y.numel()
    return {'loss': loss/count, 'top1': correct/count, 'samples': count}


@torch.no_grad()
def plain(a):
    spec, values = load_model(a)
    ops = PlainOps({k:v.to(a.device) for k,v in values.items()})
    data = [make_data(a, spec, p) for p in range(3)]
    eval_data = ([PartyData(0, spec, a.eval_root, a.eval_manifest)]
                 if a.eval_manifest else data)
    encoder = LocalEncoder(a.encoder, a.feature_dim, a.device, a.mae_path) if a.method == 'conditioned' else None
    cfg = ZOConfig(rank=a.rank, sigma=a.sigma, lr=a.lr, seed=a.seed)
    adam = PlainAdam(cfg)
    b = basis(spec['patches']*spec['dim'], cfg).to(a.device)
    n = a.rank * (1 + a.feature_dim)
    theta = torch.zeros(n, device=a.device, dtype=torch.float64)
    velocity = theta.clone()
    records = []
    def ev():
        return evaluate_plain(a, spec, ops, eval_data, b, adam.z.to(a.device), encoder, theta)
    initial = ev()
    emit('evaluation', step=0, **initial)
    for step in range(1, a.steps+1):
        start = time.perf_counter()
        owner = (step-1) % 3
        x, y = data[owner].batch((step-1)//3, 1, a.fixed_batch)
        x, y = x.to(a.device), y.to(a.device)
        feature = encoder(x) if a.method == 'conditioned' else None
        if a.method == 'baseline':
            u = base_direction(step, cfg).to(a.device)
            center = adam.z.to(a.device)
        else:
            u = direction(step, n, a.seed).to(a.device)
            center = theta + a.momentum * velocity
        def query(p):
            z = p if feature is None else conditioned_latent(p, feature, a.rank, a.feature_dim)
            return F.cross_entropy(forward(spec, ops, x, b.matmul(z)), y).item()
        diff = query(center+a.sigma*u)-query(center-a.sigma*u)
        if a.method == 'baseline':
            adam.update(diff, u)
        else:
            coefficient = n*diff/(2*a.sigma)
            if not math.isfinite(coefficient) or abs(coefficient) > 8:
                raise ArithmeticError('conditioned SPSA coefficient outside [-8, 8]')
            velocity = a.momentum*velocity - a.lr*coefficient*u
            theta += velocity
            if not torch.isfinite(theta).all() or theta.abs().max() > 8:
                raise ArithmeticError('conditioned parameters outside [-8, 8]')
        row = {'step': step, 'owner': owner, 'update_seconds': time.perf_counter()-start}
        if step % a.eval_every == 0 or step == a.steps:
            row['evaluation'] = ev()
        records.append(row)
        emit('step', **row)
    result = {'backend':'plaintext', 'method':a.method, 'encoder':a.encoder,
              'evaluation_scope':'explicit independent manifest' if a.eval_manifest else 'training manifests',
              'initial':initial, 'steps':records, 'trainable_parameters':a.rank if a.method=='baseline' else n}
    torch.save({'theta':theta.cpu(), 'z':adam.z, 'velocity':velocity.cpu(), 'options':vars(a)}, Path(a.output)/'state.pt')
    return result


def secure_worker(rank, rendezvous, a, connections):
    from training.secure import SecretOps, SecretAdam, precision, open_to, raw_share, check_secret
    from training.step_benchmark import StepBenchmark, aggregate_stages
    from training.truncation_dealer import trusted_truncations
    from hattention.experimental_crypten_truncation import experimental_truncation
    import crypten
    device = torch.device(f'cuda:{rank}' if a.device == 'cuda' else 'cpu')
    if device.type == 'cuda': torch.cuda.set_device(device)
    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE='3', RENDEZVOUS=rendezvous, DISTRIBUTED_BACKEND='gloo')
    crypten.cfg.encoder.precision_bits = 17
    crypten.init(device=device)
    crypten.cfg.communicator.verbose = True
    meter = None
    try:
        loaded = public_owner_error(0, lambda: load_model(a))
        metadata = [None]
        if rank == 0:
            spec, values = loaded
            metadata[0] = (spec, {k:list(v.shape) for k,v in values.items()})
        dist.broadcast_object_list(metadata, src=0)
        spec, shapes = metadata[0]
        params = {k:crypten.cryptensor(values[k].to(device) if rank == 0 else torch.zeros(shape, device=device), src=0, precision=17, device=device) for k,shape in shapes.items()}
        data = None
        for owner in range(3):
            local = public_owner_error(owner, lambda: make_data(a, spec, owner))
            if rank == owner: data = local
        encoder = LocalEncoder(a.encoder, a.feature_dim, device, a.mae_path) if a.method == 'conditioned' else None
        ops = SecretOps(params, device)
        meter = StepBenchmark(crypten, device, ops.base)
        meter.install()
        ops.step_benchmark = meter
        cfg = ZOConfig(rank=a.rank, sigma=a.sigma, lr=a.lr, seed=a.seed)
        b = basis(spec['patches']*spec['dim'], cfg).to(device)
        n = a.rank * (1+a.feature_dim)
        records = []
        with experimental_truncation('carry_save'), trusted_truncations(connections[rank] if connections else None):
            adam = SecretAdam(cfg, device) if a.method == 'baseline' else None
            theta = crypten.cryptensor(torch.zeros(n, device=device), src=0, precision=24) if adam is None else None
            velocity = theta.clone() if theta is not None else None
            for step in range(1, a.steps+1):
                owner = (step-1)%3
                if rank == 0: emit('step_started', step=step, owner=owner)
                before = meter.start()
                item = public_owner_error(owner, lambda: data.batch((step-1)//3, 1, a.fixed_batch))
                x,y = item if rank == owner else (torch.zeros(1,*spec['image_shape'], dtype=torch.float64), None)
                x = x.to(device)
                feature_seconds = 0.
                if rank == owner and encoder:
                    if device.type == 'cuda': torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    feature = encoder(x)
                    if device.type == 'cuda': torch.cuda.synchronize(device)
                    feature_seconds = time.perf_counter() - started
                else:
                    feature = torch.zeros(1,a.feature_dim,device=device)
                duration = [feature_seconds if rank == owner else None]
                dist.broadcast_object_list(duration, src=owner)
                image = crypten.cryptensor(x, src=owner, precision=17, device=device)
                secret_feature = crypten.cryptensor(feature.to(device), src=owner, precision=24, device=device) if encoder else None
                u = (base_direction(step, cfg) if adam else direction(step, n, a.seed)).to(device)
                center = adam.z if adam else theta + a.momentum*velocity
                losses = []
                for sign in (1, -1):
                    candidate = center + sign*a.sigma*u
                    latent = candidate if adam else conditioned_latent(candidate, secret_feature, a.rank, a.feature_dim)
                    prompt = precision(latent.matmul(b.t()), 17)
                    logits = forward(spec, ops, image, prompt)
                    opened = open_to(logits, owner)
                    def loss_at_owner():
                        if not torch.isfinite(opened).all():
                            raise ArithmeticError('nonfinite logits')
                        loss = F.cross_entropy(opened, y.to(device)).item()
                        if not math.isfinite(loss):
                            raise ArithmeticError('nonfinite loss')
                        return loss
                    value = public_owner_error(owner, loss_at_owner)
                    losses.append(value if rank == owner else 0.)
                    if rank == 0: emit('query_complete', step=step, query='plus' if sign == 1 else 'minus')
                delta = crypten.cryptensor(torch.tensor(losses[0]-losses[1], device=device), src=owner, precision=24, device=device)
                fcost = meter.finish(before)
                before = meter.start()
                if adam:
                    adam.update(delta, u)
                else:
                    coefficient = (n/(2*a.sigma))*delta
                    check_secret(coefficient, 8., 'conditioned.spsa_coefficient')
                    velocity = a.momentum*velocity - a.lr*coefficient*u
                    theta = theta + velocity
                    check_secret(theta, 8., 'conditioned.parameters')
                ocost = meter.finish(before)
                stages = aggregate_stages({'forward':fcost, 'optimizer':ocost})
                row = {'step':step, 'owner':owner, 'feature_extraction_seconds_owner':duration[0], 'benchmark':stages}
                records.append(row)
                if rank == 0: emit('step_complete', **row)
            if rank == 0:
                result = {'backend':'3pc','method':a.method,'encoder':a.encoder,'steps':records,
                          'status':'completed_not_security_validated', 'trainable_parameters':a.rank if adam else n,
                          'note':'inline online estimate; feature extraction is included in forward wall; not a security proof'}
                (Path(a.output)/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
            torch.save({'parameter_share':raw_share(adam.z if adam else theta).cpu(),
                        'momentum_share':None if adam else raw_share(velocity).cpu()}, Path(a.output)/f'party-{rank}.pt')
    finally:
        if meter: meter.close()
        crypten.uninit()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['plaintext','3pc'], required=True)
    p.add_argument('--method', choices=['baseline','conditioned'], required=True)
    p.add_argument('--encoder', choices=['fixture','mae'], default='fixture')
    p.add_argument('--mae-path')
    p.add_argument('--fixture', action='store_true')
    p.add_argument('--checkpoint'); p.add_argument('--fixed-student'); p.add_argument('--mpcvit-root')
    p.add_argument('--dataset-roots', nargs=3); p.add_argument('--manifests', nargs=3)
    p.add_argument('--eval-root'); p.add_argument('--eval-manifest')
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--steps', type=int, default=5); p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--rank', type=int, default=4); p.add_argument('--feature-dim', type=int, default=8)
    p.add_argument('--sigma', type=float, default=.1); p.add_argument('--lr', type=float, default=.03)
    p.add_argument('--momentum', type=float, default=.9); p.add_argument('--seed', type=int, default=20261107)
    p.add_argument('--fixed-batch', action='store_true'); p.add_argument('--trusted-dealer', action='store_true')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if min(a.steps,a.eval_every,a.rank,a.feature_dim) < 1 or a.lr <= 0 or a.sigma <= 0 or not 0 <= a.momentum < 1: p.error('invalid hyperparameters')
    if not a.fixture and not all((a.checkpoint,a.fixed_student,a.mpcvit_root,a.dataset_roots,a.manifests)): p.error('real backbone requires checkpoints, roots and manifests')
    if not a.fixture and a.method == 'conditioned' and a.encoder != 'mae': p.error('real conditioned experiment requires a local pretrained MAE')
    if bool(a.eval_root) != bool(a.eval_manifest): p.error('specify eval root and manifest together')
    if a.backend == '3pc' and a.eval_manifest: p.error('independent evaluation is currently plaintext only')
    if a.device == 'cuda' and torch.cuda.device_count() < (3 if a.backend=='3pc' else 1): p.error('insufficient visible GPUs')
    root = Path(a.output).resolve(); root.mkdir(parents=True, exist_ok=False, mode=0o700); a.output = str(root)
    (root/'options.json').write_text(json.dumps(vars(a), indent=2)+'\n')
    torch.set_num_threads(1)
    if a.backend == 'plaintext':
        with torch.no_grad(): result = plain(a)
        (root/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        emit('complete', report=str(root/'report.json'))
        return
    ctx = mp.get_context('spawn'); dealer = None; clients = (); servers = ()
    try:
        if a.trusted_dealer:
            from training.truncation_dealer import serve
            servers, clients = zip(*[ctx.Pipe() for _ in range(3)])
            dealer = ctx.Process(target=serve, args=(servers,)); dealer.start()
            for c in servers: c.close()
            servers = ()
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            mp.spawn(secure_worker, args=(Path(tmp,'store').as_uri(), a, clients), nprocs=3, join=True)
    finally:
        if dealer and dealer.is_alive(): dealer.terminate()
        if dealer: dealer.join(timeout=5)
        for c in clients: c.close()
    emit('complete', report=str(root/'report.json'))


if __name__ == '__main__': main()
