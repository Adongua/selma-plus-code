"""Independent image-space CNN prompt + owner-local MAE + three-party SPSA-GC."""
import argparse
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

from fot_explore.conditioning import LocalEncoder
from training.data import PartyData
from training.model import fixture, load_backbone, PlainOps, forward
from training.run import public_owner_error
from .decoder import (FEATURE_DIM, PARAMS, PlainDecoderOps, SecretDecoderOps,
                      image_prompt, initial_parameters, perturb_image, rademacher)


def emit(event, **fields):
    print(json.dumps({'event':event, **fields}, allow_nan=False), flush=True)


def model(a):
    return fixture() if a.fixture else load_backbone(a.checkpoint, a.mpcvit_root, a.fixed_student)


def dataset(a, spec, owner):
    return PartyData(owner, spec, None if a.fixture else a.dataset_roots[owner],
                     None if a.fixture else a.manifests[owner])


def exact_loss(spec, backbone, decoder, images, labels, feature, theta, a):
    prompt = image_prompt(feature, theta, decoder, spec['image_shape'])
    perturbed = perturb_image(images, prompt, a.prompt_scale, spec['fixture'])
    zero_patch = torch.zeros(spec['patches']*spec['dim'], device=images.device, dtype=torch.float64)
    logits = forward(spec, backbone, perturbed, zero_patch)
    return F.cross_entropy(logits, labels), logits, prompt


@torch.no_grad()
def plaintext(a):
    device = torch.device(a.device)
    spec, values = model(a)
    backbone = PlainOps({k:v.to(device) for k,v in values.items()})
    decoder = PlainDecoderOps()
    encoder = LocalEncoder(a.encoder, FEATURE_DIM, device, a.mae_path, seed=a.seed+44)
    parties = [dataset(a, spec, i) for i in range(3)]
    validation = ([PartyData(0, spec, a.eval_root, a.eval_manifest)] if a.eval_manifest else parties)
    theta = initial_parameters(a.seed).to(device)
    velocity = torch.zeros_like(theta)

    def evaluate():
        total_loss = total_correct = total = rms = 0
        for ds in validation:
            for i in range(len(ds)):
                x, y = ds.batch(i, 1)
                x, y = x.to(device), y.to(device)
                loss, logits, prompt = exact_loss(spec, backbone, decoder, x, y, encoder(x), theta, a)
                total_loss += loss.item()
                total_correct += (logits.argmax(-1)==y).sum().item()
                rms += (a.prompt_scale * prompt).square().mean().sqrt().item()
                total += y.numel()
        return {'loss':total_loss/total, 'top1':total_correct/total, 'samples':total,
                'pixel_prompt_rms':rms/total}

    initial = evaluate()
    emit('evaluation', step=0, **initial)
    rows = []
    for step in range(1, a.steps+1):
        owner = (step-1)%3
        started = time.perf_counter()
        x,y = parties[owner].batch((step-1)//3, a.batch_size, a.fixed_batch)
        x,y = x.to(device), y.to(device)
        feature = encoder(x)
        u = rademacher(step, a.seed).to(device)
        center = theta + a.momentum*velocity
        plus = exact_loss(spec, backbone, decoder, x,y, feature,center+a.sigma*u,a)[0].item()
        minus = exact_loss(spec, backbone, decoder, x,y, feature,center-a.sigma*u,a)[0].item()
        coefficient = (plus-minus)/(2*a.sigma)
        if not math.isfinite(coefficient) or abs(coefficient)>a.coefficient_bound:
            raise ArithmeticError('SPSA coefficient outside configured contract')
        velocity = a.momentum*velocity-a.lr*coefficient*u
        theta = theta+velocity
        if not torch.isfinite(theta).all() or theta.abs().max()>a.parameter_bound:
            raise ArithmeticError('decoder parameter outside configured contract')
        if device.type=='cuda': torch.cuda.synchronize(device)
        row = {'step':step, 'owner':owner, 'update_seconds':time.perf_counter()-started}
        if a.diagnostic_loss:
            row['perturbation_loss_mean'] = (plus+minus)/2
        if step % a.eval_every == 0 or step == a.steps:
            row['evaluation'] = evaluate()
            torch.save({'theta':theta.cpu(), 'velocity':velocity.cpu(), 'step':step,
                        'options':vars(a)}, Path(a.output)/f'step-{step:06d}.pt')
        rows.append(row)
        emit('step_complete', **row)
    torch.save({'theta':theta.cpu(), 'velocity':velocity.cpu(), 'options':vars(a)}, Path(a.output)/'state.pt')
    return {'backend':'plaintext', 'status':'completed', 'parameters':PARAMS, 'initial':initial,
            'steps':rows, 'prompt_space':'pixel', 'pixel_clipping':False,
            'evaluation_scope':'explicit manifest' if a.eval_manifest else 'training data',
            'encoder':a.encoder, 'model':spec}


def secure(rank, rendezvous, a, connections):
    import crypten
    from training.secure import SecretOps, precision, open_to, raw_share, check_secret
    from training.step_benchmark import StepBenchmark, aggregate_stages
    from training.truncation_dealer import trusted_truncations
    from training.trace import ProgressTrace
    from hattention.experimental_crypten_truncation import experimental_truncation

    device = torch.device(f'cuda:{rank}' if a.device == 'cuda' else 'cpu')
    if device.type == 'cuda': torch.cuda.set_device(device)
    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE='3', RENDEZVOUS=rendezvous, DISTRIBUTED_BACKEND='gloo')
    crypten.cfg.encoder.precision_bits = 17
    crypten.init(device=device)
    crypten.cfg.communicator.verbose = True
    meter = None
    lock = threading.Lock()
    def record(event):
        if rank == 0:
            line = json.dumps(event, allow_nan=False)
            with lock:
                print(line, flush=True)
                with (Path(a.output)/'progress.jsonl').open('a') as handle:
                    handle.write(line+'\n')
    def event(name, **fields):
        record({'event':name, **fields})
    try:
        event('model_loading_started')
        loaded = public_owner_error(0, lambda: model(a))
        metadata = [None]
        if rank == 0:
            spec, values = loaded
            metadata[0] = (spec, {k:list(v.shape) for k,v in values.items()})
        dist.broadcast_object_list(metadata, src=0)
        spec, shapes = metadata[0]
        params = {k:crypten.cryptensor(values[k].to(device) if rank == 0 else torch.zeros(shape, device=device),
                                      src=0, precision=17, device=device) for k,shape in shapes.items()}
        local_data = None
        for owner in range(3):
            item = public_owner_error(owner, lambda: dataset(a, spec, owner))
            if owner == rank: local_data = item
        event('encoder_loading_started')
        encoder = LocalEncoder(a.encoder, FEATURE_DIM, device, a.mae_path, seed=a.seed+44)
        trace = ProgressTrace(a.debug_progress and rank == 0, device, a.heartbeat_seconds, record)
        backbone = SecretOps(params, device, trace, check_forward_ranges=a.check_forward_ranges)
        decoder = SecretDecoderOps(backbone)
        meter = StepBenchmark(crypten, device, backbone.base)
        meter.install()
        backbone.step_benchmark = meter
        theta = crypten.cryptensor(initial_parameters(a.seed).to(device), src=0, precision=24, device=device)
        velocity = theta*0
        rows = []
        with experimental_truncation('carry_save'), trusted_truncations(connections[rank] if connections else None):
            for step in range(1, a.steps+1):
                owner = (step-1)%3
                event('step_started', step=step, owner=owner)
                before = meter.start()
                item = public_owner_error(owner, lambda: local_data.batch((step-1)//3, a.batch_size, a.fixed_batch))
                x,y = item if rank == owner else (torch.zeros(a.batch_size,*spec['image_shape'], dtype=torch.float64), None)
                x = x.to(device)
                feature_seconds = 0.
                if rank == owner:
                    if device.type == 'cuda': torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    feat = encoder(x)
                    if device.type == 'cuda': torch.cuda.synchronize(device)
                    feature_seconds = time.perf_counter()-started
                else:
                    feat = torch.zeros(a.batch_size, FEATURE_DIM, dtype=torch.float64, device=device)
                duration = [feature_seconds if rank == owner else None]
                dist.broadcast_object_list(duration, src=owner)
                shared_x = crypten.cryptensor(x, src=owner, precision=17, device=device)
                shared_f = crypten.cryptensor(feat, src=owner, precision=17, device=device)
                u = rademacher(step, a.seed).to(device)
                center = theta+a.momentum*velocity
                losses = []
                for sign in (1, -1):
                    query = 'plus' if sign == 1 else 'minus'
                    trace.context = {'step':step, 'owner':owner, 'query':query}
                    event('query_started', step=step, owner=owner, query=query)
                    with trace.heartbeat():
                        candidate = precision(center+sign*a.sigma*u, 17)
                        prompt = image_prompt(shared_f, candidate, decoder, spec['image_shape'])
                        inputs = perturb_image(shared_x, prompt, a.prompt_scale, spec['fixture'])
                        zero = torch.zeros(spec['patches']*spec['dim'], dtype=torch.float64, device=device)
                        logits = forward(spec, backbone, inputs, zero)
                    opened = open_to(logits, owner)
                    def local_loss():
                        if not torch.isfinite(opened).all(): raise ArithmeticError('nonfinite logits')
                        result = F.cross_entropy(opened, y.to(device)).item()
                        if not math.isfinite(result): raise ArithmeticError('nonfinite loss')
                        return result
                    value = public_owner_error(owner, local_loss)
                    losses.append(value if rank == owner else 0.)
                    event('query_complete', step=step, query=query)
                delta = crypten.cryptensor(torch.tensor(losses[0]-losses[1], dtype=torch.float64, device=device),
                                          src=owner, precision=24, device=device)
                forward_cost = meter.finish(before)
                before = meter.start()
                coefficient = delta*(1/(2*a.sigma))
                check_secret(coefficient, a.coefficient_bound, 'image_spsa.coefficient')
                velocity = a.momentum*velocity-a.lr*coefficient*u
                theta = theta+velocity
                check_secret(theta, a.parameter_bound, 'image_spsa.theta')
                optimizer_cost = meter.finish(before)
                measured = aggregate_stages({'forward':forward_cost, 'optimizer':optimizer_cost})
                row = {'step':step, 'owner':owner, 'batch_size':a.batch_size, 'feature_extraction_seconds_owner':duration[0],
                       'benchmark':measured}
                if a.diagnostic_loss:
                    local = torch.tensor([(losses[0]+losses[1])/2 if rank==owner else 0.], dtype=torch.float64)
                    dist.broadcast(local, src=owner)
                    row['perturbation_loss_mean'] = local.item()
                rows.append(row)
                event('step_complete', **row)
        torch.save({'theta_share':raw_share(theta).cpu(), 'velocity_share':raw_share(velocity).cpu(),
                    'precision':24, 'rank':rank}, Path(a.output)/f'party-{rank}.pt')
        if rank == 0:
            result = {'backend':'3pc', 'status':'completed_not_security_validated', 'model':spec,
                      'steps':rows, 'parameters':PARAMS, 'prompt_space':'pixel', 'pixel_clipping':False,
                      'note':'Inline online estimate; owner-local encoder included; no security proof.'}
            (Path(a.output)/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    finally:
        if meter: meter.close()
        crypten.uninit()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['plaintext','3pc'], required=True)
    p.add_argument('--encoder', choices=['fixture','mae'], default='fixture'); p.add_argument('--mae-path')
    p.add_argument('--fixture', action='store_true')
    p.add_argument('--checkpoint'); p.add_argument('--fixed-student'); p.add_argument('--mpcvit-root')
    p.add_argument('--dataset-roots', nargs=3); p.add_argument('--manifests', nargs=3)
    p.add_argument('--eval-root'); p.add_argument('--eval-manifest')
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--steps', type=int, default=5); p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--sigma', type=float, default=.05); p.add_argument('--lr', type=float, default=.01)
    p.add_argument('--prompt-scale', type=float, default=.1); p.add_argument('--momentum', type=float, default=.9)
    p.add_argument('--coefficient-bound', type=float, default=8.); p.add_argument('--parameter-bound', type=float, default=8.)
    p.add_argument('--seed', type=int, default=20261107)
    p.add_argument('--fixed-batch', action='store_true'); p.add_argument('--trusted-dealer', action='store_true')
    p.add_argument('--diagnostic-loss', action='store_true', help='explicitly disclose the owner training loss diagnostic')
    p.add_argument('--check-forward-ranges', action='store_true')
    p.add_argument('--debug-progress', action='store_true', help='log per-block and preprocessing progress with extra synchronizations')
    p.add_argument('--heartbeat-seconds', type=float, default=30.)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if min(a.steps,a.eval_every,a.batch_size)<1 or any(not math.isfinite(v) or v<=0 for v in (a.sigma,a.lr,a.prompt_scale,a.coefficient_bound,a.parameter_bound,a.heartbeat_seconds)) or not 0 <= a.momentum < 1: p.error('invalid hyperparameters')
    if not a.fixture and not all((a.checkpoint,a.fixed_student,a.mpcvit_root,a.dataset_roots,a.manifests,a.mae_path)): p.error('real experiment requires model, data, and local MAE')
    if not a.fixture and a.encoder!='mae': p.error('real experiment requires --encoder mae')
    if a.fixture and a.encoder!='fixture': p.error('fixture requires fixture encoder')
    if bool(a.eval_root)!=bool(a.eval_manifest): p.error('specify eval root and manifest together')
    if a.backend=='3pc' and a.eval_manifest: p.error('independent evaluation currently plaintext only')
    if a.device=='cuda' and torch.cuda.device_count() < (3 if a.backend=='3pc' else 1): p.error('insufficient visible GPUs')
    root = Path(a.output).resolve(); root.mkdir(parents=True, exist_ok=False, mode=0o700); a.output = str(root)
    (root/'options.json').write_text(json.dumps(vars(a),indent=2)+'\n')
    torch.set_num_threads(1)
    if a.backend=='plaintext':
        report = plaintext(a)
        (root/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        emit('complete', report=str(root/'report.json'))
        return
    ctx = mp.get_context('spawn'); dealer = None; clients = ()
    try:
        if a.trusted_dealer:
            from training.truncation_dealer import serve
            servers, clients = zip(*[ctx.Pipe() for _ in range(3)])
            dealer = ctx.Process(target=serve, args=(servers,)); dealer.start()
            for c in servers: c.close()
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            mp.spawn(secure, args=(Path(tmp,'store').as_uri(), a, clients), nprocs=3, join=True)
    finally:
        if dealer and dealer.is_alive(): dealer.terminate()
        if dealer: dealer.join(timeout=5)
        for c in clients: c.close()
    emit('complete', report=str(root/'report.json'))


if __name__=='__main__': main()
