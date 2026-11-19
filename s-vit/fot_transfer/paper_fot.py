"""Plaintext BlackVIP-style and learned-subspace visual prompt experiments."""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from fot_explore.conditioning import LocalEncoder
from fot_image.plaintext_ops import AccuracyPlainOps
from hattention.student_checkpoint import file_sha256
from training.model import forward
from .common import install_head, load_data, load_head, source_model
from .data import CifarData, balanced_subset
from .epoch_compare import epoch_batches
from .train import emit


class Decoder(nn.Module):
    """BlackVIP's MAE + 800-element trigger decoder; frozen BN statistics."""
    def __init__(self):
        super().__init__()
        self.trigger = nn.Parameter(torch.empty(800).uniform_(0, 0.1))
        layers = []
        for inp, out in ((32,64), (64,32), (32,32), (32,16)):
            intermediate = 64 if inp == 32 and not layers else inp
            layers.extend([nn.ConvTranspose2d(inp, intermediate, 2, 2, groups=inp),
                           nn.ConvTranspose2d(intermediate, out, 1, bias=False),
                           nn.BatchNorm2d(out), nn.GELU()])
        layers.append(nn.ConvTranspose2d(16, 3, 2, 2, bias=False))
        self.body = nn.Sequential(*layers)

    def forward(self, features):
        trigger = self.trigger.expand(features.shape[0], -1)
        return self.body(torch.cat((features, trigger), 1).reshape(-1, 32, 7, 7))


class MaeEncoder(LocalEncoder):
    def __call__(self, images):
        with torch.no_grad():
            mean = images.new_tensor([.480,.448,.397]).view(1,3,1,1)
            std = images.new_tensor([.272,.265,.274]).view(1,3,1,1)
            pixels = (images * std + mean).clamp(0,1).float()
            pixels = F.interpolate(pixels, size=(224,224), mode='bicubic', align_corners=False)
            pixels = (pixels-pixels.new_tensor([.485,.456,.406]).view(1,3,1,1))/pixels.new_tensor([.229,.224,.225]).view(1,3,1,1)
            n = (self.model.config.image_size // self.model.config.patch_size)**2
            noise = torch.arange(n, device=images.device, dtype=pixels.dtype).expand(len(images),-1)
            return self.model(pixel_values=pixels, noise=noise).last_hidden_state[:,0].double()


def pack(module):
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()])


def unpack(module, vector):
    state = {}
    offset = 0
    for name, param in module.named_parameters():
        state[name] = vector[offset:offset+param.numel()].reshape(param.shape)
        offset += param.numel()
    if offset != vector.numel():
        raise ValueError('decoder parameter size mismatch')
    return state


def random_projection(dim, width, existing, generator, device):
    matrix = torch.randn(dim, width, generator=generator, dtype=torch.float64).to(device)
    if existing.shape[1]:
        matrix -= existing @ (existing.T @ matrix)
    q, r = torch.linalg.qr(matrix, mode='reduced')
    if r.diag().abs().min() < 1e-10:
        raise ArithmeticError('subspace projection lost rank')
    return q


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('archive','splits','checkpoint','fixed-student','mpcvit-root','head','mae-path','output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--method', choices=('blackvip','subspace'), required=True)
    p.add_argument('--phase', choices=('train','test'), default='train')
    p.add_argument('--training-split', choices=('party','public'), default='party')
    p.add_argument('--test-checkpoint')
    p.add_argument('--test-max-samples', type=int, default=0, help='smoke only; 0 evaluates full test split')
    p.add_argument('--initial-decoder', help='BlackVIP checkpoint for subspace initialization')
    p.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--eval-batch-size', type=int, default=16)
    p.add_argument('--val-samples', type=int, default=1000)
    p.add_argument('--max-batches', type=int, default=0)
    p.add_argument('--directions', type=int, default=5)
    p.add_argument('--sigma', type=float, default=.01)
    p.add_argument('--lr', type=float, default=.01)
    p.add_argument('--momentum', type=float, default=.9)
    p.add_argument('--alpha', type=float, default=.4)
    p.add_argument('--gamma', type=float, default=.1)
    p.add_argument('--offset', type=float, default=1.)
    p.add_argument('--prompt-scale', type=float, default=.1)
    p.add_argument('--subspace-columns', type=int, default=5)
    p.add_argument('--projection-width', type=int, default=8)
    p.add_argument('--random-columns', type=int, default=2)
    p.add_argument('--subspace-steps', type=int, default=50)
    p.add_argument('--seed', type=int, default=20261107)
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--allow-plaintext-gelu-out-of-range', action='store_true')
    a = p.parse_args()
    if min(a.epochs,a.batch_size,a.eval_batch_size,a.directions,a.subspace_columns,a.projection_width,a.subspace_steps,a.log_every)<1 or min(a.val_samples,a.max_batches,a.random_columns,a.test_max_samples)<0 or min(a.sigma,a.lr,a.prompt_scale)<=0 or not 0<=a.momentum<1:
        p.error('invalid configuration')
    if a.phase == 'test' and not a.test_checkpoint:
        p.error('test phase requires --test-checkpoint')
    if a.phase == 'train' and a.method=='subspace' and not a.initial_decoder:
        p.error('subspace requires --initial-decoder')
    root = Path(a.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    archive, split, data = load_data(a.archive,a.splits,include_test=a.phase == 'test')
    head = load_head(a,split)
    if bool(head['allow_plaintext_gelu_out_of_range']) != a.allow_plaintext_gelu_out_of_range:
        raise ValueError('classifier head GeLU policy mismatch')
    spec, params = install_head(*source_model(a),head)
    params = {k:v.to(device) for k,v in params.items()}
    ops = AccuracyPlainOps(params,a.allow_plaintext_gelu_out_of_range)
    encoder = MaeEncoder('mae',32,device,a.mae_path,seed=a.seed+44)
    decoder = Decoder().to(device=device,dtype=torch.float64).eval()
    names = ['archive','splits','checkpoint','fixed_student','head']
    hashes = {name:file_sha256(getattr(a,name)) for name in names}
    mae = next(x for x in (Path(a.mae_path)/'model.safetensors',Path(a.mae_path)/'pytorch_model.bin') if x.exists())
    hashes['mae'] = file_sha256(mae)
    hashes['mae_config'] = file_sha256(Path(a.mae_path)/'config.json')
    if a.initial_decoder:
        state = torch.load(a.initial_decoder,map_location='cpu',weights_only=True)
        if state['format'] != 'paper_fot_v1' or state['hashes'] != hashes:
            raise ValueError('initial decoder provenance mismatch')
        decoder.load_state_dict(state['decoder'],strict=True)
    initial = pack(decoder)
    trigger_dim = decoder.trigger.numel()
    if a.projection_width + a.subspace_columns + a.random_columns > trigger_dim:
        raise ValueError('requested trigger subspace too large')
    zero = torch.zeros(spec['patches']*spec['dim'],dtype=torch.float64,device=device)
    generator = torch.Generator().manual_seed(a.seed+537)
    calls = {'pretrain':0,'subspace':0,'train':0}
    nval = a.val_samples or len(data['val'])
    val = CifarData(archive,'train',balanced_subset(data['val'].indices,archive['train'][1],nval))

    def predict(x, feature, weights):
        prompt = functional_call(decoder,unpack(decoder,weights),(feature,))
        prompt = F.interpolate(prompt,size=(64,64),mode='bicubic',align_corners=False).double()
        mean=x.new_tensor([.480,.448,.397]).reshape(1,3,1,1)
        std=x.new_tensor([.272,.265,.274]).reshape(1,3,1,1)
        pixels=(x*std+mean+a.prompt_scale*prompt).clamp(0,1)
        return forward(spec,ops,(pixels-mean)/std,zero)

    def evaluate(weights, dataset=val):
        total=correct=count=0
        with torch.no_grad():
            for x,y in dataset.chunks(a.eval_batch_size):
                x,y=x.to(device),y.to(device)
                out=predict(x,encoder(x),weights)
                total+=F.cross_entropy(out,y,reduction='sum').item()
                correct+=(out.argmax(-1)==y).sum().item()
                count+=len(y)
                if count % (a.eval_batch_size*50) == 0 or count == len(dataset):
                    emit('evaluation_progress', samples=count, total=len(dataset))
        return {'loss':total/count,'top1':correct/count,'samples':count}

    def loss(w,x,y,feature,phase):
        calls[phase]+=1
        value=F.cross_entropy(predict(x,feature,w),y).item()
        if not math.isfinite(value):
            raise ArithmeticError('nonfinite loss')
        return value

    def save(path, weights, epoch):
        decoder.load_state_dict({**decoder.state_dict(),**unpack(decoder,weights)},strict=True)
        torch.save({'format':'paper_fot_v1','method':a.method,'epoch':epoch,
                    'decoder':decoder.state_dict(),'hashes':hashes,'options':vars(a),
                    'queries':dict(calls)},path)

    def projection(existing):
        return random_projection(trigger_dim,a.projection_width,existing[0],generator,device), random_projection(initial.numel()-trigger_dim,a.projection_width,existing[1],generator,device)

    def map_latent(z, projections):
        d=projections[0].shape[1]
        return torch.cat((projections[0]@z[:d],projections[1]@z[d:]))

    def gaussian_update(z, projections, x,y,feature,phase):
        estimate=torch.zeros_like(z)
        total=0.0
        for _ in range(a.directions):
            u=torch.randn(z.shape,generator=generator,dtype=z.dtype).to(device)
            delta=map_latent(u,projections)*a.sigma
            lp=loss(initial+map_latent(z,projections)+delta,x,y,feature,phase)
            lm=loss(initial+map_latent(z,projections)-delta,x,y,feature,phase)
            estimate+=(lp-lm)/(2*a.sigma)*u
            total+=(lp+lm)/2
        return z-a.lr*estimate/a.directions, total/a.directions

    if a.phase == 'test':
        selected = torch.load(a.test_checkpoint,map_location='cpu',weights_only=True)
        if selected['format'] != 'paper_fot_v1' or selected['method'] != a.method or selected['hashes'] != hashes:
            raise ValueError('test checkpoint provenance mismatch')
        if selected['options']['prompt_scale'] != a.prompt_scale:
            raise ValueError('test checkpoint prompt scale mismatch')
        decoder.load_state_dict(selected['decoder'],strict=True)
        test_data = data['test']
        if a.test_max_samples:
            test_data = CifarData(archive, 'test', test_data.indices[:a.test_max_samples])
        result={'status':'completed','phase':'test','method':a.method,
                'test':evaluate(pack(decoder), test_data),
                'checkpoint_sha256':file_sha256(a.test_checkpoint),
                'range_audit':ops.range_report()}
        (root/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        emit('complete',**result)
        return

    rows=[]
    emit('started',method=a.method,decoder_parameters=initial.numel())
    initial_val=evaluate(initial)
    emit('validation',epoch=0,**initial_val)
    learned=None
    with torch.no_grad():
        if a.method=='subspace':
            learned=[initial.new_empty((trigger_dim,0)),initial.new_empty((initial.numel()-trigger_dim,0))]
            public_batches=list(data['public'].chunks(a.batch_size))
            emit('subspace_started', columns=a.subspace_columns, steps_per_column=a.subspace_steps)
            for col in range(a.subspace_columns):
                projections=projection(learned)
                z=initial.new_zeros(2*a.projection_width)
                for step in range(a.subspace_steps):
                    x,y=public_batches[(col*a.subspace_steps+step)%len(public_batches)]
                    x,y=x.to(device),y.to(device)
                    z,_=gaussian_update(z,projections,x,y,encoder(x),'subspace')
                for i,part in enumerate((z[:a.projection_width],z[a.projection_width:])):
                    v=projections[i]@part
                    norm=v.norm()
                    if norm < 1e-12:
                        raise ArithmeticError('learned subspace direction has zero norm')
                    learned[i]=torch.cat((learned[i],(v/norm).unsqueeze(1)),dim=1)
                emit('subspace_column_complete',column=col+1,queries=calls['subspace'])
            if a.random_columns:
                for i,dim in enumerate((trigger_dim,initial.numel()-trigger_dim)):
                    extra=random_projection(dim,a.random_columns,learned[i],generator,device)
                    learned[i]=torch.cat((learned[i],extra),dim=1)
            torch.save({'basis':learned,'initial_sha256':file_sha256(a.initial_decoder)},root/'basis.pt')
            z=initial.new_zeros(2*learned[0].shape[1])
        weights=initial.clone()
        momentum=torch.zeros_like(weights)
        step=0
        for epoch in range(1,a.epochs+1):
            total=count=0
            if a.training_split == 'party':
                batches = epoch_batches(data,epoch,a.batch_size,a.seed)
            else:
                batches = ((None,x,y) for x,y in data['public'].chunks(a.batch_size))
            for ix,(owner,x,y) in enumerate(batches):
                if a.max_batches and ix>=a.max_batches:
                    break
                step+=1
                x,y=x.to(device),y.to(device)
                feature=encoder(x)
                if a.method=='blackvip':
                    lr=a.lr/(step+a.offset)**a.alpha
                    sigma=a.sigma/step**a.gamma
                    estimate=torch.zeros_like(weights)
                    value=0.
                    phase = 'pretrain' if a.training_split == 'public' else 'train'
                    for _ in range(a.directions):
                        magnitude=(torch.rand(weights.shape,generator=generator,dtype=weights.dtype)+1)/2
                        sign=2*torch.randint(0,2,weights.shape,generator=generator).double()-1
                        u=(magnitude*sign).to(device)
                        lp=loss(weights+sigma*u,x,y,feature,phase)
                        lm=loss(weights-sigma*u,x,y,feature,phase)
                        estimate+=(lp-lm)/(2*sigma)/u
                        value+=(lp+lm)/2
                    estimate/=a.directions
                    momentum=a.momentum*momentum+estimate
                    weights=weights-lr*(estimate+a.momentum*momentum)
                    value/=a.directions
                else:
                    z,value=gaussian_update(z,learned,x,y,feature,'train')
                    weights=initial+map_latent(z,learned)
                if not torch.isfinite(weights).all():
                    raise ArithmeticError('nonfinite decoder parameters')
                count+=len(y)
                if value is not None:
                    total+=value*len(y)
                if step%a.log_every==0:
                    emit('training_progress',epoch=epoch,step=step,owner=owner,samples=count,queries=dict(calls))
            validation=evaluate(weights)
            row={'epoch':epoch,'step':step,'samples':count,'validation':validation,'queries':dict(calls),
                 'training_perturbed_loss':total/count}
            rows.append(row)
            save(root/f'epoch-{epoch:04d}.pt',weights,epoch)
            report={'status':'completed' if epoch==a.epochs else 'running','method':a.method,
                    'initial_validation':initial_val,'epochs':rows,'options':vars(a),
                    'hashes':hashes,'queries':calls,'range_audit':ops.range_report(),
                    'note':'Plaintext adaptation; no MPC or security validation. Frozen decoder BN statistics. MAE features use deterministic full-token inference. Official test has been previously inspected during development.'}
            (root/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            emit('epoch_complete',**row)
    emit('complete',report=str(root/'report.json'))


if __name__=='__main__':
    main()
