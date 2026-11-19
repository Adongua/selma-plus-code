"""Plaintext, independent prompt capacity and gradient-estimator study."""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from hattention.student_checkpoint import file_sha256
from training.core import ZOConfig, PlainAdam, basis, direction
from training.model import forward
from fot_explore.conditioning import LocalEncoder
from fot_image.decoder import FEATURE_DIM, PARAMS as SMALL_PARAMS, PlainDecoderOps, image_prompt, initial_parameters, perturb_image
from fot_image.plaintext_ops import AccuracyPlainOps
from . import capacity_decoder
from .common import install_head, load_data, load_head, source_model
from .data import CifarData, balanced_subset
from .epoch_compare import epoch_batches
from .train import emit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('archive','splits','checkpoint','fixed-student','mpcvit-root','head','output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--mae-path')
    p.add_argument('--mode', choices=['audit','train'], required=True)
    p.add_argument('--method', choices=['fot','image91','image9007'], required=True)
    p.add_argument('--optimizer', choices=['zo','adam'], default='zo')
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--eval-batch-size', type=int, default=16)
    p.add_argument('--val-samples', type=int, default=1000, help='0 uses all 5000')
    p.add_argument('--max-batches-per-epoch', type=int, default=0, help='0 uses entire train split')
    p.add_argument('--audit-batches', type=int, default=3)
    p.add_argument('--directions', type=int, default=1)
    p.add_argument('--sigma', type=float, default=.01)
    p.add_argument('--lr', type=float, default=.0003)
    p.add_argument('--epsilon', type=float, default=.01)
    p.add_argument('--momentum', type=float, default=.9)
    p.add_argument('--prompt-scale', type=float, default=.1)
    p.add_argument('--seed', type=int, default=20261107)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--allow-plaintext-gelu-out-of-range', action='store_true')
    a = p.parse_args()
    if min(a.epochs,a.batch_size,a.eval_batch_size,a.audit_batches,a.directions,a.log_every)<1 or min(a.val_samples,a.max_batches_per_epoch)<0 or any(not math.isfinite(x) or x<=0 for x in (a.sigma,a.lr,a.epsilon,a.prompt_scale)) or not 0<=a.momentum<1: p.error('invalid arguments')
    if a.method != 'fot' and not a.mae_path: p.error('image methods require --mae-path')
    if a.device == 'cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable')
    if a.mode=='train' and a.method=='fot' and a.optimizer=='zo' and a.directions!=1: p.error('FoT training supports one direction in this study')
    root = Path(a.output).resolve(); root.mkdir(parents=True, exist_ok=False, mode=0o700)
    torch.set_num_threads(1); torch.manual_seed(a.seed); device = torch.device(a.device)
    archive, split, data = load_data(a.archive,a.splits,include_test=False)
    head = load_head(a,split)
    if bool(head['allow_plaintext_gelu_out_of_range']) != a.allow_plaintext_gelu_out_of_range: raise ValueError('GeLU policy mismatch')
    spec, params = install_head(*source_model(a),head)
    params = {k:v.to(device) for k,v in params.items()}
    ops = AccuracyPlainOps(params,a.allow_plaintext_gelu_out_of_range); dec = PlainDecoderOps()
    cfg = ZOConfig(rank=4,sigma=a.sigma,lr=a.lr,epsilon=a.epsilon,seed=a.seed).validate()
    b = basis(spec['patches']*spec['dim'],cfg).to(device) if a.method == 'fot' else None
    encoder = LocalEncoder('mae',FEATURE_DIM,device,a.mae_path,seed=a.seed+44) if a.method != 'fot' else None
    weights = (torch.zeros(4,dtype=torch.float64) if a.method == 'fot' else initial_parameters(a.seed) if a.method == 'image91' else capacity_decoder.initial_parameters(a.seed)).to(device)
    velocity = torch.zeros_like(weights)
    opt = torch.optim.Adam([weights.requires_grad_()],lr=a.lr) if a.optimizer=='adam' and a.mode=='train' else None
    fot_adam = PlainAdam(cfg) if a.method=='fot' and a.optimizer=='zo' and a.directions==1 else None
    hashes = {'archive_sha256':archive['archive_sha256'],'split_sha256':file_sha256(a.splits),'head_sha256':file_sha256(a.head),'fixed_student_sha256':file_sha256(a.fixed_student),'source_sha256':file_sha256(a.checkpoint)}
    if encoder:
        f = next((x for x in (Path(a.mae_path)/'model.safetensors',Path(a.mae_path)/'pytorch_model.bin') if x.is_file()),None)
        if f is None: raise ValueError('MAE weights not found')
        hashes.update(mae_sha256=file_sha256(f),mae_config_sha256=file_sha256(Path(a.mae_path)/'config.json'))
    nval = a.val_samples or len(data['val'])
    val = CifarData(archive,'train',balanced_subset(data['val'].indices,archive['train'][1],nval))

    def logits(x,feature,w):
        if a.method=='fot': return forward(spec,ops,x,b.matmul(w))
        prompt = image_prompt(feature,w,dec,spec['image_shape']) if a.method=='image91' else capacity_decoder.image_prompt(feature,w,spec['image_shape'])
        return forward(spec,ops,perturb_image(x,prompt,a.prompt_scale,spec['fixture']),torch.zeros(spec['patches']*spec['dim'],device=device,dtype=torch.float64))

    def vec(step,index):
        g=torch.Generator().manual_seed(a.seed+1000003*step+7919*index)
        if a.method=='fot':
            u=torch.randn(4,generator=g,dtype=torch.float64)
            return (u/u.norm()).to(device)
        return (2*torch.randint(0,2,(weights.numel(),),generator=g,dtype=torch.int64).double()-1).to(device)

    def evaluate():
        total=correct=count=0
        with torch.no_grad():
            for x,y in val.chunks(a.eval_batch_size):
                x,y=x.to(device),y.to(device); out=logits(x,encoder(x) if encoder else None,weights)
                if not torch.isfinite(out).all(): raise ArithmeticError('nonfinite logits')
                total+=F.cross_entropy(out,y,reduction='sum').item();correct+=(out.argmax(-1)==y).sum().item();count+=y.numel()
        return {'loss':total/count,'top1':correct/count,'samples':count}

    options=vars(a); result={'status':'running','mode':a.mode,'method':a.method,'options':options,'hashes':hashes,'trainable_parameters':weights.numel(),'observations':[], 'note':'Plaintext research only; image9007 is not BlackVIP or an MPC implementation; test split is not loaded.'}
    emit('started',mode=a.mode,method=a.method,trainable_parameters=weights.numel())
    if a.mode=='audit':
        for batch_index,(owner,x,y) in enumerate(epoch_batches(data,1,a.batch_size,a.seed)):
            if batch_index>=a.audit_batches: break
            x,y=x.to(device),y.to(device); feature=encoder(x) if encoder else None
            w=weights.detach().clone().requires_grad_(True); loss=F.cross_entropy(logits(x,feature,w),y)
            grad,=torch.autograd.grad(loss,w); squared_error=ref_sq=0.; estimate=torch.zeros_like(w); details=[]
            for k in range(a.directions):
                u=vec(batch_index+1,k)
                with torch.no_grad():
                    lp=F.cross_entropy(logits(x,feature,w+a.sigma*u),y).item()
                    lm=F.cross_entropy(logits(x,feature,w-a.sigma*u),y).item()
                finite=(lp-lm)/(2*a.sigma); exact=(grad*u).sum().item()
                squared_error+=(finite-exact)**2;ref_sq+=exact**2
                estimate+=finite*u*(4 if a.method=='fot' else 1)
                details.append({'finite_directional_derivative':finite,'autograd_directional_derivative':exact,'plus_loss':lp,'minus_loss':lm})
            estimate/=a.directions
            denom=grad.norm()*estimate.norm()
            cosine=(torch.dot(grad,estimate)/denom).item() if denom.item()>0 else None
            row={'owner':owner,'loss':loss.item(),'true_gradient_norm':grad.norm().item(),'directions':a.directions,'directional_relative_rmse':(squared_error/ref_sq)**.5 if ref_sq>0 else None,'estimate_cosine':cosine,'details':details}
            result['observations'].append(row);emit('audit_complete',**row)
    else:
        initial=evaluate();result['initial_validation']=initial;emit('initial_validation',**initial);step=0
        for epoch in range(1,a.epochs+1):
            n=0; total=0.
            for ix,(owner,x,y) in enumerate(epoch_batches(data,epoch,a.batch_size,a.seed)):
                if a.max_batches_per_epoch and ix>=a.max_batches_per_epoch:break
                step+=1;x,y=x.to(device),y.to(device);feature=encoder(x) if encoder else None
                if opt:
                    opt.zero_grad(set_to_none=True);loss=F.cross_entropy(logits(x,feature,weights),y);loss.backward()
                    if weights.grad is None or not torch.isfinite(weights.grad).all():raise ArithmeticError('invalid gradient')
                    opt.step();value=loss.item()
                else:
                    center=weights if a.method=='fot' else weights+a.momentum*velocity
                    est=torch.zeros_like(weights);value=0.; delta=None
                    with torch.no_grad():
                        for k in range(a.directions):
                            u=vec(step,k);lp=F.cross_entropy(logits(x,feature,center+a.sigma*u),y).item();lm=F.cross_entropy(logits(x,feature,center-a.sigma*u),y).item()
                            c=(lp-lm)/(2*a.sigma)
                            if a.method != 'fot' and (not math.isfinite(c) or abs(c)>8): raise ArithmeticError('SPSA coefficient outside [-8,8]')
                            est+=c*u*(4 if a.method=='fot' else 1);value+=(lp+lm)/2;delta=lp-lm
                        est/=a.directions;value/=a.directions
                        if not torch.isfinite(est).all():raise ArithmeticError('nonfinite ZO gradient')
                        if fot_adam: fot_adam.update(delta,u);weights=fot_adam.z.to(device)
                        else: velocity=a.momentum*velocity-a.lr*est;weights=weights+velocity
                if not torch.isfinite(weights).all():raise ArithmeticError('nonfinite parameters')
                if a.method != 'fot' and weights.abs().max()>8: raise ArithmeticError('prompt parameters outside [-8,8]')
                n+=y.numel();total+=value*y.numel()
                if step%a.log_every==0:emit('training_progress',epoch=epoch,step=step,samples=n,train_loss=total/n)
            validation=evaluate();row={'epoch':epoch,'step':step,'train_samples':n,'train_loss':total/n,'validation':validation,'parameter_norm':weights.norm().item()}
            result['observations'].append(row);torch.save({'format':'capacity_study_v1','options':options,'hashes':hashes,'epoch':epoch,'weights':weights.detach().cpu(),'velocity':velocity.detach().cpu()},root/f'epoch-{epoch:04d}.pt');emit('epoch_complete',**row)
            (root/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    result['status']='completed';result['range_audit']=ops.range_report()
    (root/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');emit('complete',report=str(root/'report.json'))


if __name__=='__main__':main()
