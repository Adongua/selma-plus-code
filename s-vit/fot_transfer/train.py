"""Run one CIFAR-10 plaintext transfer method with a frozen shared classifier head."""
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
from fot_image.decoder import FEATURE_DIM, PlainDecoderOps, image_prompt, initial_parameters, perturb_image, rademacher
from fot_image.plaintext_ops import AccuracyPlainOps
from .common import load_data, source_model, install_head, load_head
from .data import CifarData, balanced_subset


def emit(event,**kwargs): print(json.dumps({'event':event,**kwargs},allow_nan=False),flush=True)


def logits_for(method,spec,ops,dec,images,feature,weights,b,scale):
    zero=torch.zeros(spec['patches']*spec['dim'],dtype=torch.float64,device=images.device)
    if method=='baseline': return forward(spec,ops,images,b.matmul(weights))
    prompt=image_prompt(feature,weights,dec,spec['image_shape'])
    return forward(spec,ops,perturb_image(images,prompt,scale,spec['fixture']),zero)


def evaluate(method,spec,ops,dec,encoder,data,weights,b,scale,device,batch_size):
    loss=correct=count=0
    with torch.no_grad():
        for x,y in data.chunks(batch_size):
            x,y=x.to(device),y.to(device)
            feature=encoder(x) if encoder else None
            logits=logits_for(method,spec,ops,dec,x,feature,weights,b,scale)
            if not torch.isfinite(logits).all(): raise ArithmeticError('nonfinite logits')
            loss+=F.cross_entropy(logits,y,reduction='sum').item(); correct+=(logits.argmax(-1)==y).sum().item(); count+=y.numel()
            if count%max(batch_size*10,1)==0 or count==len(data): emit('evaluation_progress',samples=count,total=len(data))
    return {'loss':loss/count,'top1':correct/count,'samples':count}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','splits','checkpoint','fixed-student','mpcvit-root','head','output'): p.add_argument('--'+name,required=True)
    p.add_argument('--method',choices=['baseline','image-zo','image-adam'],required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda'); p.add_argument('--mae-path')
    p.add_argument('--steps',type=int,default=3000); p.add_argument('--eval-every',type=int,default=200)
    p.add_argument('--batch-size',type=int,default=4); p.add_argument('--eval-batch-size',type=int,default=8)
    p.add_argument('--val-samples',type=int,default=1000,help='0 uses full validation set')
    p.add_argument('--rank',type=int,default=4); p.add_argument('--sigma',type=float,default=.05)
    p.add_argument('--lr',type=float,default=.001); p.add_argument('--epsilon',type=float,default=.01)
    p.add_argument('--momentum',type=float,default=.9); p.add_argument('--prompt-scale',type=float,default=.1)
    p.add_argument('--seed',type=int,default=20261107); p.add_argument('--allow-plaintext-gelu-out-of-range',action='store_true')
    a=p.parse_args()
    if min(a.steps,a.eval_every,a.batch_size,a.eval_batch_size,a.rank)<1 or a.val_samples<0 or any(not math.isfinite(v) or v<=0 for v in (a.lr,a.sigma,a.epsilon,a.prompt_scale)) or not 0<=a.momentum<1: p.error('invalid arguments')
    if a.method!='baseline' and not a.mae_path: p.error('image prompt methods require MAE path')
    if a.device=='cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable')
    root=Path(a.output).resolve(); root.mkdir(parents=True,exist_ok=False,mode=0o700)
    (root/'options.json').write_text(json.dumps(vars(a),indent=2)+'\n')
    torch.set_num_threads(1); device=torch.device(a.device)
    archive,split,data=load_data(a.archive,a.splits)
    head=load_head(a,split)
    if bool(head['allow_plaintext_gelu_out_of_range'])!=a.allow_plaintext_gelu_out_of_range: raise ValueError('head and prompt experiments require same GeLU range policy')
    spec,params=source_model(a); spec,params=install_head(spec,params,head)
    ops=AccuracyPlainOps({k:v.to(device) for k,v in params.items()},a.allow_plaintext_gelu_out_of_range)
    dec=PlainDecoderOps(); encoder=LocalEncoder('mae',FEATURE_DIM,device,a.mae_path,seed=a.seed+44) if a.method!='baseline' else None
    mae_sha256=None; mae_config_sha256=None
    if encoder:
        candidates=[Path(a.mae_path)/'model.safetensors',Path(a.mae_path)/'pytorch_model.bin']
        mae_file=next((path for path in candidates if path.is_file()),None)
        if mae_file is None: raise ValueError('MAE weights not found')
        mae_sha256=file_sha256(mae_file)
        mae_config_sha256=file_sha256(Path(a.mae_path)/'config.json')
    nval=len(data['val']) if a.val_samples==0 else a.val_samples
    if nval>len(data['val']): raise ValueError('val-samples exceeds validation split')
    val_indices=balanced_subset(data['val'].indices,archive['train'][1],nval)
    val=CifarData(archive,'train',val_indices)
    cfg=ZOConfig(rank=a.rank,sigma=a.sigma,lr=a.lr,epsilon=a.epsilon,seed=a.seed).validate()
    b=basis(spec['patches']*spec['dim'],cfg).to(device) if a.method=='baseline' else None
    adam=PlainAdam(cfg) if a.method=='baseline' else None
    theta=initial_parameters(a.seed).to(device) if a.method!='baseline' else None
    velocity=torch.zeros_like(theta) if theta is not None else None
    if a.method=='image-adam': theta=torch.nn.Parameter(theta); optimizer=torch.optim.Adam([theta],lr=a.lr)
    def weights(): return adam.z.to(device) if adam else theta
    initial=evaluate(a.method,spec,ops,dec,encoder,val,weights(),b,a.prompt_scale,device,a.eval_batch_size)
    emit('evaluation',step=0,**initial); rows=[]
    for step in range(1,a.steps+1):
        owner=(step-1)%3; x,y=data[f'party-{owner}'].batch((step-1)//3,a.batch_size)
        x,y=x.to(device),y.to(device); feature=encoder(x) if encoder else None
        if a.method=='baseline':
            u=direction(step,cfg).to(device); z=adam.z.to(device)
            with torch.no_grad():
                plus=F.cross_entropy(logits_for(a.method,spec,ops,dec,x,None,z+a.sigma*u,b,a.prompt_scale),y).item()
                minus=F.cross_entropy(logits_for(a.method,spec,ops,dec,x,None,z-a.sigma*u,b,a.prompt_scale),y).item()
            adam.update(plus-minus,u)
        elif a.method=='image-zo':
            u=rademacher(step,a.seed).to(device); center=theta+a.momentum*velocity
            with torch.no_grad():
                plus=F.cross_entropy(logits_for(a.method,spec,ops,dec,x,feature,center+a.sigma*u,None,a.prompt_scale),y).item()
                minus=F.cross_entropy(logits_for(a.method,spec,ops,dec,x,feature,center-a.sigma*u,None,a.prompt_scale),y).item()
            c=(plus-minus)/(2*a.sigma)
            if not math.isfinite(c) or abs(c)>8: raise ArithmeticError('SPSA coefficient outside [-8,8]')
            velocity=a.momentum*velocity-a.lr*c*u; theta=theta+velocity
            if not torch.isfinite(theta).all() or theta.abs().max()>8: raise ArithmeticError('CNN parameter outside [-8,8]')
        else:
            optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(logits_for(a.method,spec,ops,dec,x,feature,theta,None,a.prompt_scale),y)
            loss.backward()
            if theta.grad is None or not torch.isfinite(theta.grad).all(): raise ArithmeticError('nonfinite CNN gradient')
            optimizer.step()
            if not torch.isfinite(theta).all() or theta.abs().max()>8: raise ArithmeticError('CNN parameter outside [-8,8]')
        row={'step':step,'owner':owner,'batch_size':a.batch_size}
        if step%a.eval_every==0 or step==a.steps:
            row['evaluation']=evaluate(a.method,spec,ops,dec,encoder,val,weights(),b,a.prompt_scale,device,a.eval_batch_size)
            state={'format':'cifar10_transfer_prompt_v1','method':a.method,'step':step,'options':vars(a),
                   'head_sha256':file_sha256(a.head),'fixed_student_sha256':file_sha256(a.fixed_student),
                   'archive_sha256':archive['archive_sha256'],'split_sha256':file_sha256(a.splits),
                   'mae_sha256':mae_sha256,'mae_config_sha256':mae_config_sha256,
                   'weights':weights().detach().cpu()}
            torch.save(state,root/f'step-{step:06d}.pt')
        rows.append(row)
        if step%20==0 or 'evaluation' in row: emit('step_complete',**row)
    report={'status':'completed_plaintext','method':a.method,'initial':initial,'steps':rows,
            'range_audit':ops.range_report(),'validation_samples':nval,
            'note':'Head and ViT frozen; official test not evaluated. Plaintext range excursions do not validate MPC.'}
    (root/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n'); emit('complete',report=str(root/'report.json'))


if __name__=='__main__': main()
