"""Train a public CIFAR-10 classifier head on frozen hAttention features."""
import argparse
import json
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from hattention.student_checkpoint import file_sha256
from training.model import forward
from fot_image.plaintext_ops import AccuracyPlainOps
from .common import load_data, source_model


def emit(event,**kwargs): print(json.dumps({'event':event,**kwargs},allow_nan=False),flush=True)


def extract_features(spec,params,data,device,batch_size,allow_gelu,label):
    params=dict(params)
    params['head.weight']=torch.eye(spec['dim'],dtype=torch.float64,device=device)
    params['head.bias']=torch.zeros(spec['dim'],dtype=torch.float64,device=device)
    ops=AccuracyPlainOps(params,allow_gelu)
    xrows=[]; yrows=[]; count=0
    zero=torch.zeros(spec['patches']*spec['dim'],dtype=torch.float64,device=device)
    with torch.no_grad():
        for x,y in data.chunks(batch_size):
            features=forward(spec,ops,x.to(device),zero)
            if not torch.isfinite(features).all(): raise ArithmeticError('nonfinite backbone features')
            xrows.append(features.cpu()); yrows.append(y); count+=len(y)
            if count%max(1,batch_size*10)==0 or count==len(data): emit('feature_progress',split=label,samples=count,total=len(data))
    return torch.cat(xrows),torch.cat(yrows),ops.range_report()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','splits','checkpoint','fixed-student','mpcvit-root','output'): p.add_argument('--'+name,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda'); p.add_argument('--feature-batch-size',type=int,default=8)
    p.add_argument('--epochs',type=int,default=20); p.add_argument('--train-batch-size',type=int,default=128)
    p.add_argument('--lr',type=float,default=.01); p.add_argument('--seed',type=int,default=20261109)
    p.add_argument('--allow-plaintext-gelu-out-of-range',action='store_true')
    a=p.parse_args()
    if min(a.feature_batch_size,a.epochs,a.train_batch_size)<1 or a.lr<=0 or a.seed<0: p.error('invalid arguments')
    if a.device=='cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable')
    root=Path(a.output).resolve(); root.mkdir(parents=True,exist_ok=False,mode=0o700)
    (root/'options.json').write_text(json.dumps(vars(a),indent=2)+'\n')
    torch.set_num_threads(1); device=torch.device(a.device)
    archive,split,data=load_data(a.archive,a.splits)
    spec,params=source_model(a); params={k:v.to(device) for k,v in params.items()}
    emit('feature_extraction_started',public=len(data['public']),validation=len(data['val']))
    train_x,train_y,train_audit=extract_features(spec,params,data['public'],device,a.feature_batch_size,a.allow_plaintext_gelu_out_of_range,'public')
    val_x,val_y,val_audit=extract_features(spec,params,data['val'],device,a.feature_batch_size,a.allow_plaintext_gelu_out_of_range,'validation')
    generator=torch.Generator().manual_seed(a.seed)
    weight=torch.nn.Parameter(torch.randn(10,spec['dim'],generator=generator,dtype=torch.float64)*.01)
    bias=torch.nn.Parameter(torch.zeros(10,dtype=torch.float64))
    optim=torch.optim.Adam([weight,bias],lr=a.lr)
    best=None; rows=[]
    for epoch in range(1,a.epochs+1):
        order=torch.randperm(len(train_y),generator=generator)
        for ids in order.split(a.train_batch_size):
            optim.zero_grad(set_to_none=True)
            loss=F.cross_entropy(F.linear(train_x[ids],weight,bias),train_y[ids])
            loss.backward(); optim.step()
            if not torch.isfinite(weight).all() or not torch.isfinite(bias).all(): raise ArithmeticError('nonfinite classifier parameters')
        with torch.no_grad():
            logits=F.linear(val_x,weight,bias)
            row={'epoch':epoch,'loss':F.cross_entropy(logits,val_y).item(),
                 'top1':(logits.argmax(-1)==val_y).double().mean().item(),'samples':len(val_y)}
        rows.append(row); emit('head_epoch',**row)
        if best is None or row['loss']<best['loss']:
            best={**row,'weight':weight.detach().clone(),'bias':bias.detach().clone()}
    artifact={'format':'cifar10_public_head_v1','weight':best['weight'],'bias':best['bias'],
              'fixed_student_sha256':file_sha256(a.fixed_student),'source_checkpoint_sha256':file_sha256(a.checkpoint),
              'archive_sha256':archive['archive_sha256'],'split_sha256':file_sha256(a.splits),
              'classes':split['classes'],'selected_epoch':best['epoch'],
              'allow_plaintext_gelu_out_of_range':a.allow_plaintext_gelu_out_of_range}
    torch.save(artifact,root/'head.pt')
    report={'status':'completed_plaintext','head':str(root/'head.pt'),'selection':'lowest validation CE',
            'public_samples':len(train_y),'validation_samples':len(val_y),'epochs':rows,
            'range_audit':{'public':train_audit,'validation':val_audit},
            'note':'Frozen backbone; no CIFAR official test data read during selection. GeLU excursions do not validate MPC.'}
    (root/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    emit('complete',head=str(root/'head.pt'),selected_epoch=best['epoch'],validation_top1=best['top1'])


if __name__=='__main__': main()
