"""Evaluate preselected CIFAR-10 prompts and the shared no-prompt baseline on official test."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from hattention.student_checkpoint import file_sha256
from training.core import ZOConfig, basis
from training.model import forward
from fot_explore.conditioning import LocalEncoder
from fot_image.decoder import FEATURE_DIM, PARAMS, PlainDecoderOps, initial_parameters
from fot_image.plaintext_ops import AccuracyPlainOps
from .common import load_data, source_model, install_head, load_head
from .train import logits_for, emit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','splits','checkpoint','fixed-student','mpcvit-root','head','mae-path','output'): p.add_argument('--'+name,required=True)
    p.add_argument('--selection',required=True,help='validation-only checkpoint selection JSON')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda'); p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--evaluation-split',choices=['validation','test'],default='test')
    p.add_argument('--max-samples',type=int,default=0,help='0 evaluates all; positive prefix is allowed for validation smoke only')
    p.add_argument('--allow-plaintext-gelu-out-of-range',action='store_true')
    a=p.parse_args()
    if a.batch_size<1 or a.max_samples<0: p.error('invalid arguments')
    if a.evaluation_split=='test' and a.max_samples: p.error('partial smoke evaluations must use --evaluation-split validation')
    if a.device=='cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable')
    out=Path(a.output).resolve()
    if out.exists(): p.error('output exists; choose a new path')
    torch.set_num_threads(1); device=torch.device(a.device)
    archive,split,data=load_data(a.archive,a.splits,include_test=a.evaluation_split=='test'); head=load_head(a,split)
    target=data['test' if a.evaluation_split=='test' else 'val']
    if bool(head['allow_plaintext_gelu_out_of_range'])!=a.allow_plaintext_gelu_out_of_range: raise ValueError('GeLU range policy differs from classifier training')
    spec,params=source_model(a); spec,params=install_head(spec,params,head)
    params={k:v.to(device) for k,v in params.items()}
    selection=json.loads(Path(a.selection).read_text())
    if selection.get('format')!='cifar10_transfer_selection_v1': raise ValueError('invalid validation-only selection')
    prompt_paths={method:selection['selected'][method]['path'] for method in ('baseline','image-zo','image-adam')}
    states={}
    head_hash=file_sha256(a.head); fixed_hash=file_sha256(a.fixed_student); split_hash=file_sha256(a.splits)
    for method,path in prompt_paths.items():
        if file_sha256(path)!=selection['selected'][method]['sha256']: raise ValueError('selected checkpoint changed after validation selection')
        state=torch.load(path,map_location='cpu',weights_only=True)
        if state.get('format')!='cifar10_transfer_prompt_v1' or state.get('method')!=method: raise ValueError('invalid prompt or method')
        if state['head_sha256']!=head_hash or state['fixed_student_sha256']!=fixed_hash or state['split_sha256']!=split_hash or state['archive_sha256']!=archive['archive_sha256']: raise ValueError('prompt provenance mismatch')
        if bool(state['options']['allow_plaintext_gelu_out_of_range'])!=a.allow_plaintext_gelu_out_of_range: raise ValueError('prompt GeLU range policy differs')
        value=state['weights'].double().to(device)
        if not torch.isfinite(value).all(): raise ValueError('nonfinite prompt')
        if value.numel()!=(int(state['options']['rank']) if method=='baseline' else PARAMS): raise ValueError('prompt shape mismatch')
        state['weights']=value; state['sha256']=file_sha256(path); states[method]=state
    if states['image-zo']['options']['seed']!=states['image-adam']['options']['seed']: raise ValueError('image prompts use different feature projection seeds')
    if states['image-zo']['options']['prompt_scale']!=states['image-adam']['options']['prompt_scale']: raise ValueError('image prompts use different scales')
    candidates=[Path(a.mae_path)/'model.safetensors',Path(a.mae_path)/'pytorch_model.bin']
    mae_file=next((path for path in candidates if path.is_file()),None)
    if mae_file is None: raise ValueError('MAE weights not found')
    mae_hash=file_sha256(mae_file)
    if any(states[m]['mae_sha256']!=mae_hash for m in ('image-zo','image-adam')): raise ValueError('MAE differs from training weights')
    mae_config_hash=file_sha256(Path(a.mae_path)/'config.json')
    if any(states[m]['mae_config_sha256']!=mae_config_hash for m in ('image-zo','image-adam')): raise ValueError('MAE configuration differs from training')
    encoder=LocalEncoder('mae',FEATURE_DIM,device,a.mae_path,seed=states['image-zo']['options']['seed']+44)
    base_cfg=ZOConfig(rank=states['baseline']['options']['rank'],seed=states['baseline']['options']['seed'])
    b=basis(spec['patches']*spec['dim'],base_cfg).to(device)
    names=['no-prompt','image-initial',*prompt_paths]
    ops={name:AccuracyPlainOps(params,a.allow_plaintext_gelu_out_of_range) for name in names}
    dec=PlainDecoderOps(); rows={name:{'samples':0,'correct_top1':0,'correct_top5':0,'loss_sum':0.,'correct_to_wrong':0,'wrong_to_correct':0} for name in names}
    count=0
    with torch.no_grad():
        for x,y in target.chunks(a.batch_size):
            if a.max_samples:
                remaining=a.max_samples-count
                if remaining<=0: break
                x,y=x[:remaining],y[:remaining]
            x,y=x.to(device),y.to(device); feature=encoder(x)
            zero=torch.zeros(spec['patches']*spec['dim'],dtype=torch.float64,device=device)
            preds={}
            for name in names:
                if name=='no-prompt': logits=forward(spec,ops[name],x,zero)
                elif name=='image-initial':
                    source=states['image-zo']
                    logits=logits_for('image-zo',spec,ops[name],dec,x,feature,
                                      initial_parameters(source['options']['seed']).to(device),None,source['options']['prompt_scale'])
                else:
                    state=states[name]
                    logits=logits_for(name,spec,ops[name],dec,x,feature if name!='baseline' else None,
                                      state['weights'],b if name=='baseline' else None,state['options']['prompt_scale'])
                if not torch.isfinite(logits).all(): raise ArithmeticError('nonfinite test logits')
                pred=logits.argmax(-1); preds[name]=pred
                row=rows[name]; row['samples']+=y.numel(); row['loss_sum']+=F.cross_entropy(logits,y,reduction='sum').item()
                row['correct_top1']+=(pred==y).sum().item(); row['correct_top5']+=(logits.topk(5,dim=-1).indices==y[:,None]).any(-1).sum().item()
            for name in names:
                rows[name]['correct_to_wrong']+=((preds['no-prompt']==y)&(preds[name]!=y)).sum().item()
                rows[name]['wrong_to_correct']+=((preds['no-prompt']!=y)&(preds[name]==y)).sum().item()
            count+=y.numel()
            if count%max(a.batch_size*10,1)==0 or count==len(target): emit('evaluation_progress',samples=count,total=min(a.max_samples or len(target),len(target)))
    for name,row in rows.items():
        row['top1']=row['correct_top1']/count; row['top5']=row['correct_top5']/count; row['loss']=row.pop('loss_sum')/count
        row['range_audit']=ops[name].range_report()
        if name in states: row.update({'step':states[name]['step'],'prompt_sha256':states[name]['sha256']})
        elif name=='image-initial': row.update({'step':0,'seed':states['image-zo']['options']['seed'],
                                                'note':'recreated from the recorded initial seed and current code'})
    report={'status':'evaluated_plaintext','split':'official CIFAR-10 test' if a.evaluation_split=='test' else 'held-out CIFAR-10 train validation',
            'samples':count,'head_sha256':head_hash,'fixed_student_sha256':fixed_hash,'archive_sha256':archive['archive_sha256'],
            'splits_sha256':split_hash,'mae_sha256':mae_hash,'mae_config_sha256':mae_config_hash,'selection_sha256':file_sha256(a.selection),'metrics':rows,
            'note':'Test checkpoint selection must be fixed on validation only; plaintext accuracy does not establish MPC correctness.'}
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    emit('complete',report=str(out),metrics=rows)


if __name__=='__main__': main()
