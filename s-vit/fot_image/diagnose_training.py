"""Plaintext-only SPSA/autograd diagnostics and matched one-step SGD/Adam reference."""
import argparse
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace
import torch

from fot_explore.conditioning import LocalEncoder
from training.data import PartyData
from training.model import fixture, load_backbone
from .decoder import FEATURE_DIM, PARAMS, PlainDecoderOps, initial_parameters, rademacher
from .plaintext_ops import AccuracyPlainOps
from .run import exact_loss


def emit(event, **fields):
    print(json.dumps({'event':event, **fields},allow_nan=False),flush=True)


def cosine(x,y):
    denom=x.norm()*y.norm()
    return (float(torch.dot(x,y)/denom) if denom.item()>1e-20 else None)


def directional_probe(loss_fn, theta, sigmas, directions, repeats, seed):
    """Same batch and point: reverse-mode gradient vs repeated Rademacher SPSA."""
    point=theta.detach().clone().requires_grad_(True)
    reference=loss_fn(point)
    g,=torch.autograd.grad(reference,point)
    if not torch.isfinite(g).all(): raise ArithmeticError('nonfinite exact gradient')
    g=g.detach(); point=point.detach()
    if g.norm().item()<1e-15: raise ArithmeticError('near-zero exact gradient; cosine undefined')
    sizes=[q for q in (1,4,16,64) if q<=directions]
    if directions not in sizes: sizes.append(directions)
    result=[]
    for sigma in sigmas:
        emit('sigma_started',sigma=sigma,directions_per_repeat=directions,repeats=repeats)
        summaries={q:[] for q in sizes}
        derivative_errors=[]; directional_reference=[]; grad_norms=[]; bound_violations=0; largest=0.
        for rep in range(repeats):
            total=torch.zeros_like(g)
            for idx in range(1,directions+1):
                u=rademacher(rep*directions+idx,seed).to(g.device)
                with torch.no_grad():
                    plus=loss_fn(point+sigma*u).item(); minus=loss_fn(point-sigma*u).item()
                coefficient=(plus-minus)/(2*sigma)
                if not math.isfinite(coefficient): raise ArithmeticError('nonfinite SPSA coefficient')
                largest=max(largest,abs(coefficient))
                bound_violations += abs(coefficient)>8
                reference_directional=torch.dot(g,u).item()
                derivative_errors.append(abs(coefficient-reference_directional))
                directional_reference.append(reference_directional)
                estimate=coefficient*u
                total += estimate
                if idx in summaries:
                    avg=total/idx
                    c=cosine(avg,g)
                    if c is not None: summaries[idx].append(c)
            grad_norms.append((total/directions).norm().item())
            emit('direction_repeat_complete',sigma=sigma,repeat=rep+1,repeats=repeats)
        result.append({'sigma':sigma,'directions_per_repeat':directions,'repeats':repeats,
                       'directional_absolute_error_mean':statistics.mean(derivative_errors),
                       'directional_relative_rmse': math.sqrt(sum(x*x for x in derivative_errors)/max(sum(x*x for x in directional_reference),1e-30)),
                       'coefficient_max_abs':largest,'coefficient_exceeds_8_count':bound_violations,
                       'averaged_gradient_norm_mean':statistics.mean(grad_norms),
                       'mean_cosine_by_directions':{str(k):(statistics.mean(v) if v else None) for k,v in summaries.items()}})
    return {'loss':reference.item(),'true_gradient_norm':g.norm().item(),
            'true_gradient_max_abs':g.abs().max().item(),'measurements':result,
            'note':'Single-direction cosine naturally falls with dimension; compare averaged directions and independent batches. No MPC proof.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',choices=['gradient','first-order'],required=True)
    p.add_argument('--fixture',action='store_true'); p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--encoder',choices=['fixture','mae'],default='fixture'); p.add_argument('--mae-path')
    p.add_argument('--checkpoint'); p.add_argument('--fixed-student'); p.add_argument('--mpcvit-root')
    p.add_argument('--dataset-roots',nargs=3); p.add_argument('--manifests',nargs=3)
    p.add_argument('--eval-root'); p.add_argument('--eval-manifest'); p.add_argument('--state')
    p.add_argument('--batch-size',type=int,default=4); p.add_argument('--batches',type=int,default=3)
    p.add_argument('--directions',type=int,default=16); p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--sigmas',nargs='+',type=float,default=[.01,.05,.1])
    p.add_argument('--steps',type=int,default=300); p.add_argument('--eval-every',type=int,default=50)
    p.add_argument('--lr',type=float,default=.001); p.add_argument('--prompt-scale',type=float,default=.1)
    p.add_argument('--seed',type=int,default=20261107); p.add_argument('--fixed-batch',action='store_true')
    p.add_argument('--allow-plaintext-gelu-out-of-range',action='store_true')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if min(a.batch_size,a.batches,a.directions,a.repeats,a.steps,a.eval_every)<1 or any(not math.isfinite(x) or x<=0 for x in [a.lr,a.prompt_scale,*a.sigmas]): p.error('invalid hyperparameters')
    if a.fixture and a.encoder!='fixture': p.error('fixture requires fixture encoder')
    if not a.fixture and (a.encoder!='mae' or not all((a.mae_path,a.checkpoint,a.fixed_student,a.mpcvit_root,a.dataset_roots,a.manifests))): p.error('real experiment requires MAE, backbone and training manifests')
    if a.experiment=='first-order' and not a.fixture and not all((a.eval_root,a.eval_manifest)): p.error('first-order reference requires evaluation manifest')
    if a.device=='cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable')
    out=Path(a.output).resolve(); out.mkdir(parents=True,exist_ok=False,mode=0o700)
    (out/'options.json').write_text(json.dumps(vars(a),indent=2)+'\n')
    torch.set_num_threads(1); device=torch.device(a.device)
    spec,params=fixture() if a.fixture else load_backbone(a.checkpoint,a.mpcvit_root,a.fixed_student)
    ops=AccuracyPlainOps({k:v.to(device) for k,v in params.items()},a.allow_plaintext_gelu_out_of_range)
    dec=PlainDecoderOps(); encoder=LocalEncoder(a.encoder,FEATURE_DIM,device,a.mae_path,seed=a.seed+44)
    parties=[PartyData(i,spec,None if a.fixture else a.dataset_roots[i],None if a.fixture else a.manifests[i]) for i in range(3)]
    theta=initial_parameters(a.seed).to(device)
    if a.state:
        state=torch.load(a.state,map_location='cpu',weights_only=True)
        config=state.get('options',{})
        if config.get('seed')!=a.seed or float(config.get('prompt_scale',-1))!=a.prompt_scale: raise ValueError('state seed or prompt scale differs')
        if config.get('fixed_student') and Path(config['fixed_student']).resolve()!=Path(a.fixed_student).resolve(): raise ValueError('state backbone differs')
        theta=state['theta'].to(device).double()
        if theta.numel()!=PARAMS or not torch.isfinite(theta).all(): raise ValueError('invalid theta in state')
    fargs=SimpleNamespace(prompt_scale=a.prompt_scale)
    if a.experiment=='gradient':
        results=[]
        for index in range(a.batches):
            owner=index%3
            x,y=parties[owner].batch(index//3,a.batch_size,a.fixed_batch)
            x,y=x.to(device),y.to(device)
            feature=encoder(x).detach()
            def loss_fn(value): return exact_loss(spec,ops,dec,x,y,feature,value,fargs)[0]
            row=directional_probe(loss_fn,theta,a.sigmas,a.directions,a.repeats,a.seed+index*12347)
            row.update({'batch_index':index,'owner':owner,'batch_size':x.shape[0]})
            results.append(row); emit('batch_complete',**row)
        report={'status':'completed','experiment':a.experiment,'results':results,'range_audit':ops.range_report(),
                'scope':'plaintext diagnostic on local training batches; not secure inference'}
    else:
        val=parties if a.fixture else [PartyData(0,spec,a.eval_root,a.eval_manifest)]
        theta=torch.nn.Parameter(theta.clone())
        optim=torch.optim.Adam([theta],lr=a.lr)
        def evaluate():
            loss_sum=correct=count=0
            with torch.no_grad():
                for ds in val:
                    for i in range(len(ds)):
                        x,y=ds.batch(i,1); x,y=x.to(device),y.to(device)
                        loss,logits,_=exact_loss(spec,ops,dec,x,y,encoder(x),theta,fargs)
                        loss_sum+=loss.item(); correct+=(logits.argmax(-1)==y).sum().item(); count+=y.numel()
                        if count%100==0: emit('evaluation_progress',samples=count)
            return {'loss':loss_sum/count,'top1':correct/count,'samples':count}
        initial=evaluate(); emit('evaluation',step=0,**initial); rows=[]
        for step in range(1,a.steps+1):
            owner=(step-1)%3; x,y=parties[owner].batch((step-1)//3,a.batch_size,a.fixed_batch)
            x,y=x.to(device),y.to(device); feature=encoder(x).detach()
            optim.zero_grad(set_to_none=True)
            loss,_,_=exact_loss(spec,ops,dec,x,y,feature,theta,fargs)
            loss.backward()
            if theta.grad is None or not torch.isfinite(theta.grad).all(): raise ArithmeticError('nonfinite first-order gradient')
            optim.step()
            if not torch.isfinite(theta).all() or theta.abs().max()>8: raise ArithmeticError('first-order parameters exceed bound')
            row={'step':step,'owner':owner,'training_loss':loss.item()}
            if step%a.eval_every==0 or step==a.steps:
                row['evaluation']=evaluate()
                torch.save({'theta':theta.detach().cpu(),'step':step,'options':vars(a),'optimizer':optim.state_dict()},out/f'step-{step:06d}.pt')
            rows.append(row); emit('step_complete',**row)
        report={'status':'completed','experiment':a.experiment,'initial':initial,'steps':rows,
                'range_audit':ops.range_report(),'scope':'plaintext first-order oracle only, not a 3PC benchmark'}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    emit('complete',report=str(out/'report.json'))


if __name__=='__main__': main()
