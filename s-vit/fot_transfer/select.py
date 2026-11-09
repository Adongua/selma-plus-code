"""Select one prompt checkpoint per method on validation, before any test evaluation."""
import argparse
import json
from pathlib import Path
from hattention.student_checkpoint import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',required=True); p.add_argument('--image-zo',required=True); p.add_argument('--image-adam',required=True)
    p.add_argument('--output',required=True); a=p.parse_args()
    out=Path(a.output).resolve()
    if out.exists(): p.error('output exists')
    selected={}; shared=None
    for method,directory in [('baseline',a.baseline),('image-zo',a.image_zo),('image-adam',a.image_adam)]:
        root=Path(directory).resolve(); report=json.loads((root/'report.json').read_text())
        if report.get('status')!='completed_plaintext' or report.get('method')!=method: raise ValueError('missing or wrong training report')
        options=json.loads((root/'options.json').read_text())
        signature=tuple(options[k] for k in ('seed','steps','eval_every','batch_size','val_samples'))
        if shared is not None and signature!=shared: raise ValueError('training runs use different data order, budgets or validation size')
        shared=signature
        candidates=[r for r in report['steps'] if 'evaluation' in r]
        if not candidates: raise ValueError('no validation checkpoints')
        # Prespecified selection: best Top-1, then lower CE, then earlier step.
        chosen=max(candidates,key=lambda r:(r['evaluation']['top1'],-r['evaluation']['loss'],-r['step']))
        ckpt=root/f"step-{chosen['step']:06d}.pt"
        if not ckpt.is_file(): raise FileNotFoundError(ckpt)
        selected[method]={'path':str(ckpt),'sha256':file_sha256(ckpt),'step':chosen['step'],'validation':chosen['evaluation']}
    result={'format':'cifar10_transfer_selection_v1','criterion':'maximum validation top1; ties by lower validation CE then earlier step',
            'selected':selected,'note':'Selection uses validation only; official test is untouched by this command.'}
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()
