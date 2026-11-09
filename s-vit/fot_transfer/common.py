"""Shared, provenance-checked CIFAR-10 transfer model and data helpers."""
import json
from pathlib import Path
import torch
from hattention.student_checkpoint import file_sha256
from training.model import load_backbone
from .data import CIFAR10_CLASSES, CifarData, read_official_archive


def load_data(archive_path, split_path, *, include_test=False):
    archive=read_official_archive(archive_path,include_test=include_test)
    split=json.loads(Path(split_path).read_text())
    if split.get('format')!='cifar10_transfer_splits_v1' or split.get('archive_sha256')!=archive['archive_sha256'] or tuple(split.get('classes',[]))!=CIFAR10_CLASSES:
        raise ValueError('CIFAR archive and split metadata do not match')
    indices=split.get('indices',{})
    names=['public','val','party-0','party-1','party-2']
    if any(k not in indices for k in names+['test']): raise ValueError('missing split')
    all_train=sum((indices[k] for k in names),[])
    if len(all_train)!=len(archive['train'][1]) or len(set(all_train))!=len(all_train) or set(all_train)!=set(range(len(archive['train'][1]))):
        raise ValueError('train splits overlap or omit data')
    if sorted(indices['test'])!=list(range(10000)): raise ValueError('invalid test split')
    data={key:CifarData(archive,'test' if key=='test' else 'train',value)
          for key,value in indices.items() if key!='test' or include_test}
    return archive,split,data


def source_model(a):
    spec,params=load_backbone(a.checkpoint,a.mpcvit_root,a.fixed_student)
    if spec['dim']!=192 or spec['image_shape']!=[3,64,64]: raise ValueError('unsupported ViT shape')
    return spec,params


def install_head(spec, params, head):
    if head['weight'].shape!=(10,spec['dim']) or head['bias'].shape!=(10,): raise ValueError('invalid classifier shape')
    if not torch.isfinite(head['weight']).all() or not torch.isfinite(head['bias']).all(): raise ValueError('nonfinite classifier weights')
    params=dict(params); params['head.weight']=head['weight'].detach().double(); params['head.bias']=head['bias'].detach().double()
    spec=dict(spec); spec['classes']=10
    return spec,params


def load_head(a, split):
    artifact=torch.load(a.head,map_location='cpu',weights_only=True)
    if artifact.get('format')!='cifar10_public_head_v1': raise ValueError('invalid head format')
    if artifact['fixed_student_sha256']!=file_sha256(a.fixed_student) or artifact['source_checkpoint_sha256']!=file_sha256(a.checkpoint): raise ValueError('classifier head was trained for a different ViT')
    if artifact['split_sha256']!=file_sha256(a.splits) or artifact['archive_sha256']!=split['archive_sha256']:
        raise ValueError('classifier head was trained with another data split')
    return artifact
