"""CIFAR-10 archive, stratified disjoint splits and the ViTLite input transform."""
import hashlib
import pickle
import random
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch


# Canonical CIFAR-10 python archive checksum used by the official loader.
OFFICIAL_MD5 = 'c58f30108f718f92721af3b95e74349a'
CIFAR10_CLASSES = ('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def read_official_archive(path, *, include_test=True):
    """Read only the official checksum-verified Python archive. Never extract paths.

    Python pickle is not safe for untrusted files; use the official archive only.
    MD5 here detects accidental corruption, not a cryptographic authenticity proof.
    """
    path=Path(path)
    if digest(path,'md5') != OFFICIAL_MD5:
        raise ValueError('CIFAR-10 archive checksum mismatch; use the official python archive')
    def load(tar, name):
        member=tar.getmember('cifar-10-batches-py/'+name)
        if not member.isfile(): raise ValueError('CIFAR archive member is not a regular file')
        return pickle.load(tar.extractfile(member),encoding='bytes')
    with tarfile.open(path,'r:gz') as tar:
        train=[load(tar,f'data_batch_{i}') for i in range(1,6)]
        test=load(tar,'test_batch') if include_test else None
        meta=load(tar,'batches.meta')
    names=tuple(x.decode('utf-8') for x in meta[b'label_names'])
    if names!=CIFAR10_CLASSES: raise ValueError('unexpected CIFAR-10 class order')
    def tensors(batches):
        images=np.concatenate([b[b'data'] for b in batches]).reshape(-1,3,32,32)
        labels=np.asarray(sum((b[b'labels'] for b in batches),[]),dtype=np.int64)
        if len(images)!=len(labels) or np.any(labels<0) or np.any(labels>=10): raise ValueError('invalid CIFAR examples')
        return images,labels
    training=tensors(train); testing=tensors([test]) if test is not None else None
    if len(training[1])!=50000 or (testing is not None and len(testing[1])!=10000): raise ValueError('unexpected CIFAR-10 dataset size')
    result={'train':training,'archive_sha256':digest(path)}
    if testing is not None: result['test']=testing
    return result


def make_splits(labels, public_per_class, val_per_class, seed):
    """Every train image assigned exactly once; round-robin remainder across owners."""
    labels=np.asarray(labels)
    groups={'public':[],'val':[],'party-0':[],'party-1':[],'party-2':[]}
    for cls in range(10):
        ids=np.flatnonzero(labels==cls).tolist()
        random.Random(f'{seed}:class:{cls}').shuffle(ids)
        if public_per_class<1 or val_per_class<1 or len(ids)<=public_per_class+val_per_class:
            raise ValueError('insufficient examples for stratified split')
        groups['public'].extend(ids[:public_per_class])
        groups['val'].extend(ids[public_per_class:public_per_class+val_per_class])
        for i,index in enumerate(ids[public_per_class+val_per_class:]):
            groups[f'party-{(i+cls)%3}'].append(index)
    for name,values in groups.items(): random.Random(f'{seed}:{name}').shuffle(values)
    flat=sum(groups.values(),[])
    if len(flat)!=len(labels) or len(set(flat))!=len(labels): raise AssertionError('split overlap or omission')
    return groups


def balanced_subset(indices, labels, count):
    """Take deterministic class-interleaved indices without replacement."""
    if count<1 or count>len(indices): raise ValueError('invalid validation sample count')
    groups=[[i for i in indices if int(labels[i])==label] for label in range(10)]
    result=[]
    for offset in range(max(map(len,groups))):
        for group in groups:
            if offset<len(group):
                result.append(group[offset])
                if len(result)==count: return result
    raise ValueError('not enough examples for subset')


class CifarData:
    def __init__(self, archive, split, indices, image_size=64):
        self.images,self.labels=archive[split]
        self.indices=tuple(int(i) for i in indices)
        self.image_size=image_size
        if not self.indices or len(set(self.indices))!=len(self.indices) or min(self.indices)<0 or max(self.indices)>=len(self.labels):
            raise ValueError('invalid or duplicate CIFAR indices')

    def __len__(self): return len(self.indices)

    def sample(self, i):
        index=self.indices[i]
        rgb=Image.fromarray(self.images[index].transpose(1,2,0),mode='RGB')
        image=rgb.resize((self.image_size,self.image_size),Image.Resampling.BICUBIC)
        pixels=torch.from_numpy(np.asarray(image,dtype=np.float32).copy()).permute(2,0,1)/255.
        mean=pixels.new_tensor([.480,.448,.397]).reshape(3,1,1)
        std=pixels.new_tensor([.272,.265,.274]).reshape(3,1,1)
        return ((pixels-mean)/std).double(), int(self.labels[index])

    def batch(self, step, size, fixed=False):
        start=0 if fixed else step*size
        batch=[self.sample((start+i)%len(self)) for i in range(size)]
        return torch.stack([x for x,_ in batch]),torch.tensor([y for _,y in batch],dtype=torch.long)

    def chunks(self,size):
        for start in range(0,len(self),size):
            batch=[self.sample(i) for i in range(start,min(start+size,len(self)))]
            yield torch.stack([x for x,_ in batch]),torch.tensor([y for _,y in batch],dtype=torch.long)
