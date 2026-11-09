import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from fot_explore.conditioning import LocalEncoder
from hattention.student_checkpoint import file_sha256
from training.model import fixture
from .data import CifarData, balanced_subset, make_splits
from . import calibrate, train, select, evaluate


class FakeData:
    def __init__(self, count=8):
        g=torch.Generator().manual_seed(2026)
        self.x=torch.randn(count,3,4,4,generator=g,dtype=torch.float64)*.1
        self.y=torch.arange(count)%3
        self.indices=tuple(range(count))
    def __len__(self): return len(self.indices)
    def batch(self, step, size, fixed=False):
        indices=[(0 if fixed else step*size)+i for i in range(size)]
        ids=[i%len(self) for i in indices]
        return self.x[ids],self.y[ids]
    def chunks(self,size):
        for i in range(0,len(self),size): yield self.x[i:i+size],self.y[i:i+size]


class TransferTests(unittest.TestCase):
    def test_stratified_splits_are_disjoint_and_reproducible(self):
        labels=np.repeat(np.arange(10),20)
        first=make_splits(labels,2,3,17)
        self.assertEqual(first,make_splits(labels,2,3,17))
        self.assertEqual([len(first[k]) for k in ('public','val','party-0','party-1','party-2')],[20,30,50,50,50])
        flat=sum(first.values(),[])
        self.assertEqual(len(set(flat)),200)
        self.assertTrue(all(sum(labels[i]==k for i in first['public'])==2 for k in range(10)))
        subset=balanced_subset(first['val'],labels,20)
        self.assertTrue(all(sum(labels[i]==k for i in subset)==2 for k in range(10)))

    def test_batch_preprocessing_and_last_chunk(self):
        images=np.zeros((5,3,32,32),dtype=np.uint8)
        ds=CifarData({'train':(images,np.arange(5))},'train',list(range(5)))
        x,y=ds.batch(0,4)
        self.assertEqual(tuple(x.shape),(4,3,64,64)); self.assertEqual(y.tolist(),[0,1,2,3])
        self.assertEqual([len(y) for _,y in ds.chunks(3)],[3,2])

    def test_calibration_three_trainers_and_test_pipeline(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('fixed','source','splits'): (root/name).write_text(name)
            mae=root/'mae'; mae.mkdir(); (mae/'model.safetensors').write_bytes(b'fixture weights'); (mae/'config.json').write_text('{}')
            data={name:FakeData(4) for name in ('public','val','party-0','party-1','party-2','test')}
            archive={'archive_sha256':'fixture','train':(None,np.arange(4)%3)}
            split={'archive_sha256':'fixture','classes':[str(i) for i in range(10)]}
            mock_data=lambda *args,**kwargs:(archive,split,data)
            common=['--archive',str(root/'archive'),'--splits',str(root/'splits'),
                    '--checkpoint',str(root/'source'),'--fixed-student',str(root/'fixed'),
                    '--mpcvit-root',str(root),'--device','cpu']
            with contextlib.redirect_stdout(io.StringIO()), patch('fot_transfer.calibrate.load_data',side_effect=mock_data), patch('fot_transfer.calibrate.source_model',side_effect=lambda _:fixture()), patch('sys.argv',['calibrate',*common,'--epochs','2','--feature-batch-size','2','--output',str(root/'head')]):
                calibrate.main()
            head=root/'head/head.pt'
            self.assertTrue(head.exists())
            head_args=['--head',str(head),'--mae-path',str(mae)]
            fake_encoder=lambda *args,**kwargs:LocalEncoder('fixture',32,'cpu',seed=kwargs.get('seed',104))
            for method in ('baseline','image-zo','image-adam'):
                args=['train',*common,*head_args,'--method',method,'--steps','2','--eval-every','1','--batch-size','2','--eval-batch-size','2','--val-samples','4','--output',str(root/method)]
                with contextlib.redirect_stdout(io.StringIO()), patch('fot_transfer.train.load_data',side_effect=mock_data), patch('fot_transfer.train.source_model',side_effect=lambda _:fixture()), patch('fot_transfer.train.LocalEncoder',side_effect=fake_encoder), patch('fot_transfer.train.CifarData',side_effect=lambda *args,**kwargs:data['val']), patch('sys.argv',args):
                    train.main()
                report=json.loads((root/method/'report.json').read_text())
                self.assertEqual(report['status'],'completed_plaintext')
                self.assertEqual(report['steps'][-1]['evaluation']['samples'],4)
            with contextlib.redirect_stdout(io.StringIO()), patch('sys.argv',['select','--baseline',str(root/'baseline'),'--image-zo',str(root/'image-zo'),'--image-adam',str(root/'image-adam'),'--output',str(root/'selection.json')]):
                select.main()
            args=['evaluate',*common,*head_args,'--selection',str(root/'selection.json'),'--batch-size','2','--output',str(root/'test.json')]
            with contextlib.redirect_stdout(io.StringIO()), patch('fot_transfer.evaluate.load_data',side_effect=mock_data), patch('fot_transfer.evaluate.source_model',side_effect=lambda _:fixture()), patch('fot_transfer.evaluate.LocalEncoder',side_effect=fake_encoder), patch('sys.argv',args):
                evaluate.main()
            report=json.loads((root/'test.json').read_text())
            self.assertEqual(report['samples'],4)
            self.assertEqual(set(report['metrics']),{'no-prompt','image-initial','baseline','image-zo','image-adam'})
            self.assertEqual(report['head_sha256'],file_sha256(head))


if __name__=='__main__': unittest.main()
