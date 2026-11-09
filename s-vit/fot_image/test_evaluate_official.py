import tempfile
import unittest
from pathlib import Path

import torch

from fot_explore.conditioning import LocalEncoder
from training.core import ZOConfig, basis
from training.data import PartyData
from training.model import fixture
from .decoder import FEATURE_DIM, initial_parameters
from .evaluate_official import evaluate_batches, load_prompt


class OfficialEvaluationTests(unittest.TestCase):
    def test_batched_matches_single_and_counts_last_batch(self):
        torch.set_num_threads(1)
        spec, params = fixture()
        ds = PartyData(0,spec)
        images, labels = ds.batch(0,4)
        encoder = LocalEncoder('fixture',FEATURE_DIM,'cpu')
        cfg = ZOConfig(rank=4)
        variants = {'no_prompt':{'kind':'none'},
                    'baseline_step_1':{'kind':'baseline','value':torch.ones(4,dtype=torch.float64)*.01,
                                       'basis':basis(spec['patches']*spec['dim'],cfg)},
                    'image_step_1':{'kind':'image','value':initial_parameters(),'prompt_scale':.1}}
        singles = evaluate_batches(spec,params,[(images[i:i+1],labels[i:i+1]) for i in range(3)],variants,encoder,'cpu',True)
        batches = evaluate_batches(spec,params,[(images[:2],labels[:2]),(images[2:3],labels[2:3])],variants,encoder,'cpu',True)
        for name in variants:
            self.assertEqual(batches[name]['samples'],3)
            self.assertEqual(batches[name]['correct_top1'],singles[name]['correct_top1'])
            self.assertAlmostEqual(batches[name]['loss'],singles[name]['loss'],places=8)
            self.assertEqual(batches[name]['reference_disagreements'],singles[name]['reference_disagreements'])

    def test_checkpoint_rejects_wrong_backbone(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'step.pt'
            torch.save({'z':torch.zeros(4), 'step':200, 'config':{'rank':4,'seed':1,'fixed_student':'/tmp/one.pt'}},path)
            with self.assertRaisesRegex(ValueError,'another fixed-student'):
                load_prompt(path,'baseline','/tmp/other.pt','cpu')


if __name__=='__main__': unittest.main()
