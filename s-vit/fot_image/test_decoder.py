"""Decoder semantics and real-size three-party public-input checks."""
import os
from pathlib import Path
import tempfile
import unittest

import torch
import torch.multiprocessing as mp

from .decoder import (FEATURE_DIM, PARAMS, PlainDecoderOps, image_prompt,
                      initial_parameters, perturb_image, rademacher)


def worker(rank, rendezvous):
    os.environ.update(RANK=str(rank), WORLD_SIZE='3', RENDEZVOUS=rendezvous, DISTRIBUTED_BACKEND='gloo')
    import crypten
    from training.secure import SecretOps, open_to
    from hattention.experimental_crypten_truncation import experimental_truncation
    from .decoder import SecretDecoderOps
    torch.set_num_threads(1)
    crypten.cfg.encoder.precision_bits = 17
    crypten.init()
    try:
        feature = torch.linspace(-.5, .5, FEATURE_DIM, dtype=torch.float64).reshape(1, -1)
        theta = initial_parameters()
        image = torch.linspace(-1., 1., 3*64*64, dtype=torch.float64).reshape(1,3,64,64)
        expected_prompt = image_prompt(feature, theta, PlainDecoderOps(), image.shape[1:])
        expected = perturb_image(image, expected_prompt, .1)
        with experimental_truncation('carry_save'):
            secret_f = crypten.cryptensor(feature, src=1, precision=17)
            secret_theta = crypten.cryptensor(theta, src=0, precision=17)
            secret_image = crypten.cryptensor(image, src=1, precision=17)
            ops = SecretDecoderOps(SecretOps({}, torch.device('cpu')))
            prompt = image_prompt(secret_f, secret_theta, ops, image.shape[1:])
            result = perturb_image(secret_image, prompt, .1)
            observed = open_to(result, 1)
        if rank == 1:
            assert tuple(observed.shape) == (1,3,64,64)
            torch.testing.assert_close(observed, expected, rtol=0, atol=.003)
        elif observed is not None:
            raise AssertionError('image leaked to unauthorized rank')
    finally:
        crypten.uninit()


class DecoderTests(unittest.TestCase):
    def test_shape_direction_and_feature_dependence(self):
        theta = initial_parameters()
        self.assertEqual(theta.numel(), PARAMS)
        d = rademacher(1)
        self.assertTrue(torch.all(d.abs()==1))
        x = torch.zeros(1,FEATURE_DIM, dtype=torch.float64)
        y = torch.ones_like(x)
        a = image_prompt(x, theta, PlainDecoderOps(), (3,64,64))
        b = image_prompt(y, theta, PlainDecoderOps(), (3,64,64))
        self.assertEqual(tuple(a.shape), (1,3,64,64))
        self.assertGreater((a-b).abs().max().item(), 1e-5)

    def test_real_shape_three_party(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(worker, args=(Path(tmp,'store').as_uri(),), nprocs=3, join=True)


if __name__=='__main__': unittest.main()
