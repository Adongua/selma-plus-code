"""Three-party public-fixture tests: selective outputs and secret Adam accuracy."""
import os
from pathlib import Path
import tempfile
import unittest
import torch
import torch.multiprocessing as mp


def worker(rank, rendezvous):
    os.environ.update(RANK=str(rank), WORLD_SIZE='3', RENDEZVOUS=rendezvous, DISTRIBUTED_BACKEND='gloo')
    import crypten
    from training.core import ZOConfig, PlainAdam, direction
    from training.secure import SecretAdam, SecretOps, open_to, precision, negative_bit, check_secret
    from hattention.experimental_crypten_truncation import experimental_truncation
    crypten.init()
    try:
        with experimental_truncation('carry_save'):
            # Q24 comparisons must work even when global precision is Q17.
            crypten.cfg.encoder.precision_bits = 17
            signs = crypten.cryptensor(torch.tensor([-16., -1., 0., 1., 16.]), src=0, precision=24)
            actual_signs = open_to(negative_bit(signs), 0)
            if rank == 0:
                torch.testing.assert_close(actual_signs, torch.tensor([1.,1.,0.,0.,0.], dtype=torch.float64))
            try:
                check_secret(signs, 8., 'expected-test-failure')
            except ArithmeticError:
                pass
            else:
                raise AssertionError('secret range violation was missed')

            from training.model import fixture, forward, PlainOps
            spec, params = fixture()
            secret_params = {k: crypten.cryptensor(v, src=0, precision=17) for k,v in params.items()}
            ops = SecretOps(secret_params, torch.device('cpu'))
            image = torch.full((1,3,4,4), .1, dtype=torch.float64)
            prompt = torch.linspace(-.05,.05,spec['patches']*spec['dim'], dtype=torch.float64)
            expected = forward(spec, PlainOps(params), image, prompt)
            actual = forward(spec, ops, crypten.cryptensor(image,src=1,precision=17),
                             crypten.cryptensor(prompt,src=2,precision=17))
            opened = open_to(actual,1)
            if rank == 1 and (opened-expected).abs().max() > .015:
                raise AssertionError(f'secret forward mismatch: {opened} vs {expected}')
            if rank != 1 and opened is not None:
                raise AssertionError('logits revealed to wrong owner')
            config = ZOConfig()
            secret = SecretAdam(config, torch.device('cpu'))
            plain = PlainAdam(config)
            for step, delta in enumerate([.01, -.005, .002], 1):
                u = direction(step, config)
                shared = crypten.cryptensor(torch.tensor(delta, dtype=torch.float64), src=1, precision=24)
                out = secret.update(shared, u)
                expected = plain.update(delta, u)
                actual = open_to(out, 2)
                if rank == 2:
                    if (actual-expected).abs().max() > .002:
                        raise AssertionError(f'secret Adam differs: {actual} vs {expected}')
                elif actual is not None:
                    raise AssertionError('output disclosed to unauthorized party')
            opened = open_to(precision(secret.z, 17), 0)
            if rank == 0 and (opened-plain.z).abs().max() > .002:
                raise AssertionError('Q24 to Q17 conversion changed the value')
    finally:
        crypten.uninit()


class SecureTests(unittest.TestCase):
    def test_secret_adam(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(worker, args=(Path(tmp, 'store').as_uri(),), nprocs=3, join=True)


if __name__ == '__main__':
    unittest.main()
