"""Public fixture check of standard ViTLite secret forward and gradients."""
import os
from pathlib import Path
import tempfile
import unittest

import torch
import torch.multiprocessing as mp


def worker(rank, rendezvous):
    os.environ.update(RANK=str(rank), WORLD_SIZE='3', RENDEZVOUS=rendezvous, DISTRIBUTED_BACKEND='gloo')
    torch.set_num_threads(1)
    import crypten
    from hattention.experimental_crypten_truncation import experimental_truncation
    from training.secure import open_to
    from training.standard_vitlite import fixture, forward
    crypten.cfg.encoder.precision_bits = 20
    crypten.cfg.functions.sqrt_nr_iters = 7
    crypten.init()
    try:
        spec, params = fixture()
        spec['norm_eps'] = .01
        for block in spec['blocks']:
            block['pre_eps'] = block['norm_eps'] = .01
        torch.manual_seed(517)
        images = torch.randn(1, 3, 4, 4, dtype=torch.float64) * .1
        plain_params = {name: value.clone().requires_grad_() for name, value in params.items()}
        plain_prompt = torch.zeros(spec['patches'], spec['dim'], dtype=torch.float64, requires_grad=True)
        plain_logits = forward(spec, plain_params, images, plain_prompt, softmax_scale=8)
        torch.nn.functional.cross_entropy(plain_logits, torch.tensor([1])).backward()
        secret_params = {name: crypten.cryptensor(value, src=0, precision=20, requires_grad=True)
                         for name, value in params.items()}
        prompt = crypten.cryptensor(torch.zeros_like(plain_prompt), src=0, precision=20, requires_grad=True)
        secret_images = crypten.cryptensor(images, src=1, precision=20)
        with experimental_truncation('carry_save'):
            logits = forward(spec, secret_params, secret_images, prompt, softmax_scale=8)
            opened = open_to(logits, 0)
            if rank == 0:
                gradient = opened.softmax(dim=1)
                gradient[0, 1] -= 1
            else:
                gradient = torch.zeros(1, spec['classes'])
            logits.backward(crypten.cryptensor(gradient, src=0, precision=20))
        names = ('conv.weight', 'b0.qkv.weight', 'b0.fc1.weight', 'head.weight')
        if prompt.grad is None or any(secret_params[name].grad is None for name in names):
            raise AssertionError('missing standard ViTLite gradient')
        prompt_grad = open_to(prompt.grad, 0)
        grads = {name: open_to(secret_params[name].grad, 0) for name in names}
        if rank == 0:
            torch.testing.assert_close(opened, plain_logits.detach(), atol=.08, rtol=.08)
            torch.testing.assert_close(prompt_grad, plain_prompt.grad, atol=.005, rtol=.1)
            for name in names:
                torch.testing.assert_close(grads[name], plain_params[name].grad, atol=.005, rtol=.1)
    finally:
        crypten.uninit()


class StandardViTLiteSecureTests(unittest.TestCase):
    def test_secret_forward_and_gradients(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(worker, args=(Path(tmp, 'store').as_uri(),), nprocs=3, join=True)
