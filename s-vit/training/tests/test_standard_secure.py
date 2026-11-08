import os
from pathlib import Path
import tempfile
import unittest

import torch
import torch.multiprocessing as mp


def worker(rank, rendezvous):
    os.environ.update(RANK=str(rank), WORLD_SIZE='3', RENDEZVOUS=rendezvous, DISTRIBUTED_BACKEND='gloo')
    import crypten
    from hattention.experimental_crypten_truncation import experimental_truncation
    from training.standard_vit import forward
    from training.tests.test_standard_vit import fixture
    from training.secure import open_to
    from training.core import ZOConfig
    from training.first_order_adam import SecretVectorAdam
    crypten.cfg.encoder.precision_bits = 20
    crypten.cfg.functions.sqrt_nr_iters = 7
    crypten.init()
    try:
        torch.manual_seed(543)
        spec, params = fixture()
        spec['epsilon'] = .01
        images = torch.randn(1, 3, 4, 4, dtype=torch.float64) * .1
        plain_params = {name: value.clone().requires_grad_() for name, value in params.items()}
        plain_prompt = torch.zeros(4, 4, dtype=torch.float64, requires_grad=True)
        plain = forward(spec, plain_params, images, plain_prompt, softmax_scale=8)
        torch.nn.functional.cross_entropy(plain, torch.tensor([1])).backward()
        secret_params = {name: crypten.cryptensor(value, src=0, precision=20, requires_grad=True)
                         for name, value in params.items()}
        prompt = crypten.cryptensor(torch.zeros(4, 4), src=0, precision=20, requires_grad=True)
        secret_image = crypten.cryptensor(images, src=1, precision=20)
        states = {}
        with experimental_truncation('carry_save'):
            result = forward(spec, secret_params, secret_image, prompt, softmax_scale=8,
                             trace=lambda name, value: states.update({name: value.requires_grad}))
            if not result.requires_grad:
                raise AssertionError('secure logits detached from graph')
            opened = open_to(result, 0)
            if rank == 0:
                gradient = opened.softmax(dim=1)
                gradient[0, 1] -= 1
            else:
                gradient = torch.zeros(1, 3)
            result.backward(crypten.cryptensor(gradient, src=0, precision=20))
        if prompt.grad is None:
            raise AssertionError('missing prompt gradient; graph states: ' + str(states) + '; parameter gradients present: ' +
                                 str([name for name, value in secret_params.items() if value.grad is not None]))
        prompt_gradient = open_to(prompt.grad, 0)
        classifier_gradient = open_to(secret_params['classifier.weight'].grad, 0)
        gradient_names = ('vit.embeddings.patch_embeddings.projection.weight',
                          'vit.encoder.layer.0.attention.attention.query.weight',
                          'vit.encoder.layer.0.intermediate.dense.weight')
        model_gradients = {name: open_to(secret_params[name].grad, 0) for name in gradient_names}
        initial = torch.tensor([.2, -.3, .4], dtype=torch.float64)
        gradient = torch.tensor([.1, -.2, .3], dtype=torch.float64)
        parameters = crypten.cryptensor(initial, src=0, precision=20)
        shared_gradient = crypten.cryptensor(gradient, src=1, precision=20)
        with experimental_truncation('carry_save'):
            updated = SecretVectorAdam(ZOConfig()).update(parameters, shared_gradient)
        opened_update = open_to(updated, 0)
        if rank == 0:
            torch.testing.assert_close(opened, plain.detach(), atol=.08, rtol=.08)
            torch.testing.assert_close(prompt_gradient, plain_prompt.grad, atol=.005, rtol=.1)
            torch.testing.assert_close(classifier_gradient, plain_params['classifier.weight'].grad, atol=.005, rtol=.1)
            for name in gradient_names:
                torch.testing.assert_close(model_gradients[name], plain_params[name].grad, atol=.005, rtol=.1)
            expected_update = initial - .03 * gradient / (gradient.abs() + .01)
            torch.testing.assert_close(opened_update, expected_update, atol=.003, rtol=.003)
    finally:
        crypten.uninit()


class StandardSecureTests(unittest.TestCase):
    def test_first_order_forward_backward(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(worker, args=(Path(tmp, 'store').as_uri(),), nprocs=3, join=True)
