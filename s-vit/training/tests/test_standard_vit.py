import unittest
import torch

from training.standard_vit import forward


def fixture():
    torch.manual_seed(123)
    spec = {'dim': 4, 'heads': 2, 'layers': 1, 'patches': 4, 'epsilon': 1e-4,
            'classes': 3, 'stride': (2, 2), 'padding': (0, 0), 'image_size': 4, 'patch_size': 2}
    p = {}
    def linear(name, out_dim, in_dim):
        p[name+'.weight'] = torch.randn(out_dim, in_dim, dtype=torch.float64) * .1
        p[name+'.bias'] = torch.randn(out_dim, dtype=torch.float64) * .1
    def norm(name):
        p[name+'.weight'] = torch.ones(4, dtype=torch.float64)
        p[name+'.bias'] = torch.zeros(4, dtype=torch.float64)
    p['vit.embeddings.patch_embeddings.projection.weight'] = torch.randn(4, 3, 2, 2, dtype=torch.float64) * .1
    p['vit.embeddings.patch_embeddings.projection.bias'] = torch.zeros(4, dtype=torch.float64)
    p['vit.embeddings.cls_token'] = torch.randn(1, 1, 4, dtype=torch.float64) * .1
    p['vit.embeddings.position_embeddings'] = torch.randn(1, 5, 4, dtype=torch.float64) * .1
    root = 'vit.encoder.layer.0'
    norm(root+'.layernorm_before')
    norm(root+'.layernorm_after')
    norm('vit.layernorm')
    for name in ('query', 'key', 'value'):
        linear(root+'.attention.attention.'+name, 4, 4)
    linear(root+'.attention.output.dense', 4, 4)
    linear(root+'.intermediate.dense', 8, 4)
    linear(root+'.output.dense', 4, 8)
    linear('classifier', 3, 4)
    return spec, p


class StandardViTTests(unittest.TestCase):
    def test_prompt_and_full_model_have_gradients(self):
        spec, params = fixture()
        params = {name: value.requires_grad_() for name, value in params.items()}
        prompt = torch.zeros(spec['patches'], spec['dim'], dtype=torch.float64, requires_grad=True)
        image = torch.randn(1, 3, 4, 4, dtype=torch.float64)
        logits = forward(spec, params, image, prompt, softmax_scale=8)
        torch.nn.functional.cross_entropy(logits, torch.tensor([1])).backward()
        self.assertTrue(torch.isfinite(prompt.grad).all())
        self.assertGreater(prompt.grad.norm().item(), 0)
        self.assertTrue(all(value.grad is not None and torch.isfinite(value.grad).all()
                            for value in params.values()))
