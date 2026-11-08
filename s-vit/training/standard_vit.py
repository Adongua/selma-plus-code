"""Functional standard Hugging Face ViT forward for first-order baselines.

The same functions accept torch tensors and CrypTen tensors. Evaluation mode
is assumed: dropout is disabled. PT uses additive trainable patch embeddings.
"""
import torch
import torch.nn.functional as F

from cryptgpu_gelu.operators import _chebyshev_coefficients
from hattention.benchmark_vit_native_cuda import chebyshev_gelu
from slayernorm.baselines import crypten_native_layernorm


def fixture():
    """Public tiny ViT used only for forward and gradient smoke checks."""
    g = torch.Generator().manual_seed(123)
    spec = {'dim': 4, 'heads': 2, 'layers': 1, 'patches': 4, 'epsilon': .01,
            'classes': 3, 'stride': (2, 2), 'padding': (0, 0), 'image_size': 4,
            'patch_size': 2, 'image_shape': [3, 4, 4], 'fixture': True}
    p = {}
    def rand(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float64) * .1
    def linear(name, output, inputs):
        p[name+'.weight'], p[name+'.bias'] = rand(output, inputs), rand(output)
    def norm(name):
        p[name+'.weight'], p[name+'.bias'] = torch.ones(4, dtype=torch.float64), torch.zeros(4, dtype=torch.float64)
    root = 'vit.encoder.layer.0'
    p['vit.embeddings.patch_embeddings.projection.weight'] = rand(4, 3, 2, 2)
    p['vit.embeddings.patch_embeddings.projection.bias'] = torch.zeros(4, dtype=torch.float64)
    p['vit.embeddings.cls_token'], p['vit.embeddings.position_embeddings'] = rand(1, 1, 4), rand(1, 5, 4)
    for name in (root+'.layernorm_before', root+'.layernorm_after', 'vit.layernorm'):
        norm(name)
    for name in ('query', 'key', 'value'):
        linear(root+'.attention.attention.'+name, 4, 4)
    linear(root+'.attention.output.dense', 4, 4)
    linear(root+'.intermediate.dense', 8, 4)
    linear(root+'.output.dense', 4, 8)
    linear('classifier', 3, 4)
    return spec, p


def extract_teacher(model):
    if not hasattr(model, 'vit') or not hasattr(model, 'classifier'):
        raise ValueError('expected a Hugging Face ViTForImageClassification')
    config = model.config
    if config.hidden_act != 'gelu' or config.hidden_size % config.num_attention_heads:
        raise ValueError('unsupported ViT activation or head dimensions')
    projection = model.vit.embeddings.patch_embeddings.projection
    if projection.groups != 1 or tuple(projection.dilation) != (1, 1):
        raise ValueError('unsupported patch projection')
    params = {name: value.detach().cpu().double().clone() for name, value in model.named_parameters()}
    patches = model.vit.embeddings.position_embeddings.shape[1] - 1
    image_size = (config.image_size, config.image_size) if isinstance(config.image_size, int) else tuple(config.image_size)
    spec = {'dim': config.hidden_size, 'heads': config.num_attention_heads,
            'layers': config.num_hidden_layers, 'patches': patches,
            'epsilon': config.layer_norm_eps, 'classes': config.num_labels,
            'stride': tuple(projection.stride), 'padding': tuple(projection.padding),
            'image_size': config.image_size, 'patch_size': config.patch_size,
            'image_shape': [config.num_channels, *image_size], 'fixture': False}
    return spec, params


def _linear(x, params, name):
    y = x.matmul(params[name + '.weight'].t())
    return y + params[name + '.bias'] if name + '.bias' in params else y


def _norm(x, params, name, epsilon):
    if isinstance(x, torch.Tensor):
        return F.layer_norm(x, (x.shape[-1],), params[name+'.weight'], params[name+'.bias'], epsilon)
    return crypten_native_layernorm(x, params[name+'.weight'], params[name+'.bias'], epsilon)


def forward(spec, params, image, prompt=None, *, softmax_scale=256, gelu_bound=12.0, trace=None):
    stem = 'vit.embeddings.patch_embeddings.projection'
    if isinstance(image, torch.Tensor):
        x = F.conv2d(image, params[stem+'.weight'], params.get(stem+'.bias'),
                     stride=spec['stride'], padding=spec['padding'])
        cat = torch.cat
    else:
        import crypten
        x = image.conv2d(params[stem+'.weight'], stride=spec['stride'], padding=spec['padding'])
        if stem+'.bias' in params:
            x = x + params[stem+'.bias'].reshape(1, -1, 1, 1)
        cat = crypten.cat
    x = x.flatten(2, 3).transpose(1, 2)
    if x.shape[1] != spec['patches']:
        raise ValueError('input patch count does not match teacher positional embeddings')
    if prompt is not None:
        x = x + prompt.reshape(1, spec['patches'], spec['dim'])
    if trace:
        trace('prompt', x)
    cls = params['vit.embeddings.cls_token'] + x.narrow(1, 0, 1) * 0
    x = cat([cls, x], dim=1) + params['vit.embeddings.position_embeddings']
    if trace:
        trace('tokens', x)
    heads, width = spec['heads'], spec['dim'] // spec['heads']
    coefficients = _chebyshev_coefficients(16, gelu_bound)
    for i in range(spec['layers']):
        root = f'vit.encoder.layer.{i}'
        y = _norm(x, params, root+'.layernorm_before', spec['epsilon'])
        q, k, v = [_linear(y, params, root+'.attention.attention.'+name)
                   .reshape(y.shape[0], y.shape[1], heads, width).permute((0, 2, 1, 3))
                   for name in ('query', 'key', 'value')]
        scores = q.matmul(k.transpose(-2, -1)) * (width ** -.5)
        maximum = scores.max(dim=-1, keepdim=True)[0]
        numerator = (scores - maximum).exp()
        denominator = numerator.sum(dim=-1, keepdim=True)
        if isinstance(x, torch.Tensor):
            attention = numerator / denominator
        else:
            import crypten
            with crypten.cfg.temp_override({'functions.reciprocal_all_pos': True}):
                attention = numerator * denominator.div(softmax_scale).reciprocal().div(softmax_scale)
        y = attention.matmul(v).permute((0, 2, 1, 3)).reshape(x.shape)
        x = x + _linear(y, params, root+'.attention.output.dense')
        y = _norm(x, params, root+'.layernorm_after', spec['epsilon'])
        y = _linear(y, params, root+'.intermediate.dense')
        y = chebyshev_gelu(y, gelu_bound, coefficients)
        x = x + _linear(y, params, root+'.output.dense')
        if trace:
            trace('block', x)
    x = _norm(x, params, 'vit.layernorm', spec['epsilon'])
    if trace:
        trace('norm', x)
        trace('cls', x.narrow(1, 0, 1).squeeze(1))
    return _linear(x.narrow(1, 0, 1).squeeze(1), params, 'classifier')
