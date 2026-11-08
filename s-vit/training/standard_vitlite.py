"""Standard Softmax ViTLite for first-order MPC cost diagnostics."""
import torch
import torch.nn.functional as F

from cryptgpu_gelu.operators import _chebyshev_coefficients
from hattention.benchmark_vit_native_cuda import chebyshev_gelu
from slayernorm.baselines import crypten_native_layernorm


def random_backbone(mpcvit_root, seed):
    from hattention.student_checkpoint import create_random_student
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model, _ = create_random_student(mpcvit_root)
    model.eval()
    classifier = model.classifier
    if classifier.seq_pool or len(model.tokenizer.conv_layers) != 1:
        raise ValueError('expected single-convolution CLS ViTLite')
    conv = model.tokenizer.conv_layers[0]
    if not all(isinstance(module, torch.nn.Identity) for module in conv[1:]):
        raise ValueError('expected identity tokenizer activation and pooling')
    params = {}

    def linear(name, layer):
        params[name+'.weight'] = layer.weight.detach().cpu().double().clone()
        if layer.bias is not None:
            params[name+'.bias'] = layer.bias.detach().cpu().double().clone()

    linear('conv', conv[0])
    params['cls'] = classifier.class_emb.detach().cpu().double().clone()
    if classifier.positional_emb is not None:
        params['pos'] = classifier.positional_emb.detach().cpu().double().clone()
    blocks = []
    for i, block in enumerate(classifier.blocks):
        for key, module in [('pre', block.pre_norm), ('norm', block.norm1),
                            ('qkv', block.self_attn.qkv), ('proj', block.self_attn.proj),
                            ('fc1', block.linear1), ('fc2', block.linear2)]:
            linear(f'b{i}.{key}', module)
        blocks.append({'heads': block.self_attn.num_heads,
                       'pre_eps': block.pre_norm.eps, 'norm_eps': block.norm1.eps})
    linear('norm', classifier.norm)
    linear('head', classifier.fc)
    if len(blocks) != 9 or any(block['heads'] != 12 for block in blocks):
        raise ValueError('expected nine layers with twelve heads')
    dim = params['cls'].shape[-1]
    if dim != 192 or params['pos'].shape[1] != 257:
        raise ValueError('expected 192 dimensions and 257 tokens')
    spec = {'dim': dim, 'heads': 12, 'layers': 9, 'patches': 256,
            'image_shape': [3, 64, 64], 'stride': list(conv[0].stride),
            'padding': list(conv[0].padding), 'blocks': blocks,
            'norm_eps': classifier.norm.eps, 'classes': classifier.fc.out_features,
            'fixture': False}
    return spec, params


def fixture():
    from .model import fixture as base_fixture
    spec, params = base_fixture()
    for block in spec['blocks']:
        block['heads'] = len(block.pop('paths'))
    spec['heads'] = spec['blocks'][0]['heads']
    spec['layers'] = len(spec['blocks'])
    return spec, params


def forward(spec, params, image, prompt=None, *, softmax_scale=256, gelu_bound=12.0,
            gelu_method='erf'):
    if gelu_method not in ('erf', 'chebyshev16'):
        raise ValueError('unknown GeLU method')
    is_plain = isinstance(image, torch.Tensor)
    if is_plain:
        x = F.conv2d(image, params['conv.weight'], params.get('conv.bias'),
                     stride=tuple(spec['stride']), padding=tuple(spec['padding']))
        cat = torch.cat
    else:
        import crypten
        x = image.conv2d(params['conv.weight'], stride=tuple(spec['stride']),
                         padding=tuple(spec['padding']))
        if 'conv.bias' in params:
            x = x + params['conv.bias'].reshape(1, -1, 1, 1)
        cat = crypten.cat

    def linear(value, name):
        out = value.matmul(params[name+'.weight'].t())
        return out + params[name+'.bias'] if name+'.bias' in params else out

    def norm(value, name, epsilon):
        if is_plain:
            return F.layer_norm(value, (spec['dim'],), params[name+'.weight'], params[name+'.bias'], epsilon)
        return crypten_native_layernorm(value, params[name+'.weight'], params[name+'.bias'], epsilon)

    x = x.flatten(2, 3).transpose(1, 2)
    if x.shape[1] != spec['patches']:
        raise ValueError('input patch count mismatch')
    if prompt is not None:
        x = x + prompt.reshape(1, spec['patches'], spec['dim'])
    cls = params['cls'] + x.narrow(1, 0, 1) * 0
    x = cat([cls, x], dim=1)
    if 'pos' in params:
        x = x + params['pos']
    coefficients = _chebyshev_coefficients(16, gelu_bound) if gelu_method == 'chebyshev16' else None
    for i, block in enumerate(spec['blocks']):
        key = f'b{i}'
        y = norm(x, key+'.pre', block['pre_eps'])
        heads, width = block['heads'], spec['dim'] // block['heads']
        qkv = linear(y, key+'.qkv').reshape(y.shape[0], y.shape[1], 3, heads, width).permute((2, 0, 3, 1, 4))
        q, k, v = [qkv.narrow(0, j, 1).squeeze(0) for j in range(3)]
        scores = q.matmul(k.transpose(-2, -1)) * (width ** -.5)
        numerator = (scores - scores.max(dim=-1, keepdim=True)[0]).exp()
        denominator = numerator.sum(dim=-1, keepdim=True)
        if is_plain:
            attention = numerator / denominator
        else:
            import crypten
            with crypten.cfg.temp_override({'functions.reciprocal_all_pos': True}):
                attention = numerator * denominator.div(softmax_scale).reciprocal().div(softmax_scale)
        y = attention.matmul(v).permute((0, 2, 1, 3)).reshape(x.shape)
        x = norm(x + linear(y, key+'.proj'), key+'.norm', block['norm_eps'])
        y = linear(x, key+'.fc1')
        if gelu_method == 'chebyshev16':
            y = chebyshev_gelu(y, gelu_bound, coefficients)
        elif is_plain:
            y = F.gelu(y, approximate='none')
        else:
            from sgelu.baseline import crypten_erf_gelu
            y = crypten_erf_gelu(y)
        x = x + linear(y, key+'.fc2')
    x = norm(x, 'norm', spec['norm_eps'])
    return linear(x.narrow(1, 0, 1).squeeze(1), 'head')
