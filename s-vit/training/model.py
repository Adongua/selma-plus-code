"""One shared forward for plaintext and secret-weight ViTLite. Patch-additive prompts."""
import torch
import torch.nn.functional as F
from hattention.reference import grouped_hattention
from sgelu.sgelu import g2
from slayernorm.wide_range import WideLayerNormProfile


def load_backbone(checkpoint, mpcvit_root, fixed_student):
    from hattention.benchmark_vit_cuda import load_model
    model, meta = load_model(checkpoint, mpcvit_root, fixed_student)
    model.requires_grad_(False).eval()
    c = model.classifier
    if c.seq_pool or len(model.tokenizer.conv_layers) != 1:
        raise ValueError("only single-convolution CLS ViTLite supported")
    conv = model.tokenizer.conv_layers[0]
    if not all(isinstance(x, torch.nn.Identity) for x in conv[1:]):
        raise ValueError("tokenizer must have identity activation and pooling")
    if conv[0].groups != 1 or tuple(conv[0].dilation) != (1, 1):
        raise ValueError("grouped/dilated tokenizer convolution is not supported")
    grid = [(64+2*p-k)//s+1 for p,k,s in zip(conv[0].padding, conv[0].kernel_size, conv[0].stride)]
    if grid != [16, 16]:
        raise ValueError("this initial real-model profile requires 16x16 patches from 64x64 images")
    params = {}
    def linear(name, layer):
        params[name + '.weight'] = layer.weight.detach().cpu().double()
        if layer.bias is not None:
            params[name + '.bias'] = layer.bias.detach().cpu().double()
    linear('conv', conv[0])
    params['cls'] = c.class_emb.detach().cpu().double()
    if c.positional_emb is not None:
        params['pos'] = c.positional_emb.detach().cpu().double()
    blocks = []
    for i, b in enumerate(c.blocks):
        for key, obj in [('pre', b.pre_norm), ('norm', b.norm1), ('qkv', b.self_attn.qkv),
                         ('proj', b.self_attn.proj), ('fc1', b.linear1), ('fc2', b.linear2)]:
            linear(f'b{i}.{key}', obj)
        cfg = b.self_attn.config
        if cfg.srelu_radius != .0055 or cfg.epsilon != 1e-5:
            raise ValueError("unsupported RSA constants")
        dim = params['cls'].shape[-1]
        if dim % cfg.num_heads or cfg.resolved_rho(dim//cfg.num_heads) != (dim//cfg.num_heads)**-.5:
            raise ValueError("unsupported head dimension or attention scaling")
        if not cfg.has_cls_token or cfg.patch_grid not in (None, (16, 16)):
            raise ValueError("attention token layout differs from the prompt profile")
        blocks.append({'paths': list(cfg.head_types), 'pre_eps': b.pre_norm.eps, 'norm_eps': b.norm1.eps})
    linear('norm', c.norm)
    linear('head', c.fc)
    spec = {'dim': params['cls'].shape[-1], 'patches': 256, 'grid': [16, 16],
            'image_shape': [3, 64, 64], 'stride': list(conv[0].stride),
            'padding': list(conv[0].padding), 'blocks': blocks, 'norm_eps': c.norm.eps,
            'classes': c.fc.out_features, 'source': meta, 'fixture': False}
    return spec, params


def fixture():
    """Small PUBLIC frozen transformer, not a substitute for full-checkpoint validation."""
    g = torch.Generator().manual_seed(783)
    d = 6
    def rand(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float64) * .15
    params = {'conv.weight': rand(d, 3, 2, 2), 'conv.bias': rand(d),
              'cls': rand(1, 1, d), 'pos': rand(1, 5, d),
              'b0.qkv.weight': rand(3*d, d), 'b0.qkv.bias': rand(3*d),
              'b0.proj.weight': rand(d, d), 'b0.proj.bias': rand(d),
              'b0.fc1.weight': rand(8, d), 'b0.fc1.bias': rand(8),
              'b0.fc2.weight': rand(d, 8), 'b0.fc2.bias': rand(d),
              'head.weight': rand(3, d) * 3, 'head.bias': torch.tensor([-.3, .2, .1], dtype=torch.float64)}
    for name in ('b0.pre', 'b0.norm', 'norm'):
        params[name+'.weight'] = torch.ones(d, dtype=torch.float64)
        params[name+'.bias'] = torch.zeros(d, dtype=torch.float64)
    return {'dim': d, 'patches': 4, 'grid': [2, 2], 'image_shape': [3, 4, 4],
            'stride': [2, 2], 'padding': [0, 0], 'norm_eps': 1e-5,
            'blocks': [{'paths': ['rsa', 'lsa', 'mixer'], 'pre_eps': 1e-5, 'norm_eps': 1e-5}],
            'classes': 3, 'fixture': True, 'source': 'public-mini-transformer-v1'}, params


class PlainOps:
    def __init__(self, params):
        self.params = params

    def linear(self, x, name):
        return F.linear(x, self.params[name+'.weight'], self.params.get(name+'.bias'))

    def conv(self, x, spec):
        return F.conv2d(x, self.params['conv.weight'], self.params.get('conv.bias'),
                        stride=tuple(spec['stride']), padding=tuple(spec['padding']))

    def cat(self, tensors, dim):
        return torch.cat(tensors, dim=dim)

    def norm(self, x, name, epsilon):
        effective = WideLayerNormProfile(epsilon=epsilon).effective_epsilon
        return F.layer_norm(x, (x.shape[-1],), self.params[name+'.weight'], self.params[name+'.bias'], effective)

    def gelu(self, x):
        return g2(x)

    def attention(self, q, k, v, block, spec):
        from hattention.reference import rsa_attention
        indices = [i for i,p in enumerate(block['paths']) if p == 'rsa']
        if indices:
            _, aux = rsa_attention(q[:,indices], k[:,indices], v[:,indices],
                                   epsilon=round(1e-5*2**17)/2**17, return_aux=True)
            self.check(aux['scores'], 16-2**-17, 'rsa.score')
            if aux['denominator'].min() < .0005 or aux['denominator'].max() > 2048:
                raise ArithmeticError('numerical range failure: rsa.denominator')
        return grouped_hattention(q, k, v, block['paths'], rho=q.shape[-1]**-.5,
                                  epsilon=round(1e-5*2**17)/2**17, patch_grid=tuple(spec['grid']))

    def check(self, x, bound, site):
        if not torch.isfinite(x).all() or x.abs().max() > bound:
            raise ArithmeticError(f"numerical range failure: {site}")


def forward(spec, ops, images, prompt):
    """Add [patches,dim] prompt to patch embeddings; CLS/token count unchanged."""
    p = ops.params
    x = ops.conv(images, spec).flatten(2, 3).transpose(1, 2)
    x = x + prompt.reshape(1, spec['patches'], spec['dim'])
    x = ops.cat([p['cls'].expand(x.shape[0], 1, spec['dim']), x], dim=1)
    if 'pos' in p:
        x = x + p['pos']
    for i, block in enumerate(spec['blocks']):
        key = f'b{i}'
        trace = getattr(ops, 'trace', None)
        if trace:
            trace.block_start(i + 1)
        ops.check(x, 64, key+'.pre_input')
        y = ops.norm(x, key+'.pre', block['pre_eps'])
        h = len(block['paths'])
        qkv = ops.linear(y, key+'.qkv').reshape(y.shape[0], y.shape[1], 3, h, spec['dim']//h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        for name, val in [('q', q), ('k', k), ('v', v)]:
            ops.check(val, 16, key+'.'+name)
        y = ops.attention(q, k, v, block, spec).transpose(1, 2).reshape(x.shape)
        x = x + ops.linear(y, key+'.proj')
        ops.check(x, 64, key+'.norm_input')
        x = ops.norm(x, key+'.norm', block['norm_eps'])
        y = ops.linear(x, key+'.fc1')
        ops.check(y, 12, key+'.gelu_input')
        x = x + ops.linear(ops.gelu(y), key+'.fc2')
        if trace:
            trace.block_end(i + 1)
    ops.check(x, 64, 'final_norm_input')
    out = ops.linear(ops.norm(x, 'norm', spec['norm_eps'])[:, 0], 'head')
    ops.check(out, 64, 'logits')
    return out
