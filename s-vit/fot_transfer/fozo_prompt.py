"""Supervised forward-only prompt-token adaptation for the fixed hAttention ViT."""
import torch

from hattention.reference import lsa_attention, mixer_attention, rsa_attention
from fot_image.plaintext_ops import AccuracyPlainOps


class PromptOps(AccuracyPlainOps):
    def __init__(self, params, prompt, allow_gelu_out_of_range=False):
        super().__init__(params, allow_gelu_out_of_range)
        self.prompt = prompt

    def attention(self, q, k, v, block, spec):
        outputs = [None] * len(block['paths'])
        prefix = 1 + self.prompt.shape[0]
        epsilon = round(1e-5 * 2**17) / 2**17
        for path in ('rsa', 'lsa', 'mixer'):
            ids = [i for i, value in enumerate(block['paths']) if value == path]
            if not ids:
                continue
            qg, kg, vg = (x[:, ids] for x in (q, k, v))
            if path == 'rsa':
                group, aux = rsa_attention(qg, kg, vg, epsilon=epsilon, return_aux=True,
                                           score_scale=q.shape[-1]**-.5)
                self.check(aux['scores'], 16 - 2**-17, 'rsa.score')
                if aux['denominator'].min() < .0005 or aux['denominator'].max() > 2048:
                    raise ArithmeticError('numerical range failure: rsa.denominator')
            elif path == 'lsa':
                group = lsa_attention(qg, kg, vg, rho=q.shape[-1]**-.5)
            else:
                patches = mixer_attention(vg[:, :, prefix:], tuple(spec['grid']), False)
                group = torch.cat((vg[:, :, :prefix], patches), dim=2)
            for position, head in enumerate(ids):
                outputs[head] = group[:, position]
        return torch.stack(outputs, dim=1)


def prompt_forward(spec, params, images, prompt, allow_gelu_out_of_range=False):
    """Run the original transformer with tokens inserted after positional encoding."""
    # The shared forward assumes its residual has the same length as its first
    # normalization output, so a separate loop is required for added tokens.
    ops = PromptOps(params, prompt, allow_gelu_out_of_range)
    x = ops.conv(images, spec).flatten(2, 3).transpose(1, 2)
    x = torch.cat((params['cls'].expand(x.shape[0], 1, spec['dim']), x), dim=1)
    if 'pos' in params:
        x = x + params['pos']
    x = torch.cat((x[:, :1], prompt.unsqueeze(0).expand(x.shape[0], -1, -1), x[:, 1:]), dim=1)
    for i, block in enumerate(spec['blocks']):
        key = f'b{i}'
        ops.check(x, 64, key + '.pre_input')
        y = ops.norm(x, key + '.pre', block['pre_eps'])
        heads = len(block['paths'])
        qkv = ops.linear(y, key + '.qkv').reshape(y.shape[0], y.shape[1], 3, heads, spec['dim'] // heads).permute(2, 0, 3, 1, 4)
        for name, value in zip(('q', 'k', 'v'), qkv):
            ops.check(value, 16, key + '.' + name)
        y = ops.attention(*qkv, block, spec).transpose(1, 2).reshape(x.shape)
        x = x + ops.linear(y, key + '.proj')
        ops.check(x, 64, key + '.norm_input')
        x = ops.norm(x, key + '.norm', block['norm_eps'])
        y = ops.linear(x, key + '.fc1')
        ops.check(y, 12, key + '.gelu_input')
        x = x + ops.linear(ops.gelu(y), key + '.fc2')
    ops.check(x, 64, 'final_norm_input')
    logits = ops.linear(ops.norm(x, 'norm', spec['norm_eps'])[:, 0], 'head')
    ops.check(logits, 64, 'logits')
    return logits, ops.range_report()
