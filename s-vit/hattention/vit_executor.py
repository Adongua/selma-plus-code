"""Forward-only ViTLite execution using interchangeable operator implementations.

The order is taken from MPCViT's TransformerEncoderLayer and classifier. The
operator backend determines whether intermediate tensors are plaintext or shared.
"""

import torch
import torch.nn.functional as F


class PlainOperators:
    def conv(self, x, layer):
        return layer(x)

    def cat_cls(self, x, cls):
        return torch.cat((cls.expand(x.shape[0], -1, -1), x), dim=1)

    def linear(self, x, layer):
        return F.linear(x, layer.weight, layer.bias)

    def norm(self, x, layer):
        return layer(x)

    def gelu(self, x):
        return F.gelu(x)

    def attention(self, q, k, v, module):
        return module.forward_preprojected(q, k, v)


def forward_vit(model, x, ops, on_layer=None, on_stage=None):
    """Run a fixed hAttention ViTLite model and return classification logits."""
    classifier = model.classifier
    if classifier.seq_pool or len(model.tokenizer.conv_layers) != 1:
        raise ValueError("only the single-convolution CLS-token ViTLite is supported")
    tokenizer = model.tokenizer.conv_layers[0]
    if len(tokenizer) != 3 or not all(isinstance(m, torch.nn.Identity) for m in tokenizer[1:]):
        raise ValueError("tokenizer activation and pooling must be identities")
    x = ops.conv(x, tokenizer[0]).flatten(2, 3).transpose(-2, -1)
    x = ops.cat_cls(x, classifier.class_emb)
    if classifier.positional_emb is not None:
        x = x + classifier.positional_emb.detach()
    for index, block in enumerate(classifier.blocks):
        att = block.self_attn
        if on_stage is not None:
            on_stage(index, "block_input", x)
        y = ops.norm(x, block.pre_norm)
        if on_stage is not None:
            on_stage(index, "pre_norm", y)
        b, n, dim = y.shape
        h = att.config.num_heads
        qkv = ops.linear(y, att.qkv).reshape(b, n, 3, h, dim // h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = ops.attention(q, k, v, att).transpose(1, 2).reshape(b, n, dim)
        if on_stage is not None:
            on_stage(index, "attention", y)
        x = x + ops.linear(y, att.proj)
        x = ops.norm(x, block.norm1)
        if on_stage is not None:
            on_stage(index, "norm1", x)
        y = ops.linear(x, block.linear1)
        if on_stage is not None:
            on_stage(index, "gelu_input", y)
        y = ops.gelu(y)
        if on_stage is not None:
            on_stage(index, "gelu", y)
        x = x + ops.linear(y, block.linear2)
        if on_stage is not None:
            on_stage(index, "block_output", x)
        if on_layer is not None:
            on_layer(index, x)
    x = ops.norm(x, classifier.norm)
    return ops.linear(x[:, 0], classifier.fc)
