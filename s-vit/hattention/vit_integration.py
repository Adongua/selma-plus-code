"""Replace MPCViT-style attention modules in a plaintext student model.

This module does not import or modify MPCViT.  It operates on an already
constructed model, allowing the caller to load the teacher checkpoint first.
"""

import copy

from torch import nn

from .config import HAttentionConfig
from .module import FixedHAttention


def convert_attention(source, config, *, remove_attention_dropout=False):
    """Copy an MPCViT-style attention's projections into fixed hAttention.

    Attention-weight dropout has no direct efficient equivalent across all
    three paths. Removing nonzero dropout requires an explicit caller choice.
    The returned module is independent of ``source`` and preserves device,
    dtype, gradient flags, output projection dropout, and training mode.
    """
    if not isinstance(config, HAttentionConfig):
        raise TypeError("config must be an HAttentionConfig")
    qkv, proj = getattr(source, "qkv", None), getattr(source, "proj", None)
    if not isinstance(qkv, nn.Linear) or not isinstance(proj, nn.Linear):
        raise TypeError("source attention must contain nn.Linear qkv and proj")
    dim = qkv.in_features
    if (qkv.out_features != 3 * dim or proj.in_features != dim
            or proj.out_features != dim):
        raise ValueError("expected packed 3*dim QKV and square output projection")
    if getattr(source, "num_heads", None) != config.num_heads:
        raise ValueError("source attention head count does not match configuration")
    attn_drop = getattr(source, "attn_drop", None)
    if not isinstance(attn_drop, (nn.Dropout, nn.Identity)):
        raise TypeError("source attn_drop must be nn.Dropout or nn.Identity")
    if isinstance(attn_drop, nn.Dropout) and attn_drop.p > 0 and not remove_attention_dropout:
        raise ValueError("source has nonzero attention-weight dropout; pass "
                         "remove_attention_dropout=True to explicitly remove it")
    proj_drop = getattr(source, "proj_drop", None)
    if not isinstance(proj_drop, (nn.Dropout, nn.Identity)):
        raise TypeError("source proj_drop must be nn.Dropout or nn.Identity")
    result = FixedHAttention(dim, config, qkv_bias=qkv.bias is not None,
                             projection_bias=proj.bias is not None)
    result.qkv = copy.deepcopy(qkv)
    result.proj = copy.deepcopy(proj)
    result.proj_drop = copy.deepcopy(proj_drop)
    result.train(source.training)
    return result


def make_fixed_student(model, configs, *, layer_indices=None,
                       remove_attention_dropout=False):
    """Deep-copy a ViTLite-style teacher and replace selected attention layers.

    ``configs`` has one HAttentionConfig per selected layer. By default every
    classifier block is replaced; pass layer_indices to select a subset.
    All replacements are validated before modifying the copied model.
    """
    classifier = getattr(model, "classifier", None)
    blocks = getattr(classifier, "blocks", None)
    if not isinstance(blocks, (nn.ModuleList, nn.Sequential)):
        raise TypeError("model.classifier.blocks must be a ModuleList or Sequential")
    indices = tuple(range(len(blocks))) if layer_indices is None else tuple(layer_indices)
    configs = tuple(configs)
    if not indices or len(indices) != len(configs):
        raise ValueError("provide exactly one config per selected layer")
    if (any(not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= len(blocks)
            for i in indices) or len(set(indices)) != len(indices)):
        raise ValueError("layer indices must be unique valid nonnegative integers")
    seq_pool = getattr(classifier, "seq_pool", None)
    if not isinstance(seq_pool, bool):
        raise TypeError("model.classifier.seq_pool must identify CLS-token usage")
    for config in configs:
        if not isinstance(config, HAttentionConfig):
            raise TypeError("all configs must be HAttentionConfig instances")
        if config.has_cls_token == seq_pool:
            raise ValueError("config.has_cls_token conflicts with classifier.seq_pool")
    replacements = [convert_attention(blocks[index].self_attn, config,
                     remove_attention_dropout=remove_attention_dropout)
                    for index, config in zip(indices, configs)]
    student = copy.deepcopy(model)
    for index, replacement in zip(indices, replacements):
        student.classifier.blocks[index].self_attn = replacement
    return student
