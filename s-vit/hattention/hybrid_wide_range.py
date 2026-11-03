"""Grouped wide-range hAttention; legacy hybrid.py is left untouched."""
import torch

from .config import normalize_head_types
from .hybrid import HybridMaterial
from .protocol import FixedPointProfile, _validate_share, prepare_lsa, secure_lsa
from .reference import mixer_attention
from .rsa_wide_range import WideRSAProfile, prepare_wide_rsa, wide_rsa


def prepare_wide_hybrid(base, wide, shape, head_types, rho=None):
    shape = tuple(int(x) for x in shape)
    if len(shape) != 4 or min(shape) < 1 or shape[2] > 257 or shape[3] > 16:
        raise ValueError("wide hybrid requires positive B,H,N,D with N<=257 and D<=16")
    if wide.base is not base:
        raise ValueError("wide runtime must share the base runtime")
    paths = normalize_head_types(head_types, shape[1])
    expected_rho = shape[-1] ** -0.5
    if rho is not None and rho != expected_rho:
        raise ValueError("wide RSA requires default attention scaling")
    fixed = FixedPointProfile(input_bound=16.0)
    def size(path):
        return (shape[0], paths.count(path), shape[2], shape[3])
    rsa = prepare_wide_rsa(base, wide, size("rsa"), WideRSAProfile()) if "rsa" in paths else None
    lsa = prepare_lsa(base, size("lsa"), expected_rho, fixed) if "lsa" in paths else None
    return HybridMaterial(base, wide, shape, paths, rsa, lsa)


def secure_wide_hybrid(base, wide, q, k, v, material, *, patch_grid=None, has_cls_token=True):
    if material.used or material.base is not base or material.wide is not wide:
        raise ValueError("material already used or runtime mismatch")
    for name, value in (("q", q), ("k", k), ("v", v)):
        if tuple(value.shape) != material.shape:
            raise ValueError(f"{name} shape mismatch")
        _validate_share(base, value, name)
    material.used = True
    output = torch.empty_like(v)
    for path in ("rsa", "lsa", "mixer"):
        selected = [i for i, value in enumerate(material.paths) if value == path]
        if not selected:
            continue
        index = torch.tensor(selected, dtype=torch.long, device=base.device)
        group_v = v.index_select(1, index)
        if path == "rsa":
            group = wide_rsa(base, wide, q.index_select(1, index), k.index_select(1, index),
                             group_v, material.rsa)
        elif path == "lsa":
            group = secure_lsa(base, q.index_select(1, index), k.index_select(1, index),
                               group_v, material.lsa)
        else:
            group = mixer_attention(group_v, patch_grid, has_cls_token)
        output = output.index_copy(1, index, group)
    return output
