"""Experimental grouped Ring3 RSA, LSA and Mixer attention core."""

from dataclasses import dataclass
from contextlib import nullcontext

import torch

from .config import normalize_head_types
from .protocol import FixedPointProfile, _validate_share, prepare_lsa, secure_lsa
from .reference import mixer_attention
from .rsa import RSAProfile, prepare_rsa, secure_rsa


@dataclass
class HybridMaterial:
    base: object
    wide: object
    shape: tuple
    paths: tuple
    rsa: object
    lsa: object
    used: bool = False


def prepare_hybrid(base, wide, shape, head_types, rho=None,
                   fixed=FixedPointProfile(), rsa_profile=RSAProfile()):
    shape = tuple(int(x) for x in shape)
    if len(shape) != 4 or any(x <= 0 for x in shape):
        raise ValueError("shape must be positive (B,H,N,D)")
    if wide.base is not base:
        raise ValueError("wide runtime must share the base runtime")
    paths = normalize_head_types(head_types, shape[1])
    if fixed.fraction_bits != 17:
        raise ValueError("hybrid RSA and LSA require Q17 input shares")
    rho = shape[-1] ** -0.5 if rho is None else rho
    def group_shape(path):
        return (shape[0], paths.count(path), shape[2], shape[3])
    rsa = prepare_rsa(base, wide, group_shape("rsa"), rsa_profile) if "rsa" in paths else None
    lsa = prepare_lsa(base, group_shape("lsa"), rho, fixed) if "lsa" in paths else None
    return HybridMaterial(base, wide, shape, paths, rsa, lsa)


def secure_hybrid(base, wide, q, k, v, material, *, patch_grid=None,
                  has_cls_token=True, profiler=None):
    if material.used:
        raise RuntimeError("hybrid material already consumed")
    if material.base is not base or material.wide is not wide:
        raise ValueError("hybrid material belongs to another runtime")
    for name, value in (("q", q), ("k", k), ("v", v)):
        if tuple(value.shape) != material.shape:
            raise ValueError(f"{name} must have shape {material.shape}")
        _validate_share(base, value, name)
    material.used = True
    output = torch.empty_like(v)
    for path in ("rsa", "lsa", "mixer"):
        selected = [i for i, item in enumerate(material.paths) if item == path]
        if not selected:
            continue
        with profiler.phase(f"{path}_select") if profiler else nullcontext():
            index = torch.tensor(selected, dtype=torch.long, device=base.device)
            group_v = v.index_select(1, index)
            if path != "mixer":
                group_q, group_k = q.index_select(1, index), k.index_select(1, index)
        with profiler.phase(f"{path}_compute") if profiler else nullcontext():
            if path == "rsa":
                group_output = secure_rsa(base, wide, group_q, group_k, group_v, material.rsa)
            elif path == "lsa":
                group_output = secure_lsa(base, group_q, group_k, group_v, material.lsa)
            else:
                group_output = mixer_attention(group_v, patch_grid, has_cls_token)
        with profiler.phase(f"{path}_scatter") if profiler else nullcontext():
            output = output.index_copy(1, index, group_output)
    return output
