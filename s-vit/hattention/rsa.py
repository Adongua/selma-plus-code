"""Experimental three-party RSA with a bounded Newton row reciprocal.

The caller must guarantee that every secret score has absolute value below
``score_bound`` and every corrected-sReLU row denominator is within the
declared positive interval. These secret conditions are not checked here.
This is an engineering reciprocal, not a security claim or a reproduction of
an unspecified manuscript reciprocal protocol.
"""

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from sgelu.protocol import prob_trunc, prob_trunc_material
from srelu.protocol import SReLUProfile, prepare_srelu, ruler_srelu
from .protocol import _validate_share, secure_matmul


@dataclass(frozen=True)
class RSAProfile:
    epsilon: float = 1e-5
    radius: float = 0.0055
    score_bound: float = 16.0
    denominator_min: float = 0.01
    denominator_max: float = 64.0
    reciprocal_fraction_bits: int = 24
    iterations: int = 24
    score_scale_fraction_bits: int = 20

    def __post_init__(self):
        for name in ("epsilon", "radius", "score_bound", "denominator_min", "denominator_max"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.score_bound > 16:
            raise ValueError("sReLU requires score_bound <= 16")
        if self.denominator_min >= self.denominator_max:
            raise ValueError("denominator_min must be below denominator_max")
        if self.denominator_max > 64 or self.denominator_min < 0.01:
            raise ValueError("this experimental profile requires 0.01 <= denominator <= 64")
        if self.reciprocal_fraction_bits != 24 or self.score_scale_fraction_bits != 20:
            raise ValueError("this implementation requires Q24 reciprocal and Q20 scale")
        if not 1 <= self.iterations <= 32:
            raise ValueError("iterations must be in [1,32]")
        if round(self.epsilon * (1 << 17)) == 0:
            raise ValueError("epsilon must be representable in Q17")

    @property
    def effective_epsilon(self):
        return round(self.epsilon * (1 << 17)) / (1 << 17)


@dataclass
class RSAMaterial:
    base: object
    wide: object
    shape: tuple
    profile: RSAProfile
    srelu: object
    truncations: dict
    used: bool = False


def prepare_rsa(base, wide, shape: Sequence[int], profile: RSAProfile = RSAProfile()):
    """Generate one-use preprocessing for a (B,H,N,D) RSA invocation."""
    shape = tuple(int(x) for x in shape)
    if len(shape) != 4 or any(x <= 0 for x in shape):
        raise ValueError("shape must be positive (B,H,N,D)")
    if wide.base is not base:
        raise ValueError("wide runtime must use the same Ring3 base")
    b, h, n, d = shape
    score_shape = (b, h, n, n)
    row_shape = (b, h, n, 1)
    srelu_profile = SReLUProfile(radius=profile.radius, input_bound=profile.score_bound)
    srelu_material = prepare_srelu(base, wide, score_shape, srelu_profile)
    specs = {
        "score": (score_shape, 17),
        "score_scale": (score_shape, profile.score_scale_fraction_bits),
        "weight": (score_shape, profile.reciprocal_fraction_bits),
        "output": (shape, 17),
    }
    for i in range(profile.iterations):
        specs[f"reciprocal_product_{i}"] = (row_shape, 17)
        specs[f"reciprocal_update_{i}"] = (row_shape, profile.reciprocal_fraction_bits)
    truncations = {name: prob_trunc_material(base, target, shift)
                   for name, (target, shift) in specs.items()}
    return RSAMaterial(base, wide, shape, profile, srelu_material, truncations)


def secure_rsa(base, wide, q, k, v, material: RSAMaterial):
    """Return Q17 RSA shares; no secret range checks or output openings."""
    if material.used:
        raise RuntimeError("RSA material already consumed")
    if material.base is not base or material.wide is not wide:
        raise ValueError("RSA material belongs to another runtime")
    for name, value in (("q", q), ("k", k), ("v", v)):
        if tuple(value.shape) != material.shape:
            raise ValueError(f"{name} must have shape {material.shape}")
        _validate_share(base, value, name)
    material.used = True
    profile = material.profile
    truncations = material.truncations

    def trunc(name, value, shift):
        return prob_trunc(base, value, shift, truncations[name])

    score = trunc("score", secure_matmul(base, q, k.transpose(-2, -1)), 17)
    scale = round((material.shape[-1] ** -0.5) * (1 << profile.score_scale_fraction_bits))
    score = trunc("score_scale", score * scale, profile.score_scale_fraction_bits)
    positive = ruler_srelu(base, wide, score, material.srelu)
    denominator = positive.sum(dim=-1, keepdim=True) + base.public(
        round(profile.epsilon * (1 << 17))
    )

    q24 = 1 << profile.reciprocal_fraction_bits
    estimate = base.public(torch.full_like(denominator, round(q24 / profile.denominator_max)))
    two = base.public(2 * q24)
    for i in range(profile.iterations):
        product = trunc(f"reciprocal_product_{i}", base.mul(denominator, estimate), 17)
        estimate = trunc(f"reciprocal_update_{i}", base.mul(estimate, two - product),
                         profile.reciprocal_fraction_bits)
    weights = trunc("weight", base.mul(positive, estimate), profile.reciprocal_fraction_bits)
    return trunc("output", secure_matmul(base, weights, v), 17)
