"""Experimental wider-denominator RSA. Original rsa.py remains unchanged.

Requires scores |s|<16 and reconstructed Q17 row denominators in
[0.0005, 2048]. Ranges are contracts, not checks on secret data. Ruler openings
are not privacy validated. Low denominator accuracy must be measured end to end.
"""
from dataclasses import dataclass
import math

import torch

from sgelu.protocol import prob_trunc, prob_trunc_material
from srelu.protocol import SReLUProfile, prepare_srelu, ruler_srelu
from .protocol import _validate_share, secure_matmul


@dataclass(frozen=True)
class WideRSAProfile:
    epsilon: float = 1e-5
    radius: float = 0.0055
    score_bound: float = 16.0
    denominator_min: float = 0.0005
    denominator_max: float = 2048.0
    iterations: int = 32
    reciprocal_bits: int = 24

    def __post_init__(self):
        if (not math.isfinite(self.epsilon) or not math.isfinite(self.radius) or
                self.epsilon <= 0 or self.radius <= 0 or
                self.score_bound != 16 or self.denominator_min != 0.0005 or
                self.denominator_max != 2048 or self.iterations != 32 or
                self.reciprocal_bits != 24 or round(self.epsilon * 2**17) < 1):
            raise ValueError("invalid wide RSA profile")


@dataclass
class WideReciprocalMaterial:
    base: object
    shape: tuple
    profile: WideRSAProfile
    steps: tuple
    used: bool = False


def prepare_reciprocal(base, shape, profile=WideRSAProfile()):
    shape = tuple(shape)
    if not shape or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("positive reciprocal shape required")
    steps = tuple((prob_trunc_material(base, shape, 17),
                   prob_trunc_material(base, shape, profile.reciprocal_bits))
                  for _ in range(profile.iterations))
    return WideReciprocalMaterial(base, shape, profile, steps)


def wide_reciprocal(base, denominator, materials, profile=WideRSAProfile()):
    """Q17 denominator -> Q24 reciprocal, with one-use truncation materials."""
    if (materials.used or materials.base is not base or materials.profile != profile or
            tuple(denominator.shape) != materials.shape):
        raise ValueError("reciprocal material already used or incompatible")
    _validate_share(base, denominator, "denominator")
    materials.used = True
    q = 1 << profile.reciprocal_bits
    estimate = base.public(torch.full_like(denominator, round(q / profile.denominator_max)))
    two = base.public(2 * q)
    for product_material, update_material in materials.steps:
        product = prob_trunc(base, base.mul(denominator, estimate), 17, product_material)
        estimate = prob_trunc(base, base.mul(estimate, two - product),
                              profile.reciprocal_bits, update_material)
    return estimate


@dataclass
class WideRSAMaterial:
    base: object
    wide: object
    shape: tuple
    profile: WideRSAProfile
    srelu: object
    reciprocal: WideReciprocalMaterial
    truncations: dict
    used: bool = False


def prepare_wide_rsa(base, wide, shape, profile=WideRSAProfile()):
    shape = tuple(int(x) for x in shape)
    if len(shape) != 4 or any(x <= 0 for x in shape) or wide.base is not base:
        raise ValueError("positive B,H,N,D shape and matching runtimes required")
    b, h, n, _ = shape
    score, rows = (b, h, n, n), (b, h, n, 1)
    srelu = prepare_srelu(base, wide, score, SReLUProfile(radius=profile.radius,
                                                           input_bound=profile.score_bound))
    specs = {"score": (score, 17), "score_scale": (score, 20),
             "weight": (score, profile.reciprocal_bits), "output": (shape, 17)}
    truncations = {name: (shift, prob_trunc_material(base, size, shift))
                   for name, (size, shift) in specs.items()}
    return WideRSAMaterial(base, wide, shape, profile, srelu,
                           prepare_reciprocal(base, rows, profile), truncations)


def wide_rsa(base, wide, q, k, v, material):
    if material.used or material.base is not base or material.wide is not wide:
        raise ValueError("material already used or runtime mismatch")
    for name, value in (("q", q), ("k", k), ("v", v)):
        if tuple(value.shape) != material.shape:
            raise ValueError(f"{name} shape mismatch")
        _validate_share(base, value, name)
    material.used = True
    def trunc(name, value):
        shift, correlation = material.truncations[name]
        return prob_trunc(base, value, shift, correlation)
    score = trunc("score", secure_matmul(base, q, k.transpose(-2, -1)))
    score = trunc("score_scale", score * round(material.shape[-1] ** -0.5 * (1 << 20)))
    positive = ruler_srelu(base, wide, score, material.srelu)
    denominator = positive.sum(-1, keepdim=True) + base.public(round(material.profile.epsilon * 2**17))
    inverse = wide_reciprocal(base, denominator, material.reciprocal, material.profile)
    weights = trunc("weight", base.mul(positive, inverse))
    return trunc("output", secure_matmul(base, weights, v))
