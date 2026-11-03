"""Experimental wide-input LayerNorm; original protocol.py stays unchanged.

Q17 input/output, Q21 centered value and Q28 scaled variance. Requires |x|<=64,
|gamma|,|beta|<=4; secret ranges are not checked. Ruler privacy is unvalidated.
"""
from dataclasses import dataclass
import math

import torch

from sgelu.protocol import (prob_trunc_material, prob_trunc, sample_scale,
                            extension_material, extend_online)
from .public_isqrt96 import public_isqrt96


@dataclass(frozen=True)
class WideLayerNormProfile:
    epsilon: float = 1e-5
    input_bound: float = 64.0
    scale: float = 16.0
    affine_bound: float = 4.0
    variance_bits: int = 28

    @property
    def epsilon_integer(self):
        return round(self.epsilon / self.scale**2 * (1 << self.variance_bits))

    @property
    def effective_epsilon(self):
        return self.epsilon_integer * self.scale**2 / (1 << self.variance_bits)

    def validate(self, shape):
        if not shape or any(type(n) is not int or n < 1 for n in shape):
            raise ValueError("positive shape required")
        if (not math.isfinite(self.epsilon) or self.epsilon <= 0 or
                self.input_bound != 64 or self.scale != 16 or self.affine_bound != 4 or
                self.variance_bits != 28 or self.epsilon_integer < 1):
            raise ValueError("invalid wide LayerNorm profile")
        d = shape[-1]
        m = round(2**20 / d)
        mean_coefficient = round(2**30 / d)
        mean_bound = self.input_bound * d * mean_coefficient / 2**30 + 2**-17
        z_bound = math.ceil((self.input_bound + mean_bound) / self.scale * 2**21) + 2
        if (m < 1 or mean_coefficient < 1 or z_bound**2 >= 2**62 or
                m * d * ((z_bound**2 >> 14) + 2) >= 2**62):
            raise ValueError("variance accumulator exceeds truncation domain")
        # v_Q28 * a^2 must fit the unsigned 96-bit Ruler arithmetic.
        variance_bound = (m * d * ((z_bound**2 >> 14) + 2) >> 20) + self.epsilon_integer + 2
        if variance_bound * (1 << 60) >= 1 << 96:
            raise ValueError("masked variance exceeds the 96-bit domain")


@dataclass
class WideLayerNormMaterial:
    base: object
    wide: object
    shape: tuple
    profile: WideLayerNormProfile
    truncations: dict
    ruler: tuple
    used: bool = False


def prepare_wide_layernorm(base, wide, shape, profile=WideLayerNormProfile()):
    shape = tuple(shape)
    profile.validate(shape)
    if wide.base is not base:
        raise ValueError("wide runtime must share the base runtime")
    token = shape[:-1] + (1,)
    a = sample_scale(base, token)
    a2 = extend_online(base, wide, base.square(a), extension_material(base, wide, token))
    ruler = a, a2, extension_material(base, wide, token)
    specs = {"mean": (token, 30), "square": (shape, 14), "variance": (token, 20),
             "inverse": (token, 24), "normalize": (shape, 28), "affine": (shape, 17)}
    truncations = {name: (shift, prob_trunc_material(base, size, shift))
                   for name, (size, shift) in specs.items()}
    return WideLayerNormMaterial(base, wide, shape, profile, truncations, ruler)


def wide_layernorm(base, wide, x, gamma, beta, material):
    if material.used or material.base is not base or material.wide is not wide:
        raise ValueError("material already used or runtime mismatch")
    if tuple(x.shape) != material.shape or tuple(gamma.shape) != (x.shape[-1],) or beta.shape != gamma.shape:
        raise ValueError("tensor shape mismatch")
    if any(t.dtype != torch.int64 or t.device != base.device for t in (x, gamma, beta)):
        raise ValueError("int64 Q17 shares on the runtime device are required")
    material.used = True
    def trunc(name, value):
        shift, correlation = material.truncations[name]
        return prob_trunc(base, value, shift, correlation)
    p = material.profile
    m = round(2**20 / x.shape[-1])
    mean = trunc("mean", x.sum(-1, keepdim=True) * round(2**30 / x.shape[-1]))
    # (x-mean) is Q17; it also represents (x-mean)/16 in Q21.
    z = x - mean
    squares = trunc("square", base.square(z))  # Q42 -> Q28
    variance = trunc("variance", squares.sum(-1, keepdim=True) * m)
    variance = variance + base.public(p.epsilon_integer)
    a, a2, extension = material.ruler
    v96 = extend_online(base, wide, variance, extension)
    roots = public_isqrt96(wide.reveal(wide.mul(a2, v96)))
    if torch.any(roots <= 0):
        raise ArithmeticError("nonpositive masked variance root")
    reciprocal = torch.div((1 << 62) + roots // 2, roots, rounding_mode="floor")
    inverse = trunc("inverse", a * reciprocal)
    normalized = trunc("normalize", base.mul(z, inverse))  # Q21 * Q24 -> Q17
    return trunc("affine", base.mul(normalized, gamma)) + beta
