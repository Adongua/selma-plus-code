"""Q17 LayerNorm with fused Q34 reduction and a Q16 variance.

The Ruler branch opens a^2 V and inherits its unresolved privacy limitations.
The Newton branch is a same-runtime engineering baseline, not an external SOTA.
All affine parameters are secret shares; material is one-use only.
"""

from dataclasses import dataclass
import math

import torch

from sgelu.protocol import (prob_trunc_material, prob_trunc, sample_scale,
                            extension_material, extend_online)
from sgelu.wide import isqrt_from_limbs


@dataclass(frozen=True)
class Profile:
    epsilon: float = 1e-5
    scale: float = 1.0
    input_bound: float = 4.0
    newton_steps: int = 32

    @property
    def epsilon_integer(self):
        return max(1, round(self.epsilon / self.scale**2 * 2**16))

    def validate(self, shape):
        if len(shape) < 1 or any(type(n) is not int or n < 1 for n in shape):
            raise ValueError("positive shape required; last axis is hidden width")
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.epsilon, self.scale, self.input_bound)):
            raise ValueError("finite positive epsilon, scale and bound required")
        if not (1 <= self.scale <= 4 and self.input_bound <= 4):
            raise ValueError("validated profile requires scale in [1,4], |x|<=4")
        if not (1 <= self.newton_steps <= 40) or self.epsilon_integer > 65536:
            raise ValueError("invalid Newton steps or epsilon outside profile")
        d = shape[-1]
        m = round(2**20 / d)
        mean_bound = self.input_bound * d * m / 2**20 + 2**-17
        z_bound = math.ceil((self.input_bound + mean_bound) / self.scale * 2**17) + 2
        if m < 1 or m * d * z_bound**2 >= 2**62:
            raise ValueError("fused accumulator exceeds truncation range")
        if (self.input_bound + mean_bound) * 2**17 * round(2**24 / self.scale) >= 2**62:
            raise ValueError("scale multiplication exceeds truncation range")


@dataclass
class Material:
    runtime: object
    wide: object
    shape: tuple
    profile: Profile
    method: str
    truncations: dict
    ruler: tuple | None
    used: bool = False


METHODS = ("fused_ruler", "unfused_ruler", "fused_newton",
           "unfused_newton", "secformer_goldschmidt")


def prepare(base, wide, shape, profile=Profile(), method="fused_ruler"):
    shape = tuple(shape)
    profile.validate(shape)
    if method not in METHODS or wide.base is not base:
        raise ValueError("invalid method or runtime")
    token = shape[:-1] + (1,)
    specs = {"mean": (token, 20), "scale": (shape, 24),
             "normalize": (shape, 24), "affine": (shape, 17)}
    if method in ("unfused_ruler", "unfused_newton"):
        specs.update(square=(shape, 17), variance=(token, 21))
    else:
        specs["variance"] = (token, 38)
    ruler = None
    if method.endswith("ruler"):
        a = sample_scale(base, token)
        a2 = extend_online(base, wide, base.square(a), extension_material(base, wide, token))
        ruler = (a, a2, extension_material(base, wide, token))
        specs["inverse"] = (token, 24)
    elif method in ("fused_newton", "unfused_newton"):
        for i in range(profile.newton_steps):
            for name, shift in (("vr", 16), ("vrr", 24), ("update", 25)):
                specs[f"{name}_{i}"] = (token, shift)
    else:
        for i in range(11):
            specs[f"m_{i}"] = (token, 1)
            specs[f"msquare_{i}"] = (token, 24)
            specs[f"q_{i}"] = (token, 24)
            specs[f"p_{i}"] = (token, 24)
        specs["gold_final"] = (token, 2)
    truncations = {name: (shift, prob_trunc_material(base, size, shift))
                   for name, (size, shift) in specs.items()}
    return Material(base, wide, shape, profile, method, truncations, ruler)


def layernorm(base, wide, x, gamma, beta, material):
    """Return Q17 output shares; input bound and |gamma|,|beta|<=2 are contracts."""
    if material.used:
        raise RuntimeError("LayerNorm material already consumed")
    if material.runtime is not base or material.wide is not wide or tuple(x.shape) != material.shape:
        raise ValueError("material/runtime/shape mismatch")
    if tuple(gamma.shape) != (x.shape[-1],) or beta.shape != gamma.shape:
        raise ValueError("affine parameters must have shape (hidden_width,)")
    for value in (x, gamma, beta):
        if value.dtype != torch.int64 or value.device != base.device:
            raise ValueError(f"int64 Q17 shares on {base.device} required")
    material.used = True
    p = material.profile

    def trunc(name, value):
        shift, correlation = material.truncations[name]
        return prob_trunc(base, value, shift, correlation)

    m = round(2**20 / x.shape[-1])
    mean = trunc("mean", x.sum(-1, keepdim=True) * m)
    z = trunc("scale", (x - mean) * round(2**24 / p.scale))
    squares = base.square(z)
    if material.method in ("unfused_ruler", "unfused_newton"):
        squares = trunc("square", squares)
    v = trunc("variance", squares.sum(-1, keepdim=True) * m)
    v = v + base.public(p.epsilon_integer)
    if material.ruler is not None:
        a, a2, extension = material.ruler
        v96 = extend_online(base, wide, v, extension)
        roots = isqrt_from_limbs(wide.reveal(wide.mul(a2, v96)))
        if torch.any(roots <= 0):
            raise ArithmeticError("nonpositive variance root")
        reciprocal = torch.div(2**56 + roots // 2, roots, rounding_mode="floor")
        r = trunc("inverse", a * reciprocal)
    elif material.method in ("fused_newton", "unfused_newton"):
        r = base.public(torch.full(v.shape, 2**21, dtype=torch.int64,
                                   device=v.device))  # 0.125 in Q24
        for i in range(p.newton_steps):
            vr = trunc(f"vr_{i}", base.mul(v, r))
            vrr = trunc(f"vrr_{i}", base.mul(vr, r))
            r = trunc(f"update_{i}", base.mul(r, base.public(3 * 2**24) - vrr))
    else:
        # SecFormer Algorithm 2: Goldschmidt inverse square root with 11
        # iterations. eta=16 keeps q0 bounded for the |x|<=4 benchmark.
        q = v * 16  # Q16 -> Q24 after division by eta=16
        p_gold = base.public(torch.full(v.shape, 2**24, dtype=torch.int64))
        for i in range(11):
            m = trunc(f"m_{i}", base.public(3 * 2**24) - q)
            m_square = trunc(f"msquare_{i}", base.square(m))
            q = trunc(f"q_{i}", base.mul(q, m_square))
            p_gold = trunc(f"p_{i}", base.mul(p_gold, m))
        r = trunc("gold_final", p_gold)  # divide by sqrt(eta)=4
    normalized = trunc("normalize", base.mul(z, r))
    return trunc("affine", base.mul(normalized, gamma)) + beta
