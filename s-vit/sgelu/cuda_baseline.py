"""CUDA-native fixed-point baselines for the sGeLU microbenchmark.

The first baseline is the MPCFormer-style quadratic approximation
``0.125*x**2 + 0.25*x + 0.5`` evaluated through the same experimental
three-party CUDA ring as sGeLU.  This is an engineering port of the arithmetic
formula, not a claim that it has the same preprocessing or security model as
CrypTen's implementation.
"""

from dataclasses import dataclass

import torch

from .protocol import prob_trunc_batch, prob_trunc_material


@dataclass
class QuadMaterial:
    shape: tuple
    runtime: object
    truncations: dict
    used: bool = False

    def consume(self, base, value):
        if self.used:
            raise RuntimeError("Quad material already consumed")
        if base is not self.runtime or tuple(value.shape) != self.shape:
            raise ValueError("material belongs to another runtime or tensor shape")
        if value.dtype != torch.int64 or value.device != base.device:
            raise ValueError(f"expected int64 Q17 share on {base.device}")
        self.used = True


def prepare_quad(base, shape):
    """Prepare one-use truncation masks for the fixed-point Quad baseline."""
    shape = tuple(shape)
    if any(not isinstance(n, int) or n <= 0 for n in shape):
        raise ValueError("shape dimensions must be positive integers")
    specs = {"square": (shape, 20), "linear": (shape, 2)}
    truncations = {name: prob_trunc_material(base, size, shift)
                   for name, (size, shift) in specs.items()}
    return QuadMaterial(shape, base, truncations)


def quad_gelu(base, x_q17, material):
    """Evaluate the MPCFormer quadratic approximation and return Q17 shares."""
    material.consume(base, x_q17)
    square_q34 = base.square(x_q17)
    # x_q17^2 / 2^20 represents 0.125*x^2 in Q17; x_q17 / 4
    # represents 0.25*x.  Independent masks are opened together.
    quadratic_q17, linear_q17 = prob_trunc_batch(
        base, (square_q34, x_q17), (20, 2),
        (material.truncations["square"], material.truncations["linear"]),
    )
    return quadratic_q17 + linear_q17 + base.public(1 << 16)
