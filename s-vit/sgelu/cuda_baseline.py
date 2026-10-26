"""CUDA-native fixed-point baselines for the sGeLU microbenchmark.

The first baseline is the MPCFormer-style quadratic approximation
``0.125*x**2 + 0.25*x + 0.5`` evaluated through the same experimental
three-party CUDA ring as sGeLU.  This is an engineering port of the arithmetic
formula, not a claim that it has the same preprocessing or security model as
CrypTen's implementation.
"""

import math
from dataclasses import dataclass

import torch

from .baseline import SECFORMER_BETA
from .protocol import prob_trunc, prob_trunc_batch, prob_trunc_material


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


@dataclass
class ErfMaterial:
    """One-use truncation material for CrypTen's Taylor-erf computation."""

    shape: tuple
    runtime: object
    truncations: list
    iterations: int
    used: bool = False

    def consume(self, base, value):
        if self.used:
            raise RuntimeError("erf material already consumed")
        if base is not self.runtime or tuple(value.shape) != self.shape:
            raise ValueError("material belongs to another runtime or tensor shape")
        self.used = True


def _q17_constant(base, value, shape):
    return base.public(round(float(value) * (1 << 17))).expand(shape)


def prepare_erf(base, shape, iterations=8):
    """Prepare masks for CrypTen's ``erf_iterations`` Taylor approximation."""
    shape = tuple(shape)
    if iterations < 1:
        raise ValueError("iterations must be positive")
    # z scaling, then two powers and one coefficient multiplication per term,
    # final erf scaling, x*(erf+1), and the final factor one half.
    shifts = [17]
    for _ in range(iterations):
        shifts.extend((17, 17, 17))
    shifts.extend((17, 17, 1))
    return ErfMaterial(shape, base,
                       [prob_trunc_material(base, shape, shift) for shift in shifts],
                       iterations)


def _erf_take(material):
    item = material.truncations.pop(0)
    return item


def erf_gelu(base, x_q17, material):
    """CUDA Ring3 port of CrypTen's Taylor-erf GeLU baseline."""
    material.consume(base, x_q17)
    # z = x / sqrt(2), represented in Q17.
    z = prob_trunc(base, round((1.0 / math.sqrt(2.0)) * (1 << 17)) * x_q17,
                   17, _erf_take(material))
    power = z
    erf_value = z
    for n in range(1, material.iterations + 1):
        power = prob_trunc(base, base.mul(power, z), 17, _erf_take(material))
        power = prob_trunc(base, base.mul(power, z), 17, _erf_take(material))
        coefficient = ((-1) ** n) / (math.factorial(n) * (2 * n + 1))
        term = prob_trunc(base, round(coefficient * (1 << 17)) * power,
                          17, _erf_take(material))
        erf_value = erf_value + term
    erf_value = prob_trunc(
        base, round((2.0 / math.sqrt(math.pi)) * (1 << 17)) * erf_value,
        17, _erf_take(material),
    )
    product = prob_trunc(base, base.mul(x_q17, erf_value + base.public(1 << 17)),
                         17, _erf_take(material))
    return prob_trunc(base, product, 1, _erf_take(material))


@dataclass
class FourierMaterial:
    shape: tuple
    runtime: object
    truncations: list
    iterations: int
    used: bool = False

    def consume(self, base, value):
        if self.used:
            raise RuntimeError("Fourier material already consumed")
        if base is not self.runtime or tuple(value.shape) != self.shape:
            raise ValueError("material belongs to another runtime or tensor shape")
        self.used = True


def prepare_fourier(base, shape, iterations=8):
    """Prepare masks for a direct secure Fourier evaluation."""
    shape = tuple(shape)
    if iterations < 1:
        raise ValueError("iterations must be positive")
    # Seven harmonics: angle scaling, complex exponential initialization,
    # (2 squares + 1 multiply) per subsequent squaring, coefficient scaling.
    count = 7 * (1 + 2 + 3 * (iterations - 1) + 1) + 2
    return FourierMaterial(shape, base,
                           [prob_trunc_material(base, shape, 17) for _ in range(count)],
                           iterations)


def _fourier_take(material):
    return material.truncations.pop(0)


def _cossin_q17(base, value, material):
    """Match CrypTen's repeated-squaring approximation to exp(i*value)."""
    im = prob_trunc(base, value, material.iterations, _fourier_take(material))
    re = base.public(1 << 17).expand(value.shape)
    re = re - prob_trunc(base, base.mul(im, im), 17, _fourier_take(material))
    im = im * 2
    for _ in range(material.iterations - 1):
        a2 = prob_trunc(base, base.mul(re, re), 17, _fourier_take(material))
        b2 = prob_trunc(base, base.mul(im, im), 17, _fourier_take(material))
        im = prob_trunc(base, base.mul(im, re), 17, _fourier_take(material)) * 2
        re = a2 - b2
    return re, im


def fourier_gelu(base, x_q17, material, threshold=2.5):
    """CUDA port of the direct SecFormer Fourier arithmetic baseline.

    The Fourier terms remain secret-shared and use the same repeated-squaring
    trigonometric approximation as CrypTen.  The current CUDA prototype opens
    only x for the two interval masks because it has no comparison primitive;
    this is therefore a public-threshold reference, not a security claim.
    """
    material.consume(base, x_q17)
    shape = tuple(x_q17.shape)
    opened_x = base.reveal(x_q17).double() / (1 << 17)
    lower = (opened_x / math.sqrt(2.0) < -threshold).to(torch.int64)
    upper = (opened_x / math.sqrt(2.0) < threshold).to(torch.int64)
    lower = lower.to(base.device)
    upper = upper.to(base.device)
    fourier = None
    for harmonic, beta in enumerate(SECFORMER_BETA, start=1):
        angle_coeff = harmonic * math.pi / (10.0 * math.sqrt(2.0))
        angle = prob_trunc(base,
                           round(angle_coeff * (1 << 17)) * x_q17,
                           17, _fourier_take(material))
        _, sine = _cossin_q17(base, angle, material)
        term = prob_trunc(base, round(beta * (1 << 17)) * sine,
                          17, _fourier_take(material))
        fourier = term if fourier is None else fourier + term
    scale = 1 << 17
    branch = (base.public(-lower * scale) + (upper - lower) * fourier
              + base.public((1 - upper) * scale))
    output = prob_trunc(base, base.mul(x_q17, branch + base.public(1 << 17)),
                        17, _fourier_take(material))
    return prob_trunc(base, output, 1, _fourier_take(material))
