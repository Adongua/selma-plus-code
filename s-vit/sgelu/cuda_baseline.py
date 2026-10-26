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
    """One-use masked-Fourier material matching the SecFormer construction."""

    shape: tuple
    runtime: object
    mask_share: torch.Tensor
    mask_sin_share: torch.Tensor
    mask_cos_share: torch.Tensor
    truncations: list
    used: bool = False

    def consume(self, base, value):
        if self.used:
            raise RuntimeError("Fourier material already consumed")
        if base is not self.runtime or tuple(value.shape) != self.shape:
            raise ValueError("material belongs to another runtime or tensor shape")
        self.used = True


def prepare_fourier(base, shape):
    """Prepare SecFormer masked Fourier material on CUDA.

    The random mask is generated only at rank 0 and secret-shared.  Its seven
    sine/cosine tables are also secret-shared, so the online path can evaluate
    sine and cosine of the opened masked residue locally on CUDA.
    """
    shape = tuple(shape)
    if any(not isinstance(n, int) or n <= 0 for n in shape):
        raise ValueError("shape dimensions must be positive integers")
    if base.rank == 0:
        mask = (2 * math.pi * torch.rand(shape, device=base.device,
                                         generator=base.local_generator) - math.pi)
    else:
        mask = torch.zeros(shape, dtype=torch.float32, device=base.device)
    mask_q17 = (mask * (1 << 17)).round().to(torch.int64)
    harmonics = torch.arange(1, 8, device=base.device, dtype=torch.float64)
    angles = mask.double().unsqueeze(-1) * (harmonics * math.pi / 10.0)
    mask_sin = torch.sin(angles)
    mask_cos = torch.cos(angles)
    mask_sin_q17 = (mask_sin * (1 << 17)).round().to(torch.int64)
    mask_cos_q17 = (mask_cos * (1 << 17)).round().to(torch.int64)
    # The three tensors are shaped (..., 7) for the harmonic dimension.
    mask_share = base.private(mask_q17, 0)
    mask_sin_share = base.private(mask_sin_q17, 0)
    mask_cos_share = base.private(mask_cos_q17, 0)
    # z scaling; seven masked sine combinations; seven beta products; final
    # x*(1+erf) product; final multiplication by one half.
    truncations = [prob_trunc_material(base, shape, 17)]
    truncations.extend(prob_trunc_material(base, shape + (7,), 17) for _ in range(2))
    truncations.extend((prob_trunc_material(base, shape, 17),
                        prob_trunc_material(base, shape, 1)))
    return FourierMaterial(shape, base, mask_share, mask_sin_share,
                           mask_cos_share, truncations)


def fourier_gelu(base, x_q17, material, threshold=2.5):
    """CUDA masked-Fourier port of SecFormer's GeLU baseline.

    The Fourier table is evaluated with a masked opening and local CUDA
    ``sin``/``cos``.  Thresholds are currently derived from one additional
    public opening because this experimental Ring3 runtime has no comparison
    primitive; this is not a security-validated implementation.
    """
    material.consume(base, x_q17)
    z = prob_trunc(base, round((1.0 / math.sqrt(2.0)) * (1 << 17)) * x_q17,
                   17, material.truncations[0])
    # SecFormer's masked opening: delta = z - t.  The opened residue is
    # public, so all seven CUDA trigonometric evaluations are local.
    delta = base.reveal(z - material.mask_share).double() / (1 << 17)
    harmonics = torch.arange(1, 8, device=base.device, dtype=torch.float64)
    angles = delta.unsqueeze(-1) * (harmonics * math.pi / 10.0)
    p = (torch.sin(angles) * (1 << 17)).round().to(torch.int64)
    q = (torch.cos(angles) * (1 << 17)).round().to(torch.int64)
    masked_sine = material.mask_cos_share * p + material.mask_sin_share * q
    masked_sine = prob_trunc(base, masked_sine, 17, material.truncations[1])
    beta = torch.tensor(SECFORMER_BETA, dtype=torch.float64, device=base.device)
    beta_q17 = (beta * (1 << 17)).round().to(torch.int64)
    terms = masked_sine * beta_q17
    terms = prob_trunc(base, terms, 17, material.truncations[2])
    fourier = terms.sum(dim=-1)

    opened_z = base.reveal(z).double() / (1 << 17)
    lower = (opened_z < -threshold).to(torch.int64)
    upper = (opened_z < threshold).to(torch.int64)
    scale = 1 << 17
    branch = (base.public(-lower * scale) + (upper - lower) * fourier
              + base.public((1 - upper) * scale))
    output = prob_trunc(base, base.mul(x_q17, branch + base.public(scale)),
                        17, material.truncations[3])
    return prob_trunc(base, output, 1, material.truncations[4])
