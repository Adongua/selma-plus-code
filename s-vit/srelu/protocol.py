"""Standalone Ruler-based secure smooth ReLU protocol.

The protocol follows the manuscript's two-stage convention:

    shifted = (x + sqrt(x^2 + r^2) - r) / 2
    corrected = shifted + r / 2

The public ``r/2`` correction is applied elementwise before RSA row summation
and normalization.  This module reproduces the experimental Ruler arithmetic;
it does not claim that the masked opening has a completed privacy proof.
"""

from dataclasses import dataclass
import math
from typing import Sequence, Tuple

import torch

from sgelu.protocol import (
    extend_online,
    extension_material,
    prob_trunc,
    prob_trunc_material,
    sample_scale,
)
from sgelu.wide import isqrt_from_limbs


@dataclass(frozen=True)
class SReLUProfile:
    radius: float = 0.0055
    input_fraction_bits: int = 17
    value_fraction_bits: int = 16
    scale_bits: int = 30
    reciprocal_scale_bits: int = 50
    reciprocal_work_fraction_bits: int = 15
    reciprocal_iteration_fraction_bits: int = 28
    reciprocal_iterations: int = 4
    input_bound: float = 16.0

    def __post_init__(self):
        if self.radius <= 0 or not math.isfinite(self.radius):
            raise ValueError("radius must be positive and finite")
        if self.input_fraction_bits != 17:
            raise ValueError("the current Ruler-sReLU profile requires Q17 input")
        if self.value_fraction_bits != 16:
            raise ValueError("the current Ruler-sReLU profile requires Q16 radicands")
        if self.scale_bits != 30 or self.reciprocal_scale_bits != 50:
            raise ValueError("the current reciprocal profile requires LA=30 and FB=50")
        if self.reciprocal_iterations <= 0:
            raise ValueError("reciprocal_iterations must be positive")
        if self.input_bound <= 0 or not math.isfinite(self.input_bound):
            raise ValueError("input_bound must be positive and finite")

    @property
    def input_scale(self) -> int:
        return 1 << self.input_fraction_bits

    @property
    def radius_squared_q16(self) -> int:
        return round(self.radius * self.radius * (1 << self.value_fraction_bits))

    @property
    def radius_q17(self) -> int:
        return round(self.radius * self.input_scale)

    @property
    def half_radius_q17(self) -> int:
        return round(0.5 * self.radius * self.input_scale)

    @property
    def square_shift(self) -> int:
        return 2 * self.input_fraction_bits - self.value_fraction_bits

    @property
    def sqrt_shift(self) -> int:
        return (
            self.value_fraction_bits // 2
            + self.reciprocal_scale_bits
            - self.input_fraction_bits
        )

    def encode(self, value: torch.Tensor, device=None) -> torch.Tensor:
        value = torch.as_tensor(value, dtype=torch.float64, device=device)
        if not torch.isfinite(value).all():
            raise ValueError("cannot encode a non-finite tensor")
        if torch.any(value.abs() >= self.input_bound):
            raise ValueError(f"input must satisfy |x| < {self.input_bound}")
        return torch.round(value * self.input_scale).to(torch.int64)

    def decode(self, value: torch.Tensor) -> torch.Tensor:
        return value.to(torch.float64) / self.input_scale

    def shifted_reference(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(value, dtype=torch.float64)
        return 0.5 * (
            value + torch.sqrt(value.square() + self.radius * self.radius) - self.radius
        )

    def corrected_reference(self, value: torch.Tensor) -> torch.Tensor:
        return self.shifted_reference(value) + 0.5 * self.radius

    def fixed_point_corrected_reference(self, value: torch.Tensor) -> torch.Tensor:
        """Reference using the public constants after protocol quantization."""

        value = torch.as_tensor(value, dtype=torch.float64)
        radius_squared = self.radius_squared_q16 / (1 << self.value_fraction_bits)
        radius = self.radius_q17 / self.input_scale
        half_radius = self.half_radius_q17 / self.input_scale
        shifted = 0.5 * (value + torch.sqrt(value.square() + radius_squared) - radius)
        return shifted + half_radius


def _validate_shape(shape: Sequence[int]) -> Tuple[int, ...]:
    shape = tuple(int(item) for item in shape)
    if not shape or any(item <= 0 for item in shape):
        raise ValueError("shape dimensions must be positive")
    return shape


def _truncate_now(base, value: torch.Tensor, shift: int) -> torch.Tensor:
    """Generate and immediately consume one offline truncation material."""

    material = prob_trunc_material(base, tuple(value.shape), shift)
    return prob_trunc(base, value, shift, material)


def _prepare_scale_reciprocal_q50(
    base,
    scale: torch.Tensor,
    profile: SReLUProfile,
) -> torch.Tensor:
    """Compute secret ``round(2^50 / scale)`` during preprocessing."""

    work_fraction = profile.reciprocal_work_fraction_bits
    reciprocal_fraction = profile.reciprocal_iteration_fraction_bits
    normalized = _truncate_now(
        base,
        scale,
        profile.scale_bits - work_fraction,
    )
    estimate = base.public(
        torch.full(
            tuple(scale.shape),
            round(2.9142 * (1 << reciprocal_fraction)),
            dtype=torch.int64,
            device=base.device,
        )
    ) - (normalized << (reciprocal_fraction + 1 - work_fraction))
    two = base.public(
        torch.full(
            tuple(scale.shape),
            2 << reciprocal_fraction,
            dtype=torch.int64,
            device=base.device,
        )
    )
    for _ in range(profile.reciprocal_iterations):
        product = _truncate_now(
            base,
            base.mul(normalized, estimate),
            work_fraction,
        )
        estimate = _truncate_now(
            base,
            base.mul(estimate, two - product),
            reciprocal_fraction,
        )
    final_shift = reciprocal_fraction - (
        profile.reciprocal_scale_bits - profile.scale_bits
    )
    if final_shift <= 0:
        raise ValueError("invalid reciprocal fixed-point profile")
    return _truncate_now(base, estimate, final_shift)


@dataclass
class SReLUMaterial:
    runtime: object
    wide_runtime: object
    shape: Tuple[int, ...]
    profile: SReLUProfile
    squared_scale96: torch.Tensor
    reciprocal_scale_q50: torch.Tensor
    value_extension: tuple
    value_truncation: tuple
    sqrt_truncation: tuple
    half_truncation: tuple
    used: bool = False

    def consume(self, base, wide, value: torch.Tensor) -> None:
        if self.used:
            raise RuntimeError("sReLU material already consumed")
        if base is not self.runtime or wide is not self.wide_runtime:
            raise ValueError("sReLU material belongs to another runtime")
        if tuple(value.shape) != self.shape:
            raise ValueError(f"expected input shape {self.shape}")
        if value.dtype != torch.int64 or value.device != base.device:
            raise ValueError(f"expected an int64 Q17 share on {base.device}")
        self.used = True


def prepare_srelu(
    base,
    wide,
    shape: Sequence[int],
    profile: SReLUProfile = SReLUProfile(),
) -> SReLUMaterial:
    """Prepare one-time material for one Ruler-sReLU invocation."""

    if wide.base is not base:
        raise ValueError("64-bit and 96-bit runtimes must share the same base")
    shape = _validate_shape(shape)
    scale = sample_scale(base, shape)
    squared_scale = base.square(scale)
    squared_scale96 = extend_online(
        base,
        wide,
        squared_scale,
        extension_material(base, wide, shape),
    )
    reciprocal_scale_q50 = _prepare_scale_reciprocal_q50(base, scale, profile)
    return SReLUMaterial(
        runtime=base,
        wide_runtime=wide,
        shape=shape,
        profile=profile,
        squared_scale96=squared_scale96,
        reciprocal_scale_q50=reciprocal_scale_q50,
        value_extension=extension_material(base, wide, shape),
        value_truncation=prob_trunc_material(base, shape, profile.square_shift),
        sqrt_truncation=prob_trunc_material(base, shape, profile.sqrt_shift),
        half_truncation=prob_trunc_material(base, shape, 1),
    )


def ruler_srelu_shifted(
    base,
    wide,
    value_q17: torch.Tensor,
    material: SReLUMaterial,
) -> torch.Tensor:
    """Return the shifted Q17 value ``(x + sqrt(x^2+r^2) - r) / 2``."""

    if wide.base is not base:
        raise ValueError("64-bit and 96-bit runtimes must share the same base")
    material.consume(base, wide, value_q17)
    profile = material.profile
    square_q34 = base.square(value_q17)
    value_q16 = prob_trunc(
        base,
        square_q34,
        profile.square_shift,
        material.value_truncation,
    ) + base.public(
        torch.full(
            material.shape,
            profile.radius_squared_q16,
            dtype=torch.int64,
            device=base.device,
        )
    )
    value96 = extend_online(
        base,
        wide,
        value_q16,
        material.value_extension,
    )
    opened = wide.reveal(wide.mul(material.squared_scale96, value96))
    root = isqrt_from_limbs(opened).reshape(material.shape)
    if torch.any(root <= 0):
        raise ArithmeticError("nonpositive opened root in Ruler-sReLU")

    # ``root`` is public after the masked opening, so multiplying every secret
    # reciprocal share by it is local and does not need a secure multiplication.
    scaled_root = material.reciprocal_scale_q50 * root
    sqrt_q17 = prob_trunc(
        base,
        scaled_root,
        profile.sqrt_shift,
        material.sqrt_truncation,
    )
    radius = base.public(
        torch.full(
            material.shape,
            profile.radius_q17,
            dtype=torch.int64,
            device=base.device,
        )
    )
    return prob_trunc(
        base,
        value_q17 + sqrt_q17 - radius,
        1,
        material.half_truncation,
    )


def add_public_half_radius(
    base,
    shifted_q17: torch.Tensor,
    profile: SReLUProfile,
) -> torch.Tensor:
    """Apply the manuscript's public ``r/2`` correction elementwise."""

    correction = torch.full(
        tuple(shifted_q17.shape),
        profile.half_radius_q17,
        dtype=torch.int64,
        device=base.device,
    )
    return shifted_q17 + base.public(correction)


def ruler_srelu(
    base,
    wide,
    value_q17: torch.Tensor,
    material: SReLUMaterial,
) -> torch.Tensor:
    """Return corrected Q17 sReLU, ready for RSA row normalization."""

    shifted = ruler_srelu_shifted(base, wide, value_q17, material)
    return add_public_half_radius(base, shifted, material.profile)
