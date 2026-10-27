"""Ring3 building blocks for secure hAttention.

This phase implements the linear LSA and public Mixer paths on both the
CPU/Gloo and CUDA/NCCL Ring3 runtimes.  RSA is intentionally rejected until
its sReLU and reciprocal protocols are wired and validated.
"""

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import torch

from sgelu.protocol import prob_trunc, prob_trunc_material

from .config import normalize_head_types
from .reference import mixer_attention


@dataclass(frozen=True)
class FixedPointProfile:
    fraction_bits: int = 17
    coefficient_fraction_bits: int = 30
    input_bound: float = 8.0

    def __post_init__(self):
        if not 1 <= self.fraction_bits <= 24:
            raise ValueError("fraction_bits must be in [1,24]")
        if not self.fraction_bits <= self.coefficient_fraction_bits <= 40:
            raise ValueError(
                "coefficient_fraction_bits must be in [fraction_bits,40]"
            )
        if self.input_bound <= 0 or not math.isfinite(self.input_bound):
            raise ValueError("input_bound must be positive and finite")

    @property
    def scale(self) -> int:
        return 1 << self.fraction_bits

    def encode(self, value: torch.Tensor, device=None) -> torch.Tensor:
        value = torch.as_tensor(value, dtype=torch.float64, device=device)
        if not torch.isfinite(value).all():
            raise ValueError("cannot encode a non-finite tensor")
        if torch.any(value.abs() > self.input_bound):
            raise ValueError(
                f"input exceeds fixed-point bound +/-{self.input_bound}"
            )
        return torch.round(value * self.scale).to(torch.int64)

    def decode(self, value: torch.Tensor) -> torch.Tensor:
        return value.to(torch.float64) / self.scale


def _validate_share(base, value: torch.Tensor, name: str) -> None:
    if value.dtype != torch.int64:
        raise ValueError(f"{name} must be an int64 fixed-point share")
    if value.device != base.device:
        raise ValueError(f"{name} must be on {base.device}, got {value.device}")


def _ring_matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Integer matmul with a bounded-memory CUDA compatibility fallback."""

    try:
        return torch.matmul(left, right)
    except RuntimeError as error:
        message = str(error).lower()
        unsupported_long = (
            left.is_cuda
            and ("not implemented" in message or "unsupported" in message)
            and ("long" in message or "int64" in message)
        )
        if not unsupported_long:
            raise

    # Some PyTorch/CUDA builds do not expose an int64 GEMM kernel.  Preserve
    # exact ring arithmetic with blocked multiply-reduce operations that stay
    # on CUDA and avoid a full (..., M, K, N) temporary.
    if left.shape[:-2] != right.shape[:-2]:
        raise ValueError("the CUDA int64 fallback requires equal batch dimensions")
    output = torch.zeros(
        left.shape[:-2] + (left.shape[-2], right.shape[-1]),
        dtype=torch.int64,
        device=left.device,
    )
    block_size = 32
    for start in range(0, left.shape[-1], block_size):
        stop = min(start + block_size, left.shape[-1])
        products = (
            left[..., :, start:stop].unsqueeze(-1)
            * right[..., start:stop, :].unsqueeze(-3)
        )
        output = output + products.sum(dim=-2)
    return output


def secure_matmul(base, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """One-round Ring3 batched matrix multiplication.

    This is the matrix analogue of ``Ring3.mul``.  It exchanges predecessor
    shares once and evaluates the three local cross terms.  The output remains
    secret shared and is refreshed without communication.
    """

    _validate_share(base, left, "left")
    _validate_share(base, right, "right")
    if left.ndim < 2 or right.ndim < 2:
        raise ValueError("secure_matmul requires tensors with at least two dimensions")
    if left.shape[-1] != right.shape[-2]:
        raise ValueError(
            f"matmul dimensions do not align: {left.shape[-1]} != {right.shape[-2]}"
        )
    left_predecessor, right_predecessor = base.exchange([left, right])
    result = (
        _ring_matmul(left, right)
        + _ring_matmul(left, right_predecessor)
        + _ring_matmul(left_predecessor, right)
    )
    return base.refresh(result)


@dataclass
class LSAMaterial:
    runtime: object
    q_shape: Tuple[int, int, int, int]
    value_dim: int
    profile: FixedPointProfile
    rho: float
    kv_truncation: tuple
    qkv_truncation: tuple
    scale_truncation: tuple
    used: bool = False

    def consume(self, base, q, k, v):
        if self.used:
            raise RuntimeError("LSA material already consumed")
        if base is not self.runtime:
            raise ValueError("LSA material belongs to another runtime")
        expected_q = self.q_shape
        expected_v = self.q_shape[:-1] + (self.value_dim,)
        if tuple(q.shape) != expected_q or tuple(k.shape) != expected_q:
            raise ValueError(f"expected q/k shape {expected_q}")
        if tuple(v.shape) != expected_v:
            raise ValueError(f"expected v shape {expected_v}")
        for name, tensor in (("q", q), ("k", k), ("v", v)):
            _validate_share(base, tensor, name)
        self.used = True


def prepare_lsa(
    base,
    q_shape: Sequence[int],
    rho: float,
    profile: FixedPointProfile = FixedPointProfile(),
    value_dim: Optional[int] = None,
) -> LSAMaterial:
    """Generates one-time truncation material for an LSA invocation."""

    q_shape = tuple(int(item) for item in q_shape)
    if len(q_shape) != 4 or any(item <= 0 for item in q_shape):
        raise ValueError("q_shape must be positive (B,H,N,D)")
    if not math.isfinite(rho):
        raise ValueError("rho must be finite")
    batch, heads, tokens, key_dim = q_shape
    value_dim = key_dim if value_dim is None else int(value_dim)
    if value_dim <= 0:
        raise ValueError("value_dim must be positive")
    kv_shape = (batch, heads, key_dim, value_dim)
    output_shape = (batch, heads, tokens, value_dim)
    return LSAMaterial(
        runtime=base,
        q_shape=q_shape,
        value_dim=value_dim,
        profile=profile,
        rho=float(rho),
        kv_truncation=prob_trunc_material(base, kv_shape, profile.fraction_bits),
        qkv_truncation=prob_trunc_material(base, output_shape, profile.fraction_bits),
        scale_truncation=prob_trunc_material(
            base, output_shape, profile.coefficient_fraction_bits
        ),
    )


def secure_lsa(base, q, k, v, material: LSAMaterial) -> torch.Tensor:
    """Evaluates ``rho/N * Q(K^T V)`` on Qf secret shares."""

    material.consume(base, q, k, v)
    fraction_bits = material.profile.fraction_bits
    key_value_raw = secure_matmul(base, k.transpose(-2, -1), v)
    key_value = prob_trunc(
        base,
        key_value_raw,
        fraction_bits,
        material.kv_truncation,
    )
    output_raw = secure_matmul(base, q, key_value)
    output = prob_trunc(
        base,
        output_raw,
        fraction_bits,
        material.qkv_truncation,
    )
    public_scale = round(
        material.rho
        / material.q_shape[-2]
        * (1 << material.profile.coefficient_fraction_bits)
    )
    return prob_trunc(
        base,
        output * public_scale,
        material.profile.coefficient_fraction_bits,
        material.scale_truncation,
    )


@dataclass
class GroupedLinearMaterial:
    runtime: object
    input_shape: Tuple[int, int, int, int]
    head_types: Tuple[str, ...]
    lsa: Optional[LSAMaterial]
    used: bool = False

    def consume(self, base, q, k, v):
        if self.used:
            raise RuntimeError("grouped hAttention material already consumed")
        if base is not self.runtime:
            raise ValueError("grouped material belongs to another runtime")
        if tuple(q.shape) != self.input_shape or tuple(k.shape) != self.input_shape:
            raise ValueError(f"expected q/k shape {self.input_shape}")
        if tuple(v.shape) != self.input_shape:
            raise ValueError(f"expected v shape {self.input_shape}")
        self.used = True


def prepare_grouped_lsa_mixer(
    base,
    input_shape: Sequence[int],
    head_types: Sequence[str],
    rho: Optional[float] = None,
    profile: FixedPointProfile = FixedPointProfile(),
) -> GroupedLinearMaterial:
    """Prepares a fixed head configuration containing LSA and Mixer only."""

    input_shape = tuple(int(item) for item in input_shape)
    if len(input_shape) != 4 or any(item <= 0 for item in input_shape):
        raise ValueError("input_shape must be positive (B,H,N,D)")
    paths = normalize_head_types(head_types, input_shape[1])
    if "rsa" in paths:
        raise NotImplementedError(
            "secure RSA is not enabled until sReLU and reciprocal validation passes"
        )
    lsa_heads = paths.count("lsa")
    if rho is None:
        rho = input_shape[-1] ** -0.5
    lsa_material = None
    if lsa_heads:
        lsa_shape = (input_shape[0], lsa_heads, input_shape[2], input_shape[3])
        lsa_material = prepare_lsa(base, lsa_shape, rho=rho, profile=profile)
    return GroupedLinearMaterial(base, input_shape, paths, lsa_material)


def secure_grouped_lsa_mixer(
    base,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    material: GroupedLinearMaterial,
    patch_grid: Optional[Tuple[int, int]] = None,
    has_cls_token: bool = True,
) -> torch.Tensor:
    """Executes grouped secure LSA and local public Mixer heads."""

    material.consume(base, q, k, v)
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        _validate_share(base, tensor, name)
    output = torch.empty_like(v)
    for path in ("lsa", "mixer"):
        selected = [
            index for index, item in enumerate(material.head_types) if item == path
        ]
        if not selected:
            continue
        index = torch.tensor(selected, dtype=torch.long, device=base.device)
        selected_v = v.index_select(1, index)
        if path == "lsa":
            selected_output = secure_lsa(
                base,
                q.index_select(1, index),
                k.index_select(1, index),
                selected_v,
                material.lsa,
            )
        else:
            selected_output = mixer_attention(
                selected_v,
                patch_grid=patch_grid,
                has_cls_token=has_cls_token,
            )
        output = output.index_copy(1, index, selected_output)
    return output
