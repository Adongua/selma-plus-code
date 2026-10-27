"""Plaintext reference functions for the three hAttention paths.

All tensors use ``(batch, heads, tokens, features)`` layout.  Head path
choices are public deployment metadata; no function branches on secret data.
"""

import math
from typing import Dict, Optional, Sequence, Tuple, Union

import torch

from .config import infer_patch_grid, normalize_head_types


TensorOrResult = Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must have (B,H,N,D) layout")
    if q.shape[:2] != k.shape[:2] or q.shape[:2] != v.shape[:2]:
        raise ValueError("q, k and v must share batch and head dimensions")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must have the same token count")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k feature dimensions must match")


def positive_srelu(value: torch.Tensor, radius: float = 0.0055) -> torch.Tensor:
    """Non-negative smooth ReLU used by the RSA path.

    This is ``(x + sqrt(x^2 + r^2)) / 2``.  It is strictly non-negative,
    which preserves RSA's non-negative attention-weight invariant.
    """

    if radius <= 0:
        raise ValueError("radius must be positive")
    return 0.5 * (value + torch.sqrt(value.square() + radius * radius))


def legacy_zero_origin_srelu(value: torch.Tensor, radius: float = 0.01) -> torch.Tensor:
    """Legacy appendix profile, retained only for controlled reproduction.

    Subtracting ``r/2`` makes the function zero at the origin, but introduces a
    small negative tail.  It must not be called a non-negative RSA activation.
    """

    return positive_srelu(value, radius=radius) - 0.5 * radius


def rsa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    epsilon: float = 1e-5,
    radius: float = 0.0055,
    use_exact_relu: bool = False,
    score_scale: Optional[float] = None,
    return_aux: bool = False,
) -> TensorOrResult:
    """ReLU/sReLU attention with row normalization."""

    _validate_qkv(q, k, v)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if score_scale is None:
        score_scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * score_scale
    if use_exact_relu:
        positive = torch.relu(scores)
        shifted = positive
    else:
        # Keep the manuscript's two-stage convention explicit: first use the
        # zero-at-origin shifted sReLU, then add the public r/2 before row
        # normalization.  The resulting weight is non-negative and is
        # algebraically identical to positive_srelu(scores, radius).
        shifted = legacy_zero_origin_srelu(scores, radius)
        positive = shifted + 0.5 * radius
    denominator = positive.sum(dim=-1, keepdim=True) + epsilon
    weights = positive / denominator
    output = torch.matmul(weights, v)
    if return_aux:
        return output, {
            "scores": scores,
            "shifted_srelu": shifted,
            "positive": positive,
            "denominator": denominator,
            "weights": weights,
        }
    return output


def lsa_attention_direct(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rho: float,
) -> torch.Tensor:
    """Literal ``rho/N * (Q K^T) V`` reference used for equivalence tests."""

    _validate_qkv(q, k, v)
    token_count = k.shape[-2]
    return torch.matmul(torch.matmul(q, k.transpose(-2, -1)), v) * (rho / token_count)


def lsa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rho: float,
) -> torch.Tensor:
    """Reassociated LSA: ``rho/N * Q (K^T V)``.

    The function never materializes the secret ``N x N`` score matrix.
    """

    _validate_qkv(q, k, v)
    token_count = k.shape[-2]
    key_value = torch.matmul(k.transpose(-2, -1), v)
    return torch.matmul(q, key_value) * (rho / token_count)


def mixer_attention(
    v: torch.Tensor,
    patch_grid: Optional[Tuple[int, int]] = None,
    has_cls_token: bool = True,
) -> torch.Tensor:
    """Applies the public reshape-transpose-reshape token permutation.

    For ViT inputs, the leading CLS token is preserved and only patch tokens
    are permuted.  This supports N=197, 257 and 577 via 14x14, 16x16 and 24x24
    patch grids respectively.
    """

    if v.ndim != 4:
        raise ValueError("v must have (B,H,N,D) layout")
    batch, heads, token_count, features = v.shape
    if patch_grid is None:
        patch_grid = infer_patch_grid(token_count, has_cls_token=has_cls_token)
    rows, columns = patch_grid
    prefix = 1 if has_cls_token else 0
    patch_count = token_count - prefix
    if rows * columns != patch_count:
        raise ValueError(
            f"patch_grid {patch_grid} contains {rows * columns} positions, "
            f"but the tensor has {patch_count} patch tokens"
        )
    patches = v[:, :, prefix:, :].reshape(batch, heads, rows, columns, features)
    mixed = patches.transpose(2, 3).contiguous().reshape(
        batch, heads, patch_count, features
    )
    if not has_cls_token:
        return mixed
    return torch.cat((v[:, :, :1, :], mixed), dim=2)


def grouped_hattention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    head_types: Sequence[str],
    rho: Optional[float] = None,
    epsilon: float = 1e-5,
    srelu_radius: float = 0.0055,
    patch_grid: Optional[Tuple[int, int]] = None,
    has_cls_token: bool = True,
    rsa_exact_relu: bool = False,
) -> torch.Tensor:
    """Executes heads in path groups and restores their original order."""

    _validate_qkv(q, k, v)
    paths = normalize_head_types(head_types, q.shape[1])
    if q.shape[-2] != v.shape[-2]:
        raise ValueError("grouped self-attention requires matching q and v token counts")
    if rho is None:
        rho = q.shape[-1] ** -0.5

    output = torch.empty_like(v)
    for path in ("rsa", "lsa", "mixer"):
        selected = [index for index, item in enumerate(paths) if item == path]
        if not selected:
            continue
        index = torch.tensor(selected, dtype=torch.long, device=v.device)
        group_v = v.index_select(1, index)
        if path == "rsa":
            group_output = rsa_attention(
                q.index_select(1, index),
                k.index_select(1, index),
                group_v,
                epsilon=epsilon,
                radius=srelu_radius,
                use_exact_relu=rsa_exact_relu,
            )
        elif path == "lsa":
            group_output = lsa_attention(
                q.index_select(1, index),
                k.index_select(1, index),
                group_v,
                rho=rho,
            )
        else:
            group_output = mixer_attention(
                group_v,
                patch_grid=patch_grid,
                has_cls_token=has_cls_token,
            )
        output = output.index_copy(1, index, group_output)
    return output
