"""Public configuration and validation for hAttention."""

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Tuple


HEAD_TYPES = ("rsa", "lsa", "mixer")


def normalize_head_types(head_types: Iterable[str], num_heads: Optional[int] = None) -> Tuple[str, ...]:
    """Returns canonical lower-case head types and rejects unknown paths."""

    normalized = tuple(str(item).strip().lower() for item in head_types)
    if num_heads is not None and len(normalized) != num_heads:
        raise ValueError(
            "head type count does not match num_heads: "
            f"{len(normalized)} != {num_heads}"
        )
    unknown = sorted(set(normalized).difference(HEAD_TYPES))
    if unknown:
        raise ValueError(f"unknown hAttention head types: {unknown}")
    return normalized


def infer_patch_grid(token_count: int, has_cls_token: bool = True) -> Tuple[int, int]:
    """Infers a square patch grid while excluding a leading CLS token."""

    patch_count = token_count - int(has_cls_token)
    if patch_count <= 0:
        raise ValueError("token_count must contain at least one patch token")
    side = math.isqrt(patch_count)
    if side * side != patch_count:
        raise ValueError(
            f"cannot infer a square patch grid from {patch_count} patch tokens; "
            "pass patch_grid explicitly"
        )
    return side, side


@dataclass(frozen=True)
class HAttentionConfig:
    """Public, deployment-time configuration for one attention layer."""

    num_heads: int
    head_types: Tuple[str, ...]
    rho: Optional[float] = None
    epsilon: float = 1e-5
    srelu_radius: float = 0.0055
    has_cls_token: bool = True
    patch_grid: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        object.__setattr__(
            self,
            "head_types",
            normalize_head_types(self.head_types, self.num_heads),
        )
        if self.rho is not None and not math.isfinite(self.rho):
            raise ValueError("rho must be finite")
        if self.epsilon <= 0 or not math.isfinite(self.epsilon):
            raise ValueError("epsilon must be positive and finite")
        if self.srelu_radius <= 0 or not math.isfinite(self.srelu_radius):
            raise ValueError("srelu_radius must be positive and finite")
        if self.patch_grid is not None:
            rows, columns = self.patch_grid
            if rows <= 0 or columns <= 0:
                raise ValueError("patch_grid dimensions must be positive")

    def resolved_rho(self, head_dim: int) -> float:
        """Defaults to MPCViT's 1/sqrt(d_h) score scaling."""

        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.rho is not None:
            return self.rho
        return head_dim ** -0.5
