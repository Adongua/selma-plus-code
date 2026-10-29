"""Fixed-structure plaintext hAttention module."""

import torch
from torch import nn

from .config import HAttentionConfig
from .reference import grouped_hattention


class FixedHAttention(nn.Module):
    """ViT attention module that executes one public path per head.

    This phase-one reference keeps a packed QKV projection for checkpoint
    compatibility.  Physical Q/K pruning for Mixer heads belongs to the later
    deployment export stage.
    """

    def __init__(
        self,
        dim: int,
        config: HAttentionConfig,
        qkv_bias: bool = False,
        projection_bias: bool = True,
    ):
        super().__init__()
        if dim <= 0 or dim % config.num_heads != 0:
            raise ValueError("dim must be positive and divisible by num_heads")
        self.dim = dim
        self.config = config
        self.head_dim = dim // config.num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        # Keep the original MPCViT parameter names (qkv/proj) so checkpoints
        # can be migrated without renaming their learned projection tensors.
        self.proj = nn.Linear(dim, dim, bias=projection_bias)
        self.proj_drop = nn.Identity()

    def forward_preprojected(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        return grouped_hattention(
            q,
            k,
            v,
            self.config.head_types,
            rho=self.config.resolved_rho(self.head_dim),
            epsilon=self.config.epsilon,
            srelu_radius=self.config.srelu_radius,
            patch_grid=self.config.patch_grid,
            has_cls_token=self.config.has_cls_token,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or value.shape[-1] != self.dim:
            raise ValueError(f"input must have (B,N,{self.dim}) layout")
        batch, tokens, _ = value.shape
        qkv = self.qkv(value).reshape(
            batch,
            tokens,
            3,
            self.config.num_heads,
            self.head_dim,
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        output = self.forward_preprojected(q, k, v)
        output = output.transpose(1, 2).reshape(batch, tokens, self.dim)
        return self.proj_drop(self.proj(output))

    def extra_repr(self) -> str:
        counts = {
            path: self.config.head_types.count(path)
            for path in ("rsa", "lsa", "mixer")
        }
        return (
            f"dim={self.dim}, num_heads={self.config.num_heads}, "
            f"paths={counts}, rho={self.config.resolved_rho(self.head_dim):.8g}"
        )
