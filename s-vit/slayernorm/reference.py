"""Float64 reference for the scaled LayerNorm identity in the manuscript."""

import math
import torch


def scaled_layernorm(x, gamma, beta, epsilon=1e-5, scale=1.0):
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("finite positive epsilon and scale required")
    centered = (x - x.mean(-1, keepdim=True)) / scale
    variance = centered.square().mean(-1, keepdim=True) + epsilon / scale**2
    return gamma * centered * torch.rsqrt(variance) + beta
