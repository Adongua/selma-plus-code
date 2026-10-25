"""Floating-point references and public, per-layer scale calibration."""

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

MU0 = (7.0 / 6.0) * math.sqrt(math.pi / 2.0)


def validate_mu(mu):
    mu = float(mu)
    if not math.isfinite(mu) or mu <= 0:
        raise ValueError("mu must be a finite positive public scalar")
    return mu


def g1(x, mu=MU0):
    mu = validate_mu(mu)
    square = x.square()
    return 0.5 * x + 0.5 * square * torch.rsqrt(square + mu * mu)


def g2(x, mu=MU0):
    mu = validate_mu(mu)
    square = x.square()
    bracket = (4.0 * torch.rsqrt(square + mu * mu)
               - torch.rsqrt(square + 4.0 * mu * mu)) / 3.0
    return 0.5 * x + 0.5 * square * bracket


class SmoothGeLU(nn.Module):
    """Drop-in plaintext activation; calibrated mu is a buffer, not a parameter.

    MPCViT uses F.gelu via block.activation. Assign an instance to that attribute
    on a loaded model; checkpoint loading and model import are separate steps.
    """

    def __init__(self, mu=MU0):
        super().__init__()
        self.register_buffer("mu", torch.tensor(validate_mu(mu), dtype=torch.float64))

    def forward(self, x):
        # Tensor-only expression retains input autograd and avoids CLI imports.
        mu = self.mu.to(dtype=x.dtype, device=x.device)
        square = x.square()
        return x * 0.5 + square * (
            4 * torch.rsqrt(square + mu.square())
            - torch.rsqrt(square + 4 * mu.square())
        ) / 6


def error_metrics(actual, expected):
    difference = actual.double() - expected.double()
    return {"max_abs": difference.abs().max().item(),
            "mae": difference.abs().mean().item(),
            "mse": difference.square().mean().item()}


@dataclass(frozen=True)
class CalibrationResult:
    mu: float
    initial_mu: float
    initial_mse: float
    fitted_mse: float
    sample_count: int
    bounds: tuple
    at_boundary: bool

    def to_dict(self):
        return asdict(self)


@torch.no_grad()
def calibrate_mu(public_activations, bounds=(0.25, 4.0), grid_points=65,
                 iterations=48):
    """Bounded log-scale grid + local golden-section refinement of empirical MSE.

    Input MUST be public calibration data. No runtime private activations should
    be opened to call this function. This is a numerical fit, not a global proof.
    """
    low, high = map(validate_mu, bounds)
    if low >= high or grid_points < 3 or iterations < 1 or not low <= MU0 <= high:
        raise ValueError("bounds must contain MU0; require low < high and >=3 grid points")
    x = torch.as_tensor(public_activations).detach().to(device="cpu", dtype=torch.float64).flatten()
    if not x.numel() or not torch.isfinite(x).all():
        raise ValueError("calibration requires nonempty finite public activations")
    target = F.gelu(x, approximate="none")

    def objective(log_mu):
        return (g2(x, math.exp(log_mu)) - target).square().mean().item()

    grid = torch.linspace(math.log(low), math.log(high), grid_points).tolist()
    scores = [objective(point) for point in grid]
    best = min(range(grid_points), key=scores.__getitem__)
    left, right = grid[max(0, best - 1)], grid[min(grid_points - 1, best + 1)]
    ratio = (math.sqrt(5) - 1) / 2
    c, d = right - ratio * (right - left), left + ratio * (right - left)
    fc, fd = objective(c), objective(d)
    for _ in range(iterations):
        if fc < fd:
            right, d, fd = d, c, fc
            c = right - ratio * (right - left)
            fc = objective(c)
        else:
            left, c, fc = c, d, fd
            d = left + ratio * (right - left)
            fd = objective(d)
    initial = objective(math.log(MU0))
    candidates = [(math.log(MU0), initial), (grid[best], scores[best]), (c, fc), (d, fd)]
    log_mu, mse = min(candidates, key=lambda pair: pair[1])
    mu = math.exp(log_mu)
    return CalibrationResult(mu, MU0, initial, mse, x.numel(), (low, high),
                             math.isclose(mu, low, rel_tol=1e-6)
                             or math.isclose(mu, high, rel_tol=1e-6))
