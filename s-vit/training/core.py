"""Public subspace, spherical directions and the common ZO-Adam definition."""
from dataclasses import dataclass, asdict
import math
import torch


@dataclass(frozen=True)
class ZOConfig:
    rank: int = 4
    sigma: float = 0.1
    lr: float = 0.03
    beta1: float = 0.9
    beta2: float = 0.99
    epsilon: float = 0.01
    seed: int = 20261107
    c_bound: float = 8.0
    z_bound: float = 8.0
    sqrt_iters: int = 20
    reciprocal_iters: int = 24

    def validate(self):
        if self.rank < 1 or self.sqrt_iters < 1 or self.reciprocal_iters < 1:
            raise ValueError("positive rank and fixed iteration counts required")
        for name in ("sigma", "lr", "epsilon", "c_bound", "z_bound"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"invalid {name}")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Adam beta must be in [0,1)")
        # Q24 fixed point: bound all optimizer products below signed ring overflow.
        if self.c_bound > 16 or self.z_bound > 16 or self.epsilon < 0.005:
            raise ValueError("initial numerical profile requires c,z bounds <=16 and epsilon>=0.005")
        return self


def basis(prompt_size, config):
    if config.rank > prompt_size:
        raise ValueError("subspace rank cannot exceed prompt size")
    g = torch.Generator().manual_seed(config.seed)
    q, r = torch.linalg.qr(torch.randn(prompt_size, config.rank, generator=g, dtype=torch.float64), mode="reduced")
    # Fix QR column signs for reproducible public basis.
    return q * torch.where(r.diag() < 0, -1., 1.)


def direction(step, config):
    g = torch.Generator().manual_seed(config.seed + 1000003 * step)
    u = torch.randn(config.rank, generator=g, dtype=torch.float64)
    return u / u.norm()


def weights(values):
    if len(values) != 3 or any(not math.isfinite(v) or v < 0 for v in values) or sum(values) <= 0:
        raise ValueError("three nonnegative party weights with positive sum required")
    return [v / sum(values) for v in values]


class PlainAdam:
    def __init__(self, config):
        self.config = config.validate()
        self.z = torch.zeros(config.rank, dtype=torch.float64)
        self.m = torch.zeros_like(self.z)
        self.v = torch.zeros_like(self.z)
        self.step = 0

    def update(self, delta, u):
        p = self.config
        c = p.rank * float(delta) / (2 * p.sigma)
        if not math.isfinite(c) or abs(c) > p.c_bound:
            raise ArithmeticError("direction coefficient outside numerical contract")
        g = c * u.cpu().double()
        self.step += 1
        self.m = p.beta1 * self.m + (1 - p.beta1) * g
        self.v = p.beta2 * self.v + (1 - p.beta2) * (c * c) * u.cpu().double().square()
        mh = self.m / (1 - p.beta1 ** self.step)
        vh = self.v / (1 - p.beta2 ** self.step)
        self.z = self.z - p.lr * mh / (vh.sqrt() + p.epsilon)
        if not torch.isfinite(self.z).all() or self.z.abs().max() > p.z_bound:
            raise ArithmeticError("prompt latent outside numerical contract")
        return self.z

    def state(self):
        return {"z": self.z, "m": self.m, "v": self.v, "step": self.step, "config": asdict(self.config)}
