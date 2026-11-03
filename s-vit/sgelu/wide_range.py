"""Experimental |x|<=12 G2 profile using the unchanged G2 protocol.

The caller must enforce the range privately or by its public input contract;
checking public calibration samples is not a proof about future secret inputs.
"""
from dataclasses import dataclass
import math

from .protocol import prepare_g2, ruler_g2


@dataclass(frozen=True)
class WideGeLUProfile:
    mu: float = 1.4621998268680836
    input_bound: float = 12.0

    def __post_init__(self):
        if (not math.isfinite(self.mu) or not 0.25 <= self.mu <= 4 or
                self.input_bound != 12):
            raise ValueError("unsupported wide GeLU profile")
        # scale^2 < 2^60 and the opened radicand must remain below 2^84.
        if self.input_bound**2 + 4 * self.mu**2 + 2**-15 >= 256:
            raise ValueError("masked radicand can exceed the public isqrt domain")


def prepare_wide_gelu(base, wide, shape, profile=WideGeLUProfile()):
    return prepare_g2(base, wide, shape, profile.mu)


def wide_gelu(base, wide, x, material):
    return ruler_g2(base, wide, x, material)
