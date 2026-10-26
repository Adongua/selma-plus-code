"""CryptGPU-native comparison operators for the sGeLU microbenchmark.

These functions deliberately use CryptGPU ``MPCTensor`` operations rather
than the custom Ring3 runtime.  They are engineering ports of published
approximations, not claims that the original projects used CryptGPU.
"""

import math
from dataclasses import dataclass

import torch

PUMA_T0, PUMA_T1, PUMA_T2 = -4.0, -1.95, 3.0
PUMA_F0 = (-0.011034134030615728, -0.11807612951181953,
           -0.42226581151983866, -0.5054031199708174)
PUMA_F1 = (0.0018067462606141187, -0.037688200365904236,
           0.3603292692789629, 0.5, 0.008526321541038084)
SECFORMER_BETA = (1.25772, -0.0299154, 0.382155, -0.0519123,
                  0.196033, -0.0624557, 0.118029)


def _chebyshev_coefficients(degree=16, bound=8.0):
    """Return interpolation coefficients for GeLU on ``[-bound, bound]``."""
    nodes = degree + 1
    values = []
    angles = []
    for index in range(nodes):
        angle = math.pi * (index + 0.5) / nodes
        x = bound * math.cos(angle)
        gelu = 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))
        angles.append(angle)
        values.append(gelu)
    return tuple(
        (2.0 / nodes) * sum(
            value * math.cos(order * angle)
            for value, angle in zip(values, angles)
        )
        for order in range(degree + 1)
    )


CHEBYSHEV16_BOUND = 8.0
CHEBYSHEV16_COEFFICIENTS = _chebyshev_coefficients(
    degree=16, bound=CHEBYSHEV16_BOUND
)


def _require_arithmetic(value):
    from crypten.mpc import MPCTensor, ptype
    if not isinstance(value, MPCTensor) or value.ptype != ptype.arithmetic:
        raise ValueError("expected a CryptGPU arithmetic MPCTensor")


def cryptgpu_quad_gelu(value):
    """MPCFormer quadratic approximation on CryptGPU."""
    _require_arithmetic(value)
    return value.square().mul(0.125).add(value.mul(0.25)).add(0.5)


def cryptgpu_erf_gelu(value, iterations=8):
    """CrypTen-style Taylor-erf GeLU evaluated by CryptGPU primitives."""
    _require_arithmetic(value)
    if iterations < 1:
        raise ValueError("iterations must be positive")
    z = value.mul(1.0 / math.sqrt(2.0))
    power, erf_value = z, z
    for n in range(1, iterations + 1):
        power = power.mul(z).mul(z)
        coefficient = ((-1) ** n) / (math.factorial(n) * (2 * n + 1))
        erf_value = erf_value.add(power.mul(coefficient))
    erf_value = erf_value.mul(2.0 / math.sqrt(math.pi))
    return value.mul(erf_value.add(1.0)).mul(0.5)


def cryptgpu_chebyshev16_gelu(value):
    """Degree-16 Chebyshev GeLU interpolant evaluated with Clenshaw recurrence."""
    _require_arithmetic(value)
    z = value.mul(1.0 / CHEBYSHEV16_BOUND)
    zero = value.mul(0)
    b_kplus1, b_kplus2 = zero, zero.clone()
    for coefficient in reversed(CHEBYSHEV16_COEFFICIENTS[1:]):
        b_k = z.mul(b_kplus1).mul(2).sub(b_kplus2).add(coefficient)
        b_kplus2, b_kplus1 = b_kplus1, b_k
    return z.mul(b_kplus1).sub(b_kplus2).add(
        0.5 * CHEBYSHEV16_COEFFICIENTS[0]
    )


def chebyshev16_gelu_reference(value):
    """Plaintext reference using the same coefficients and recurrence."""
    z = value / CHEBYSHEV16_BOUND
    b_kplus1 = torch.zeros_like(value)
    b_kplus2 = torch.zeros_like(value)
    for coefficient in reversed(CHEBYSHEV16_COEFFICIENTS[1:]):
        b_k = 2.0 * z * b_kplus1 - b_kplus2 + coefficient
        b_kplus2, b_kplus1 = b_kplus1, b_k
    return z * b_kplus1 - b_kplus2 + 0.5 * CHEBYSHEV16_COEFFICIENTS[0]


def cryptgpu_puma_gelu(value):
    """PUMA piecewise polynomial using CryptGPU secure comparisons."""
    _require_arithmetic(value)
    x2 = value.square()
    x3 = x2.mul(value)
    x4 = x2.square()
    x6 = x4.mul(x2)
    f0 = (x3.mul(PUMA_F0[0]).add(x2.mul(PUMA_F0[1]))
          .add(value.mul(PUMA_F0[2])).add(PUMA_F0[3]))
    f1 = (x6.mul(PUMA_F1[0]).add(x4.mul(PUMA_F1[1]))
          .add(x2.mul(PUMA_F1[2])).add(value.mul(PUMA_F1[3]))
          .add(PUMA_F1[4]))
    below0 = value.lt(PUMA_T0)
    below1 = value.lt(PUMA_T1)
    below2 = value.lt(PUMA_T2)
    return ((below1 - below0).mul(f0)
            .add((below2 - below1).mul(f1))
            .add((1 - below2).mul(value)))


@dataclass
class FourierMaterial:
    mask: object
    mask_sin: object
    mask_cos: object


def prepare_fourier(shape, device, src=0):
    """Secret-share SecFormer's mask and seven sine/cosine mask tables."""
    import crypten
    import crypten.communicator as comm

    rank = comm.get().get_rank()
    if rank == src:
        mask = 2.0 * math.pi * torch.rand(shape, device=device) - math.pi
    else:
        mask = torch.zeros(shape, dtype=torch.float32, device=device)
    harmonics = torch.arange(1, 8, dtype=torch.float64, device=device)
    angles = mask.double().unsqueeze(-1) * (harmonics * math.pi / 10.0)
    mask_sin = torch.sin(angles)
    mask_cos = torch.cos(angles)
    return FourierMaterial(
        crypten.cryptensor(mask, src=src, precision=17, device=device),
        crypten.cryptensor(mask_sin, src=src, precision=17, device=device),
        crypten.cryptensor(mask_cos, src=src, precision=17, device=device),
    )


def cryptgpu_fourier_gelu(value, material, threshold=2.5):
    """SecFormer-style masked Fourier GeLU on CryptGPU.

    ``delta=z-mask`` is intentionally opened, matching the masked-Fourier
    construction.  Interval predicates stay encrypted through CryptGPU LT.
    """
    _require_arithmetic(value)
    z = value.mul(1.0 / math.sqrt(2.0))
    delta = (z - material.mask).get_plain_text()
    harmonics = torch.arange(1, 8, dtype=torch.float64, device=delta.device)
    angles = delta.double().unsqueeze(-1) * (harmonics * math.pi / 10.0)
    p, q = torch.sin(angles), torch.cos(angles)
    beta = torch.tensor(SECFORMER_BETA, dtype=torch.float64, device=delta.device)
    fourier = ((material.mask_cos.mul(p) + material.mask_sin.mul(q))
               .mul(beta).sum(-1))
    lower, upper = z.lt(-threshold), z.lt(threshold)
    erf_approx = (-lower).add((upper - lower).mul(fourier)).add(1 - upper)
    return value.mul(erf_approx.add(1.0)).mul(0.5)
