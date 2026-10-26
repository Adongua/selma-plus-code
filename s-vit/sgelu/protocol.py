"""Experimental Ruler-G2 from technical draft E.6, corrected to GeLU v2.

Q17 input/output; Q16 radicands; Q24 reciprocal roots; 96-bit products.
Branches are batched and get independent per-element, per-call randomness.
Correctness domain: |x| <= 8, 0.25 <= mu <= 4, CPU int64 additive shares.
This is NOT a validated private GeLU protocol: a^2*V openings leak information
under the manuscript's bounded integer scale distribution. See README.md.
"""

from dataclasses import dataclass

import torch

from .sgelu import validate_mu
from .wide import add96, sub96, to_limbs, limbs_from_uint64, isqrt_from_limbs

INPUT_FRAC = 17
VALUE_FRAC = 16
INVERSE_FRAC = 24
PUBLIC_RECIPROCAL_FRAC = 56
INVERSE_SHIFT = PUBLIC_RECIPROCAL_FRAC - VALUE_FRAC // 2 - INVERSE_FRAC


def joint_random_bits(base, shape, count):
    local = base.random_word(tuple(shape))
    bits = (local.unsqueeze(-1) >> torch.arange(count, device=local.device)) & 1
    parts = [base.private(bits, owner) for owner in range(3)]
    result = parts[0] + parts[1] - 2 * base.mul(parts[0], parts[1])
    result = result + parts[2] - 2 * base.mul(result, parts[2])
    return base.refresh(result)


def prob_trunc_material(base, shape, shift):
    if not 1 <= shift <= 62:
        raise ValueError("truncation shift must be in [1,62]")
    bits = joint_random_bits(base, shape, 64)
    weights = torch.ones(64, dtype=torch.int64, device=bits.device) << torch.arange(
        64, device=bits.device
    )
    return ((bits * weights).sum(-1),
            (bits[..., shift:] * weights[:64-shift]).sum(-1), bits[..., 63])


def logical_shift(x, shift):
    return (x >> shift) & ((1 << (64 - shift)) - 1)


def prob_trunc(base, value, shift, material):
    """Stochastic rounding, assuming reconstructed -2**62 <= value < 2**62.

    Shifts apply only to public opened values. Never right-shift secret shares.
    Caller supplies fresh correlated material; this low-level helper is internal.
    """
    mask, high, top = material
    opened = base.reveal(value + base.public(1 << 62) + mask)
    return _finish_prob_trunc(base, opened, shift, high, top)


def _finish_prob_trunc(base, opened, shift, high, top):
    wrap = top * (1 - logical_shift(opened, 63))
    factor = torch.tensor(1, dtype=torch.int64, device=opened.device) << (64 - shift)
    return (base.public(logical_shift(opened, shift)) - high + wrap * factor
            - base.public(1 << (62 - shift)))


def prob_trunc_batch(base, values, shifts, materials):
    """Open independent truncation masks together when tensor shapes match.

    This preserves independent one-time masks and the exact result of separate
    ``prob_trunc`` calls, but pays for one reconstruction phase instead of one
    phase per value.  It is useful for data-independent truncations that are
    already available at the same point in the online graph.
    """
    if not (len(values) == len(shifts) == len(materials)) or not values:
        raise ValueError("values, shifts and materials must have the same nonzero length")
    shape = tuple(values[0].shape)
    if any(tuple(value.shape) != shape for value in values):
        raise ValueError("batched truncation values must have the same shape")
    masked = []
    for value, shift, material in zip(values, shifts, materials):
        if not 1 <= shift <= 62:
            raise ValueError("truncation shift must be in [1,62]")
        mask, _, _ = material
        masked.append(value + base.public(1 << 62) + mask)
    opened = base.reveal(torch.stack(masked))
    return tuple(
        _finish_prob_trunc(base, item, shift, material[1], material[2])
        for item, shift, material in zip(opened, shifts, materials)
    )


def extension_material(base, wide, shape):
    """Correlated joint mask in [2**62, 2**62 + 3*2**60); no single owner knows it."""
    shape = tuple(shape)
    mask64 = base.public(torch.full(shape, 1 << 62, dtype=torch.int64,
                                    device=base.device))
    mask96 = wide.public(to_limbs(1 << 62, device=base.device).expand(shape + (4,)))
    for owner in range(3):
        local = base.random_word(shape) & ((1 << 60) - 1)
        mask64 = mask64 + base.private(local, owner)
        mask96 = add96(mask96, wide.private(limbs_from_uint64(local), owner))
    return mask64, mask96, wide.public(
        to_limbs(1 << 64, device=base.device).expand(shape + (4,))
    )


def extend_online(base, wide, value, material):
    """Lift a NONNEGATIVE secret value <2**62; same mask in both rings."""
    mask64, mask96, two64 = material
    opened = base.reveal(value - mask64)
    return sub96(add96(wide.public(limbs_from_uint64(opened)), mask96), two64)


def sample_scale(base, shape):
    # XOR of independently secret-shared 29-bit strings is uniform. This avoids
    # the draft's two secret comparisons for reducing a sum modulo 2**29.
    bits = joint_random_bits(base, shape, 29)
    weights = torch.ones(29, dtype=torch.int64, device=bits.device) << torch.arange(
        29, device=bits.device
    )
    return base.refresh((bits * weights).sum(-1) + base.public(1 << 29))


@dataclass
class G2Material:
    shape: tuple
    mu: float
    runtime: object
    scale: torch.Tensor
    squared_scale96: torch.Tensor
    extension: tuple
    truncations: dict
    used: bool = False

    def consume(self, base, value):
        if self.used:
            raise RuntimeError("G2 material already consumed")
        if base is not self.runtime or tuple(value.shape) != self.shape:
            raise ValueError("material belongs to another runtime or tensor shape")
        if value.dtype != torch.int64 or value.device != base.device:
            raise ValueError(f"expected int64 Q17 share on {base.device}")
        self.used = True  # burn even if the subsequent online computation fails


def prepare_g2(base, wide, shape, mu):
    if wide.base is not base:
        raise ValueError("64-bit and 96-bit runtimes must share the same base")
    mu = validate_mu(mu)
    if not 0.25 <= mu <= 4:
        raise ValueError("this fixed-point profile requires 0.25 <= mu <= 4")
    shape = tuple(shape)
    if any(not isinstance(n, int) or n <= 0 for n in shape):
        raise ValueError("shape dimensions must be positive integers")
    branches = (2,) + shape
    # Independent scales in BOTH branches; unlike draft E.6, no broadcast reuse.
    scale = sample_scale(base, branches)
    scale2 = base.square(scale)
    scale96 = extend_online(base, wide, scale2, extension_material(base, wide, branches))
    extension = extension_material(base, wide, branches)
    specs = {"square": (shape, 18), "inverse": (branches, INVERSE_SHIFT),
             "bracket": (shape, 24), "product": (shape, 24), "half": (shape, 1)}
    truncations = {name: prob_trunc_material(base, size, shift)
                   for name, (size, shift) in specs.items()}
    return G2Material(shape, mu, base, scale, scale96, extension, truncations)


def ruler_g2(base, wide, x_q17, material, profiler=None):
    """Consume one prepared material and return a Q17 share (no output opening)."""
    if wide.base is not base:
        raise ValueError("64-bit and 96-bit runtimes must share the same base")
    material.consume(base, x_q17)
    trunc = material.truncations
    if profiler is not None:
        profiler.start("square_and_joint_trunc")
    square_q34 = base.square(x_q17)
    # Both values are ready after the square.  Opening their independent masks
    # as one tensor removes one reconstruction from the critical path.
    square_q16, half_q17 = prob_trunc_batch(
        base, (square_q34, x_q17), (18, 1),
        (trunc["square"], trunc["half"]),
    )
    if profiler is not None:
        profiler.stop("square_and_joint_trunc")
    constants = torch.tensor([round(material.mu**2 * (1 << 16)),
                              round(4 * material.mu**2 * (1 << 16))],
                             dtype=torch.int64, device=x_q17.device)
    constants = constants.reshape((2,) + (1,) * len(material.shape))
    values = square_q16.unsqueeze(0) + base.public(constants)
    if profiler is not None:
        profiler.start("scale_extension")
    values96 = extend_online(base, wide, values, material.extension)
    if profiler is not None:
        profiler.stop("scale_extension")
        profiler.start("wide_mul_reveal_isqrt")
    opened = wide.reveal(wide.mul(material.squared_scale96, values96))
    roots = isqrt_from_limbs(opened)
    if profiler is not None:
        profiler.stop("wide_mul_reveal_isqrt")
    if torch.any(roots <= 0):
        raise ArithmeticError("nonpositive opened root: fixed-point input contract violated")
    # Exact integer nearest rounding (ties upward) avoids float64 reciprocal drift.
    public_inverse = torch.div((1 << 56) + roots // 2, roots, rounding_mode="floor")
    if profiler is not None:
        profiler.start("inverse_trunc")
    inverse_q24 = prob_trunc(base, material.scale * public_inverse, INVERSE_SHIFT, trunc["inverse"])
    if profiler is not None:
        profiler.stop("inverse_trunc")
    c43, c13 = round((4 / 3) * (1 << 24)), round((1 / 3) * (1 << 24))
    if profiler is not None:
        profiler.start("bracket_trunc")
    bracket = prob_trunc(base, c43 * inverse_q24[0] - c13 * inverse_q24[1], 24, trunc["bracket"])
    if profiler is not None:
        profiler.stop("bracket_trunc")
        profiler.start("product_and_trunc")
    term2_q16 = prob_trunc(base, base.mul(square_q16, bracket), 24, trunc["product"])
    if profiler is not None:
        profiler.stop("product_and_trunc")
    return half_q17 + term2_q16
