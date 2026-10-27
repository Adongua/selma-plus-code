"""Recent secure-ReLU baselines for the unified three-party benchmark.

``crypten_native`` is deliberately the unmodified CrypTen
``MPCTensor.relu()`` call. It must remain the strict framework baseline even
if a particular CrypTen/CUDA combination fails the correctness gate.

``crypten_fullwidth_adapter`` is kept as a separate engineering diagnostic:
it uses CrypTen's binary sharing, B2A conversion and Beaver multiplication,
but supplies an explicit 64-bit A2B / sign circuit. It must never be
reported under the name ``crypten_native``.
The HummingBird paths use the same circuit in a public-bound reduced ring, so
their circuit depth and payload depend on the retained bit width.

``bicoptor2_keybits`` deliberately implements only Bicoptor 2.0's public
key-bit precision policy on the same reduced-ring CrypTen backend.  It is not
the paper's UBL/RSS two-round assistant protocol and is labelled as an adapted
engineering reference in every report.
"""

import math

import torch


METHODS = (
    "crypten_native",
    "crypten_fullwidth_adapter",
    "hummingbird_eco",
    "hummingbird_budget",
    "bicoptor2_keybits",
)

METHOD_YEAR = {
    "crypten_native": 2021,
    "crypten_fullwidth_adapter": 2021,
    "bicoptor2_keybits": 2023,
    "hummingbird_eco": 2024,
    "hummingbird_budget": 2024,
}

METHOD_PROVENANCE = {
    "crypten_native": (
        "Unmodified vendored CrypTen MPCTensor.relu() call on the CrypTen "
        "NCCL/CUDA backend; timing is reportable only when the exact-ReLU "
        "correctness gate passes"
    ),
    "crypten_fullwidth_adapter": (
        "Engineering diagnostic: explicit full-width 64-bit A2B, sign "
        "extraction, B2A and Beaver secret selection built from CrypTen "
        "primitives; not the upstream MPCTensor.relu() implementation"
    ),
    "hummingbird_eco": (
        "HummingBird-eco adapted to three-party CrypTen: reduced-ring DReLU "
        "retains all Q17 fractional bits and only removes public-bound-proven "
        "redundant high bits"
    ),
    "hummingbird_budget": (
        "HummingBird-style accuracy/performance ablation adapted to three-party "
        "CrypTen: DReLU is requantized to a public fractional-bit budget"
    ),
    "bicoptor2_keybits": (
        "Bicoptor 2.0 key-bit precision policy evaluated with the reduced-ring "
        "CrypTen 3PC DReLU backend; not the official two-round UBL/RSS protocol"
    ),
}

METHOD_IMPLEMENTATION_CLASS = {
    "crypten_native": "upstream_native",
    "crypten_fullwidth_adapter": "crypten_fullwidth_adapter",
    "bicoptor2_keybits": "paper_idea_adapted_3pc",
    "hummingbird_eco": "paper_idea_adapted_3pc",
    "hummingbird_budget": "engineering_ablation",
}

METHOD_MAIN_TABLE_SCOPE = {
    "crypten_native": (
        "strict CrypTen baseline; retain the raw result and exclude it from "
        "performance claims if the exact-ReLU correctness gate fails"
    ),
    "crypten_fullwidth_adapter": (
        "diagnostic only; do not present it as the official CrypTen ReLU "
        "implementation or use it as the strict native baseline"
    ),
    "bicoptor2_keybits": (
        "adapted comparison only; must not be labelled as the official "
        "Bicoptor 2.0 two-round implementation"
    ),
    "hummingbird_eco": (
        "adapted comparison only; must not reuse timing numbers from the "
        "paper's two-party experiments"
    ),
    "hummingbird_budget": "ablation only; excluded from the main comparison",
}


def effective_signed_bits(input_bound: float, fraction_bits: int) -> int:
    """Smallest signed ring width covering ``|x| < input_bound``."""

    if input_bound <= 0 or not math.isfinite(input_bound):
        raise ValueError("input_bound must be positive and finite")
    if not isinstance(fraction_bits, int) or not 0 <= fraction_bits <= 30:
        raise ValueError("fraction_bits must be an integer in [0,30]")
    scaled_bound = input_bound * (1 << fraction_bits)
    magnitude_bits = max(1, math.ceil(math.log2(scaled_bound)))
    bits = magnitude_bits + 1
    if not 2 <= bits <= 62:
        raise ValueError("effective signed width must lie in [2,62]")
    return bits


def _binary_add_mod(left, right, bits: int):
    """Add XOR shares modulo ``2**bits`` with a packed prefix circuit.

    Each prefix stage batches its generate and propagate AND gates, so the
    online depth is ``ceil(log2(bits))`` instead of CrypTen's fixed six stages
    for a 64-bit word.
    """

    from crypten.mpc.primitives.binary import BinarySharedTensor

    if bits not in (64,) and not 2 <= bits <= 62:
        raise ValueError("bits must lie in [2,62] or equal 64")
    # For a full 64-bit signed CUDA tensor, ``2**64 - 1`` cannot be passed to
    # an int64 bitwise kernel. The tensor already stores the ring element in
    # the complete 64-bit bit pattern, so omit the redundant full-width mask.
    full_width = bits == 64
    mask = None if full_width else (1 << bits) - 1
    propagate = left ^ right if full_width else (left ^ right) & mask
    original_propagate = propagate.clone()
    generate = left & right if full_width else (left & right) & mask
    distance = 1
    while distance < bits:
        low_mask = (1 << distance) - 1
        active_propagate = (
            propagate & ~low_mask if full_width else propagate & (mask ^ low_mask)
        )
        shifted = BinarySharedTensor.stack(
            [
                (generate << distance)
                if full_width
                else (generate << distance) & mask,
                (propagate << distance)
                if full_width
                else (propagate << distance) & mask,
            ]
        )
        fanout = BinarySharedTensor.stack(
            [active_propagate, active_propagate]
        ) & shifted
        generate ^= fanout[0]
        propagate = (propagate & low_mask) ^ fanout[1]
        distance <<= 1
    if full_width:
        return original_propagate ^ (generate << 1)
    return (original_propagate ^ ((generate << 1) & mask)) & mask


def _reduced_ring_drelu(value, bits: int):
    """Return an arithmetic sharing of ``value >= 0`` in a reduced ring."""

    import crypten.communicator as comm
    from crypten.mpc import MPCTensor, ptype
    from crypten.mpc.primitives.binary import BinarySharedTensor
    from crypten.mpc.primitives.converters import convert

    if value.ptype != ptype.arithmetic:
        raise ValueError("reduced-ring DReLU requires arithmetic shares")
    if comm.get().get_world_size() != 3:
        raise ValueError("this benchmark adaptation requires exactly three parties")
    mask = None if bits == 64 else (1 << bits) - 1
    local_share = value.share if mask is None else value.share & mask
    binary_parts = [
        BinarySharedTensor(local_share, src=owner, device=value.device)
        for owner in range(3)
    ]
    binary_value = _binary_add_mod(binary_parts[0], binary_parts[1], bits)
    binary_value = _binary_add_mod(binary_value, binary_parts[2], bits)
    negative = (binary_value >> (bits - 1)) & 1
    nonnegative = negative ^ 1
    arithmetic = convert(nonnegative, ptype.arithmetic, precision=0, bits=1)
    arithmetic.encoder._scale = 1
    result = MPCTensor([])
    result._tensor = arithmetic
    result.ptype = ptype.arithmetic
    return result


def _requantize(value, fraction_bits: int):
    from crypten.encoder import FixedPointEncoder

    if fraction_bits > value.encoder._precision_bits:
        raise ValueError("comparison precision cannot exceed input precision")
    result = value.clone()
    result._tensor.encode_(FixedPointEncoder(precision_bits=fraction_bits))
    return result


def crypten_native_relu(value, **_):
    """Call CrypTen's own ``MPCTensor.relu()`` without modification."""

    return value.relu()


def crypten_fullwidth_adapter_relu(value, **_):
    """Correctness-repaired full-width diagnostic, separate from native ReLU."""

    return value * _reduced_ring_drelu(value, bits=64)


def hummingbird_eco_relu(value, *, input_bound: float, **_):
    """Exact-under-bound reduced-ring ReLU."""

    fraction_bits = value.encoder._precision_bits
    bits = effective_signed_bits(input_bound, fraction_bits)
    selector = _reduced_ring_drelu(value, bits)
    return value * selector


def hummingbird_budget_relu(
    value,
    *,
    input_bound: float,
    hummingbird_fraction_bits: int,
    **_,
):
    """Reduced-precision HummingBird accuracy/performance ablation."""

    probe = _requantize(value, hummingbird_fraction_bits)
    bits = effective_signed_bits(input_bound, hummingbird_fraction_bits)
    selector = _reduced_ring_drelu(probe, bits)
    return value * selector


def bicoptor2_keybits_relu(
    value,
    *,
    input_bound: float,
    bicoptor_fraction_bits: int,
    **_,
):
    """Bicoptor 2.0 key-bit policy on the common CrypTen 3PC backend."""

    probe = _requantize(value, bicoptor_fraction_bits)
    bits = effective_signed_bits(input_bound, bicoptor_fraction_bits)
    selector = _reduced_ring_drelu(probe, bits)
    return value * selector


def crypten_relu(method: str, value, **kwargs):
    functions = {
        "crypten_native": crypten_native_relu,
        "crypten_fullwidth_adapter": crypten_fullwidth_adapter_relu,
        "hummingbird_eco": hummingbird_eco_relu,
        "hummingbird_budget": hummingbird_budget_relu,
        "bicoptor2_keybits": bicoptor2_keybits_relu,
    }
    if method not in functions:
        raise ValueError(f"unknown secure ReLU method: {method}")
    return functions[method](value, **kwargs)


def quantized_drelu_reference(
    value: torch.Tensor,
    fraction_bits: int,
) -> torch.Tensor:
    """Plain reference for a reduced-precision sign probe and full payload."""

    value = torch.as_tensor(value, dtype=torch.float64)
    scale = 1 << fraction_bits
    probe = torch.round(value * scale) / scale
    return torch.where(probe >= 0, value, torch.zeros_like(value))


def method_reference(
    method: str,
    value: torch.Tensor,
    *,
    hummingbird_fraction_bits: int,
    bicoptor_fraction_bits: int,
) -> torch.Tensor:
    if method in (
        "crypten_native",
        "crypten_fullwidth_adapter",
        "hummingbird_eco",
    ):
        return torch.relu(value)
    if method == "hummingbird_budget":
        return quantized_drelu_reference(value, hummingbird_fraction_bits)
    if method == "bicoptor2_keybits":
        return quantized_drelu_reference(value, bicoptor_fraction_bits)
    raise ValueError(method)
