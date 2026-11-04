"""Research-only secret share-wrap counting for CrypTen's 3-party truncation.

Not a reviewed secure protocol. Never enabled implicitly. This uses the same
trusted preprocessing provider and security assumptions as CrypTen and adds
binary sharing operations; it must not be described as security validated.
"""
from contextlib import contextmanager


@contextmanager
def experimental_truncation(mode="experimental"):
    """Temporarily select the experimental protocol in this process only."""
    from crypten.mpc.primitives import beaver

    original = beaver.truncate
    methods = {"experimental": experimental_truncate, "batched": batched_truncate,
               "carry_save": carry_save_truncate}
    if mode not in methods:
        raise ValueError("unknown experimental truncation mode")
    beaver.truncate = methods[mode]
    try:
        yield
    finally:
        beaver.truncate = original


def experimental_truncate(x, divisor):
    """Count signed addition wraps in binary sharing; preserve CrypTen rounding.

    Experimental only: depends on CrypTen's Boolean circuits and preprocessing.
    Supports positive Python scalar divisors.
    """
    return _truncate(x, divisor, batched=False)


def batched_truncate(x, divisor):
    """Experimental wrap counter with batched sign circuits; same assumptions."""
    return _truncate(x, divisor, batched=True)


def carry_save_truncate(x, divisor):
    """Count three-party wraps with one carry-save and one prefix addition.

    Opt-in research variant; uses CrypTen's existing Boolean AND, prefix carry,
    and masked B2A conversion. No input, output, sign or wrap bit is opened here.
    """
    _validate(divisor)
    if divisor == 1:
        return x
    theta = _carry_save_wraps(x)
    return _correct(x, divisor, theta)


def _carry_save_wraps(x):
    """Return arithmetic shares of (sum(signed shares) - signed value) / 2**64.

    For unsigned words a,b,c, let s=a XOR b XOR c and m=majority(a,b,c).
    Their integer sum is s+2*m. With t=(2*m) mod 2**64, unsigned wraps are
    msb(m)+carry_out(s+t). Signed wraps additionally add msb(s+t) and subtract
    the three input sign bits. Each party subtracts its own LOCAL share's sign
    directly from its arithmetic share of theta, without converting or opening
    these three bits. The result is exact even at signed int64 boundaries.
    """
    from crypten.mpc.primitives import beaver, circuit
    from crypten.mpc.primitives.binary import BinarySharedTensor

    a, b, c = [BinarySharedTensor(x.share, src=i) for i in range(3)]
    ab = a ^ b
    # The two products are disjoint, so XOR implements their OR. Batch both
    # independent ANDs in one round instead of propagating a first 64-bit carry.
    products = BinarySharedTensor.stack([a, ab]) & BinarySharedTensor.stack([b, c])
    majority = products[0] ^ products[1]
    xor_sum = ab ^ c
    del a, b, c, ab, products
    shifted = majority << 1
    propagate = xor_sum ^ shifted
    # Reuse the bundled adder's prefix carry, retaining its bit-63 carry-out.
    # This private CrypTen API is covered by full-ring and boundary tests.
    carry, _ = circuit.__SPK_circuit(xor_sum & shifted, propagate)
    total = propagate ^ (carry << 1)
    binary_bits = BinarySharedTensor.stack([(majority >> 63) & 1,
                                            (carry >> 63) & 1,
                                            (total >> 63) & 1])
    theta = beaver.B2A_single_bit(binary_bits).sum(0)
    theta.share -= (x.share >> 63) & 1
    return theta


def _validate(divisor):
    import crypten
    import crypten.communicator as comm

    if not isinstance(divisor, int) or isinstance(divisor, bool) or not 1 <= divisor <= 2**62:
        raise ValueError("experimental truncation requires a scalar integer divisor in [1, 2**62]")
    if divisor != 1 and (comm.get().get_world_size() != 3 or crypten.cfg.mpc.protocol != "beaver"):
        raise ValueError("experimental truncation requires three-party Beaver sharing")


def _correct(x, divisor, theta):
    # Keep the previous variants' local signed division and rounding behavior.
    x.share = x.share.div_(divisor, rounding_mode="trunc")
    correction = theta * 4 * (int(2**62) // divisor)
    x.share -= correction.share
    return x


def _truncate(x, divisor, *, batched):
    from crypten.mpc.primitives import beaver
    from crypten.mpc.primitives.binary import BinarySharedTensor

    _validate(divisor)
    if divisor == 1:
        return x
    # Convert each party's own additive share to XOR sharing, then securely add
    # the three shares. Signed overflow is determined solely by input/output
    # signs, including the edge case -2**63 + -2**63 == 0 modulo 2**64.
    shares = [BinarySharedTensor(x.share, src=i) for i in range(3)]
    partial = shares[0]
    overflow_bits = []
    sign_rows = []
    for other in shares[1:]:
        total = partial + other
        a, b, c = [(value >> 63) & 1 for value in (partial, other, total)]
        if batched:
            sign_rows.append((a, b, c))
        else:
            overflow_bits.extend([((~a) & (~b) & c) & 1,
                                  (a & b & (~c)) & 1])
        partial = total
    if batched:
        a, b, c = [BinarySharedTensor.stack([row[i] for row in sign_rows]) for i in range(3)]
        # Overflow iff both input signs differ from the sum sign.  The sign of
        # the first input identifies negative overflow. Batch the two additions.
        overflow = (a ^ c) & (b ^ c)
        negative = overflow & a
        bits = beaver.B2A_single_bit(BinarySharedTensor.stack([overflow, negative]))
        theta = (bits[0] - 2 * bits[1]).sum(0)
    else:
        bits = beaver.B2A_single_bit(BinarySharedTensor.stack(overflow_bits))
        theta = bits[0] - bits[1] + bits[2] - bits[3]
    return _correct(x, divisor, theta)
