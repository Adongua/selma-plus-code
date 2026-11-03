"""Research-only secret share-wrap counting for CrypTen's 3-party truncation.

Not a reviewed secure protocol. Never enabled implicitly. This uses the same
trusted preprocessing provider and security assumptions as CrypTen and adds
binary sharing operations; it must not be described as security validated.
"""
from contextlib import contextmanager


@contextmanager
def experimental_truncation():
    """Temporarily select the experimental protocol in this process only."""
    from crypten.mpc.primitives import beaver

    original = beaver.truncate
    beaver.truncate = experimental_truncate
    try:
        yield
    finally:
        beaver.truncate = original


def experimental_truncate(x, divisor):
    """Count signed addition wraps in binary sharing; preserve CrypTen rounding.

    Experimental only: depends on CrypTen's Boolean circuits and preprocessing.
    Supports positive Python scalar divisors.
    """
    import crypten
    import crypten.communicator as comm
    from crypten.mpc.primitives import beaver
    from crypten.mpc.primitives.binary import BinarySharedTensor

    if not isinstance(divisor, int) or isinstance(divisor, bool) or not 1 <= divisor <= 2**62:
        raise ValueError("experimental truncation requires a scalar integer divisor in [1, 2**62]")
    if divisor == 1:
        return x
    if comm.get().get_world_size() != 3 or crypten.cfg.mpc.protocol != "beaver":
        raise ValueError("experimental truncation requires three-party Beaver sharing")
    # Convert each party's own additive share to XOR sharing, then securely add
    # the three shares. Signed overflow is determined solely by input/output
    # signs, including the edge case -2**63 + -2**63 == 0 modulo 2**64.
    shares = [BinarySharedTensor(x.share, src=i) for i in range(3)]
    partial = shares[0]
    overflow_bits = []
    for other in shares[1:]:
        total = partial + other
        a, b, c = [(value >> 63) & 1 for value in (partial, other, total)]
        overflow_bits.extend([((~a) & (~b) & c) & 1,
                              (a & b & (~c)) & 1])
        partial = total
    bits = beaver.B2A_single_bit(BinarySharedTensor.stack(overflow_bits))
    theta = bits[0] - bits[1] + bits[2] - bits[3]
    x.share = x.share.div_(divisor, rounding_mode="trunc")
    correction = theta * 4 * (int(2**62) // divisor)
    x.share -= correction.share
    return x
