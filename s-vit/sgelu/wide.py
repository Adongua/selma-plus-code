"""96-bit integers: four little-endian 24-bit limbs, CPU int64 storage.

The public square root accepts ONLY nonnegative opened values below 2**84.
Never pass a single secret share to isqrt_from_limbs.
"""

import torch

LIMB_BITS = 24
NLIMB = 4
LIMB_MASK = (1 << LIMB_BITS) - 1
RING_BITS = 96


def to_limbs(value):
    value %= 1 << RING_BITS
    return torch.tensor([(value >> (24 * i)) & LIMB_MASK for i in range(4)], dtype=torch.int64)


def limbs_from_int64(x):
    # Arithmetic shifting sign-extends negative integers into Z/(2**96).
    return torch.stack([x & LIMB_MASK, (x >> 24) & LIMB_MASK,
                        (x >> 48) & LIMB_MASK, (x < 0).long() * LIMB_MASK], dim=-1)


def limbs_from_uint64(x):
    return torch.stack([x & LIMB_MASK, (x >> 24) & LIMB_MASK,
                        (x >> 48) & 0xFFFF, torch.zeros_like(x)], dim=-1)


def normalize96(x):
    out = torch.empty_like(x)
    carry = torch.zeros_like(x[..., 0])
    for i in range(4):
        value = x[..., i] + carry
        out[..., i] = value & LIMB_MASK
        carry = value >> 24
    return out


def add96(x, y):
    return normalize96(x + y)


def sub96(x, y):
    return normalize96(x - y)


def mul96(x, y):
    x, y = torch.broadcast_tensors(x, y)
    acc = torch.zeros_like(x)
    for i in range(4):
        for j in range(4 - i):
            acc[..., i + j] += x[..., i] * y[..., j]
    return normalize96(acc)


def pack_wire(x):
    low = x[..., 0] | (x[..., 1] << 24)
    high = x[..., 2] | (x[..., 3] << 24)
    return torch.stack([(low & 0xFFFFFFFF).int(),
                        (((low >> 32) & 0xFFFF) | ((high & 0xFFFF) << 16)).int(),
                        ((high >> 16) & 0xFFFFFFFF).int()], dim=-1)


def unpack_wire(words):
    w0, w1, w2 = (words.long() & 0xFFFFFFFF).unbind(-1)
    low = w0 | ((w1 & 0xFFFF) << 32)
    high = ((w1 >> 16) & 0xFFFF) | (w2 << 16)
    return torch.stack([low & LIMB_MASK, (low >> 24) & LIMB_MASK,
                        high & LIMB_MASK, (high >> 24) & LIMB_MASK], dim=-1)


def _square_gt(root, y_high, y_low):
    low, high = root & ((1 << 21) - 1), root >> 21
    cross = 2 * high * low
    acc = low * low + ((cross & ((1 << 21) - 1)) << 21)
    sq_low = acc & ((1 << 42) - 1)
    sq_high = high * high + (cross >> 21) + (acc >> 42)
    return (sq_high > y_high) | ((sq_high == y_high) & (sq_low > y_low))


def isqrt_from_limbs(opened):
    if opened.dtype != torch.int64 or opened.device.type != "cpu" or opened.shape[-1] != 4:
        raise ValueError("expected CPU int64 tensor with four limbs")
    if torch.any(opened < 0) or torch.any(opened > LIMB_MASK) or torch.any(opened[..., 3] >= (1 << 12)):
        raise ValueError("public isqrt requires canonical 0 <= Y < 2**84")
    low = opened[..., 0] + (opened[..., 1] << 24)
    high = opened[..., 2] + (opened[..., 3] << 24)
    root = torch.sqrt(low.double() + high.double() * float(1 << 48)).floor().long()
    y_low = low & ((1 << 42) - 1)
    y_high = (low >> 42) + (high << 6)
    for _ in range(6):
        root -= _square_gt(root, y_high, y_low).long()
    for _ in range(6):
        root += (~_square_gt(root + 1, y_high, y_low)).long()
    if torch.any(_square_gt(root, y_high, y_low)) or torch.any(~_square_gt(root + 1, y_high, y_low)):
        raise ArithmeticError("integer square-root correction failed")
    return root
