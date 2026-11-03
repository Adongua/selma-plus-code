"""Exact square root of opened unsigned 96-bit values, never secret shares."""
import torch

from sgelu.wide import LIMB_MASK, limbs_from_uint64, mul96


def _greater_square(root, opened):
    overflow = root >= (1 << 48)
    square = mul96(limbs_from_uint64(root), limbs_from_uint64(root))
    greater = torch.zeros_like(root, dtype=torch.bool)
    equal = torch.ones_like(root, dtype=torch.bool)
    for index in (3, 2, 1, 0):
        greater = greater | (equal & (square[..., index] > opened[..., index]))
        equal = equal & (square[..., index] == opened[..., index])
    return overflow | greater


def public_isqrt96(opened):
    if opened.dtype != torch.int64 or opened.shape[-1] != 4:
        raise ValueError("expected int64 tensor with four unsigned limbs")
    if torch.any(opened < 0) or torch.any(opened > LIMB_MASK):
        raise ValueError("noncanonical 96-bit limbs")
    # Floating point supplies a nearby guess; all corrections and the final
    # check use exact 96-bit multiplication and comparison.
    value = sum(opened[..., i].double() * float(1 << (24 * i)) for i in range(4))
    root = torch.sqrt(value).floor().long()
    for _ in range(4):
        root -= _greater_square(root, opened).long()
        root += (~_greater_square(root + 1, opened)).long()
    if torch.any(_greater_square(root, opened)) or torch.any(~_greater_square(root + 1, opened)):
        raise ArithmeticError("96-bit square-root correction failed")
    return root
