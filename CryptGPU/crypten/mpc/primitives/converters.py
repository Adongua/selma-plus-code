#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten.communicator as comm
import torch
from crypten.encoder import FixedPointEncoder

from ..ptype import ptype as Ptype
from . import beaver, resharing, circuit
from .arithmetic import ArithmeticSharedTensor
from .binary import BinarySharedTensor


def _A2B(arithmetic_tensor):
    assert comm.get().get_world_size() == 3
    rank = comm.get().get_rank()

    size = arithmetic_tensor.size()
    device = arithmetic_tensor.device

    z1, z2 = BinarySharedTensor.PRZS(size, device=device).share, BinarySharedTensor.PRZS(size, device=device).share

    x1, x2 = arithmetic_tensor.share, resharing.replicate_shares(arithmetic_tensor.share)

    if rank == 0:
        b1 = BinarySharedTensor.from_shares(z1 ^ (x1 + x2), src=rank)
        b2 = BinarySharedTensor.from_shares(z2, src=rank)
    elif rank == 1:
        b1 = BinarySharedTensor.from_shares(z1, src=rank)
        b2 = BinarySharedTensor.from_shares(z2 ^ x1, src=rank)
    else:
        b1 = BinarySharedTensor.from_shares(z1, src=rank)
        b2 = BinarySharedTensor.from_shares(z2, src=rank)

    binary_tensor = circuit.extract_msb(b1, b2)
    binary_tensor.encoder = arithmetic_tensor.encoder

    return binary_tensor


def _B2A(binary_tensor, bits=1, precision=None):
    """Convert a secret bit from XOR sharing to arithmetic sharing.

    CryptGPU's comparison circuit returns the sign as a binary-shared bit.
    The original tree calls ``to(Ptype.arithmetic, bits=1)`` afterwards but
    omitted that conversion.  Reuse the repository's three-party mixed OT to
    multiply the bit by a public one without opening it.

    Only one-bit conversion is needed by the comparison API.  Keeping the
    restriction explicit prevents silently treating a multi-bit XOR sharing
    as an arithmetic integer.
    """
    if bits != 1:
        raise NotImplementedError("CryptGPU B2A currently supports one bit")

    rank = comm.get().get_rank()
    ones = torch.ones_like(binary_tensor.share.data) if rank == 0 else None
    result = resharing.mixed_mul_scalar(
        ones, binary_tensor, bits=1, roles=[0, 1, 2]
    )
    if precision is None:
        precision = binary_tensor.encoder._precision_bits
    result.encoder = FixedPointEncoder(precision_bits=precision)
    return result


def convert(tensor, ptype, **kwargs):
    tensor_name = ptype.to_tensor()
    if isinstance(tensor, tensor_name):
        return tensor
    if isinstance(tensor, ArithmeticSharedTensor) and ptype == Ptype.binary:
        return _A2B(tensor)
    if isinstance(tensor, BinarySharedTensor) and ptype == Ptype.arithmetic:
        return _B2A(
            tensor,
            bits=kwargs.get("bits", 1),
            precision=kwargs.get("precision"),
        )
    else:
        raise TypeError("Cannot convert %s to %s" % (type(tensor), ptype.__name__))


def get_msb(arithmetic_tensor):
    return _A2B(arithmetic_tensor)
