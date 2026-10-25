"""CUDA/NCCL runtime for the experimental three-party arithmetic protocol.

This module is intentionally separate from :mod:`sgelu.runtime`.  It expects
one process per GPU, exactly three processes, and an initialized NCCL process
group.  It is an engineering port of the CPU/Gloo prototype; it does not add
new security guarantees.  Public 96-bit square roots temporarily move the
already-opened value to CPU because the reference integer square-root routine
is CPU based.
"""

import torch
import torch.distributed as dist

from .runtime import PairwiseZeroSharing
from .wide import add96, sub96, mul96, normalize96, pack_wire, unpack_wire, LIMB_MASK


class CudaRing3:
    """Three-party additive ring over CUDA tensors using NCCL point-to-point."""

    def __init__(self, rank=None, device=None):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if not dist.is_initialized() or dist.get_world_size() != 3:
            raise RuntimeError("initialize exactly three distributed processes first")
        if dist.get_backend() != "nccl":
            raise ValueError("CUDA runtime requires the NCCL backend")
        self.rank = dist.get_rank() if rank is None else rank
        if self.rank != dist.get_rank():
            raise ValueError("rank does not match the initialized process group")
        self.device = torch.device(device or f"cuda:{torch.cuda.current_device()}")
        self.sent_bytes = 0
        self.phases = 0
        self.zeros = PairwiseZeroSharing(self, b"selma-plus/ring64/cuda-v1")

    def exchange(self, tensors):
        sends = [x.contiguous() for x in tensors]
        received = [torch.empty_like(x, device=self.device) for x in sends]
        requests = []
        for x, y in zip(sends, received):
            if x.device != self.device:
                raise ValueError(f"CUDA tensor on {self.device} required")
            requests.append(dist.isend(x, (self.rank + 1) % 3))
            requests.append(dist.irecv(y, (self.rank - 1) % 3))
            self.sent_bytes += x.numel() * x.element_size()
        for request in requests:
            request.wait()
        self.phases += 1
        return received

    def public(self, value):
        value = torch.as_tensor(value, dtype=torch.int64, device=self.device)
        return value.clone() if self.rank == 0 else torch.zeros_like(value)

    def refresh(self, share):
        own, predecessor = self.zeros.draw(tuple(share.shape))
        return share + own - predecessor

    def private(self, value, owner):
        if owner not in (0, 1, 2):
            raise ValueError("owner must be 0, 1 or 2")
        value = torch.as_tensor(value, dtype=torch.int64, device=self.device)
        return self.refresh(value if self.rank == owner else torch.zeros_like(value))

    def mul(self, left, right):
        left, right = torch.broadcast_tensors(left, right)
        lp, rp = self.exchange([left, right])
        return self.refresh(left * right + left * rp + lp * right)

    def square(self, value):
        predecessor, = self.exchange([value])
        return self.refresh(value.square() + 2 * value * predecessor)

    def reveal(self, value):
        predecessor, = self.exchange([value])
        other, = self.exchange([predecessor])
        return value + predecessor + other


class CudaRing3Wide:
    """96-bit limb wrapper matching the CPU ``Ring3Wide`` interface."""

    def __init__(self, base):
        self.base = base
        self.rank = base.rank
        self.zeros = PairwiseZeroSharing(base, b"selma-plus/ring96/cuda-v1")

    def exchange(self, tensors):
        return [unpack_wire(x) for x in self.base.exchange([pack_wire(x) for x in tensors])]

    def public(self, limbs):
        return limbs.clone() if self.rank == 0 else torch.zeros_like(limbs)

    def refresh(self, share):
        own, predecessor = self.zeros.draw(tuple(share.shape))
        return add96(share, sub96(own & LIMB_MASK, predecessor & LIMB_MASK))

    def private(self, value, owner):
        if owner not in (0, 1, 2):
            raise ValueError("owner must be 0, 1 or 2")
        return self.refresh(value if self.rank == owner else torch.zeros_like(value))

    def mul(self, left, right):
        left, right = torch.broadcast_tensors(left, right)
        lp, rp = self.exchange([left, right])
        return self.refresh(add96(add96(mul96(left, right), mul96(left, rp)),
                                  mul96(lp, right)))

    def reveal(self, value):
        predecessor, = self.exchange([value])
        other, = self.exchange([predecessor])
        return normalize96(value + predecessor + other)
