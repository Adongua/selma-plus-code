"""Three-process arithmetic sharing for manuscript operator experiments.

CPU/Gloo only, exactly three parties, at most one passive corruption. Every
multiplication output is refreshed before subsequent exchange. This arithmetic
layer does not establish privacy of the Ruler masked openings; see README.md.
"""

import hashlib
import math
import secrets

import numpy as np
import torch
import torch.distributed as dist

from .wide import add96, sub96, mul96, normalize96, pack_wire, unpack_wire, LIMB_MASK


def random_word(shape, device="cpu"):
    raw = secrets.token_bytes(math.prod(shape) * 8)
    return torch.from_numpy(np.frombuffer(raw, dtype="<i8").copy()).reshape(shape).to(device)


class PairwiseZeroSharing:
    """Independent cryptographic XOF streams on the two incident ring edges."""

    def __init__(self, runtime, domain):
        self.runtime = runtime
        seed = torch.tensor(list(secrets.token_bytes(32)), dtype=torch.uint8,
                            device=runtime.device)
        predecessor, = runtime.exchange([seed])
        self.own = bytes(seed.tolist())
        self.pred = bytes(predecessor.tolist())
        self.domain = domain
        self.counter = 0

    def draw(self, shape):
        context = self.domain + self.counter.to_bytes(16, "little")
        self.counter += 1
        count = math.prod(shape) * 8

        def stream(seed):
            raw = hashlib.shake_256(seed + context).digest(count)
            return torch.from_numpy(np.frombuffer(raw, dtype="<i8").copy()).reshape(shape).to(
                self.runtime.device
            )

        return stream(self.own), stream(self.pred)


class Ring3:
    def __init__(self):
        if not dist.is_initialized() or dist.get_world_size() != 3:
            raise RuntimeError("initialize exactly three distributed processes first")
        if dist.get_backend() != "gloo":
            raise ValueError("this reference runtime requires CPU/Gloo")
        self.rank = dist.get_rank()
        self.device = torch.device("cpu")
        self.sent_bytes = 0
        self.phases = 0
        self.zeros = PairwiseZeroSharing(self, b"selma-plus/ring64/v1")

    def exchange(self, tensors):
        sends = [x.contiguous() for x in tensors]
        received = [torch.empty_like(x) for x in sends]
        requests = []
        for x, y in zip(sends, received):
            if x.device.type != "cpu":
                raise ValueError("CPU tensors required")
            requests.append(dist.isend(x, (self.rank + 1) % 3))
            requests.append(dist.irecv(y, (self.rank - 1) % 3))
            self.sent_bytes += x.numel() * x.element_size()
        for request in requests:
            request.wait()
        self.phases += 1
        return received

    def public(self, value):
        value = torch.as_tensor(value, dtype=torch.int64, device="cpu")
        return value.clone() if self.rank == 0 else torch.zeros_like(value)

    def refresh(self, share):
        own, predecessor = self.zeros.draw(tuple(share.shape))
        return share + own - predecessor

    def private(self, value, owner):
        if owner not in (0, 1, 2):
            raise ValueError("owner must be 0, 1 or 2")
        value = torch.as_tensor(value, dtype=torch.int64, device="cpu")
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


class Ring3Wide:
    def __init__(self, base):
        self.base = base
        self.rank = base.rank
        self.zeros = PairwiseZeroSharing(base, b"selma-plus/ring96/v1")

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
        return self.refresh(add96(add96(mul96(left, right), mul96(left, rp)), mul96(lp, right)))

    def reveal(self, value):
        predecessor, = self.exchange([value])
        other, = self.exchange([predecessor])
        return normalize96(value + predecessor + other)
