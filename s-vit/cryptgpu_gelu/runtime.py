"""Adapter that executes the manuscript sGeLU protocol over CryptGPU comms.

CryptGPU owns process groups, correlated edge RNG streams, and CUDA transport.
The adapter exposes the narrow arithmetic interface already consumed by
``sgelu.protocol`` so the G2/Ruler implementation is not duplicated.
"""

import secrets
import time

import torch
import torch.distributed as dist

from sgelu.wide import (LIMB_MASK, add96, mul96, normalize96, pack_wire,
                        sub96, unpack_wire)


def _words(shape, generator, device):
    parts = torch.randint(0, 1 << 32, (2,) + tuple(shape), dtype=torch.int64,
                          generator=generator, device=device)
    return parts[0] | (parts[1] << 32)


class CryptGpuRing3:
    """Raw-share arithmetic using CryptGPU's communicator and edge PRGs."""

    def __init__(self, device):
        import crypten.communicator as comm

        self.comm = comm.get()
        if self.comm.get_world_size() != 3:
            raise RuntimeError("CryptGPU sGeLU requires exactly three parties")
        self.rank = self.comm.get_rank()
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CryptGPU sGeLU requires CUDA")
        self.sent_bytes = 0
        self.phases = 0
        self.local_generator = torch.Generator(device=self.device)
        self.local_generator.manual_seed(secrets.randbits(63))

    def _edge_zero(self, shape):
        own = _words(shape, self.comm.get_generator(0, device=self.device), self.device)
        predecessor = _words(shape, self.comm.get_generator(1, device=self.device), self.device)
        return own - predecessor

    def random_word(self, shape):
        return _words(tuple(shape), self.local_generator, self.device)

    def exchange(self, tensors):
        next_rank, previous_rank = (self.rank + 1) % 3, (self.rank - 1) % 3
        sends = [x.contiguous() for x in tensors]
        received = [torch.empty_like(x) for x in sends]
        operations = []
        for source, target in zip(sends, received):
            operations.append(dist.P2POp(
                dist.isend, source, next_rank, group=self.comm.main_group
            ))
            operations.append(dist.P2POp(
                dist.irecv, target, previous_rank, group=self.comm.main_group
            ))
            self.sent_bytes += source.numel() * source.element_size()
        started = time.perf_counter()
        requests = dist.batch_isend_irecv(operations)
        for request in requests:
            request.wait()
        elapsed = time.perf_counter() - started
        if self.comm.is_verbose():
            self.comm.comm_rounds += 1
            self.comm.comm_bytes += sum(
                source.numel() * source.element_size() for source in sends
            )
            self.comm.comm_time += elapsed
        self.phases += 1
        return received

    def public(self, value):
        value = torch.as_tensor(value, dtype=torch.int64, device=self.device)
        return value.clone() if self.rank == 0 else torch.zeros_like(value)

    def refresh(self, share):
        return share + self._edge_zero(tuple(share.shape))

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
        from crypten.cuda import CUDALongTensor
        opened = self.comm.all_reduce(CUDALongTensor(value.contiguous()))
        self.sent_bytes += 2 * value.numel() * value.element_size()
        self.phases += 1
        return opened.data


class CryptGpuRing3Wide:
    """96-bit limb arithmetic sharing on CryptGPU's correlated edge PRGs."""

    def __init__(self, base):
        self.base, self.rank = base, base.rank

    def _edge_zero(self, shape):
        own = torch.randint(0, 1 << 24, tuple(shape), dtype=torch.int64,
                            generator=self.base.comm.get_generator(0, device=self.base.device),
                            device=self.base.device)
        predecessor = torch.randint(
            0, 1 << 24, tuple(shape), dtype=torch.int64,
            generator=self.base.comm.get_generator(1, device=self.base.device),
            device=self.base.device,
        )
        return sub96(own & LIMB_MASK, predecessor & LIMB_MASK)

    def exchange(self, tensors):
        return [unpack_wire(x) for x in self.base.exchange([pack_wire(x) for x in tensors])]

    def public(self, limbs):
        return limbs.clone() if self.rank == 0 else torch.zeros_like(limbs)

    def refresh(self, share):
        return add96(share, self._edge_zero(tuple(share.shape)))

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
        # Unpack each share before addition so carries do not cross packed
        # fields inside a 64-bit all-reduce.
        predecessor, = self.exchange([value])
        other, = self.exchange([predecessor])
        return normalize96(value + predecessor + other)


class CryptGpuSGeLU:
    """One-use prepared sGeLU operator wrapping a CryptGPU MPCTensor."""

    def __init__(self, device, mu):
        from sgelu.protocol import prepare_g2
        self.base = CryptGpuRing3(device)
        self.wide = CryptGpuRing3Wide(self.base)
        self.mu = float(mu)
        self._prepare = prepare_g2
        self.material = None

    def prepare(self, shape):
        if self.material is not None:
            raise RuntimeError("consume pending material before preparing another call")
        self.material = self._prepare(self.base, self.wide, tuple(shape), self.mu)

    def __call__(self, value):
        from crypten.cuda import CUDALongTensor
        from crypten.mpc import MPCTensor, ptype
        from sgelu.protocol import ruler_g2

        if not isinstance(value, MPCTensor) or value.ptype != ptype.arithmetic:
            raise ValueError("expected a CryptGPU arithmetic MPCTensor")
        if value.encoder._precision_bits != 17:
            raise ValueError("sGeLU requires Q17 CryptGPU input")
        if self.material is None:
            raise RuntimeError("call prepare(shape) before each evaluation")
        material, self.material = self.material, None
        raw_share = value.share.data if hasattr(value.share, "data") else value.share
        output = ruler_g2(self.base, self.wide, raw_share, material)
        return MPCTensor.from_shares(CUDALongTensor(output), precision=17)
