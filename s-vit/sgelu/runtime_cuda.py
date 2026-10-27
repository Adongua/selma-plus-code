"""CUDA runtime for the experimental three-party arithmetic protocol.

This module is intentionally separate from :mod:`sgelu.runtime`.  It expects
one process per GPU, exactly three processes, and an NCCL or Gloo process
group. Gloo exchanges explicitly stage secret shares through host memory. It
is an engineering port of the CPU/Gloo prototype; it does not add
new security guarantees.  Arithmetic, random-word expansion and public 96-bit
square roots stay on CUDA after one-time seed setup.
"""

import hashlib
import secrets

import torch
import torch.distributed as dist

from .wide import add96, sub96, mul96, normalize96, pack_wire, unpack_wire, LIMB_MASK


def _seed_from_bytes(seed, domain):
    digest = hashlib.blake2b(seed + domain, digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def _cuda_words(shape, generator, device):
    """Uniform words in Z/(2**64) generated without a host tensor transfer."""
    parts = torch.randint(0, 1 << 32, (2,) + tuple(shape), dtype=torch.int64,
                          device=device, generator=generator)
    return parts[0] | (parts[1] << 32)


class CudaPairwiseZeroSharing:
    """Matching edge streams expanded by CUDA Philox.

    PyTorch's Philox generator is useful for GPU performance experiments but
    is not claimed to be a cryptographic PRG.  The surrounding Ruler protocol
    is already marked not-security-validated; production MPC requires a
    reviewed GPU CSPRNG instead.
    """

    def __init__(self, runtime, domain):
        self.runtime = runtime
        seed = torch.tensor(list(secrets.token_bytes(32)), dtype=torch.uint8,
                            device=runtime.device)
        predecessor, = runtime.exchange([seed])
        own_bytes = bytes(seed.cpu().tolist())
        pred_bytes = bytes(predecessor.cpu().tolist())
        self.own = torch.Generator(device=runtime.device)
        self.pred = torch.Generator(device=runtime.device)
        self.own.manual_seed(_seed_from_bytes(own_bytes, domain))
        self.pred.manual_seed(_seed_from_bytes(pred_bytes, domain))

    def draw(self, shape):
        return (_cuda_words(shape, self.own, self.runtime.device),
                _cuda_words(shape, self.pred, self.runtime.device))


class CudaRing3:
    """Three-party additive ring over CUDA tensors, using NCCL or staged Gloo."""

    def __init__(self, rank=None, device=None, reveal_mode="all_gather"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if not dist.is_initialized() or dist.get_world_size() != 3:
            raise RuntimeError("initialize exactly three distributed processes first")
        self.backend = str(dist.get_backend()).lower()
        if self.backend not in ("nccl", "gloo"):
            raise ValueError("CUDA runtime requires the NCCL or Gloo backend")
        self.rank = dist.get_rank() if rank is None else rank
        if self.rank != dist.get_rank():
            raise ValueError("rank does not match the initialized process group")
        self.device = torch.device(device or f"cuda:{torch.cuda.current_device()}")
        if reveal_mode not in ("all_reduce", "all_gather", "p2p"):
            raise ValueError("reveal_mode must be 'all_reduce', 'all_gather' or 'p2p'")
        self.reveal_mode = reveal_mode
        self.sent_bytes = 0
        self.phases = 0
        local_seed = secrets.token_bytes(32)
        self.local_generator = torch.Generator(device=self.device)
        self.local_generator.manual_seed(
            _seed_from_bytes(local_seed, b"selma-plus/local/cuda-v2")
        )
        self.zeros = CudaPairwiseZeroSharing(self, b"selma-plus/ring64/cuda-v2")

    def random_word(self, shape):
        return _cuda_words(tuple(shape), self.local_generator, self.device)

    def exchange(self, tensors):
        sends = [x.contiguous() for x in tensors]
        for x in sends:
            if x.device != self.device:
                raise ValueError(f"CUDA tensor on {self.device} required")
        wire_sends = [x.cpu() for x in sends] if self.backend == "gloo" else sends
        received = [torch.empty_like(x) for x in wire_sends]
        operations = []
        for x, y in zip(wire_sends, received):
            # NCCL may deadlock while lazily creating peer communicators if
            # every rank issues send/recv calls in a different order.  Submit
            # the complete ring exchange as one batch instead.
            operations.append(dist.P2POp(dist.isend, x, (self.rank + 1) % 3))
            operations.append(dist.P2POp(dist.irecv, y, (self.rank - 1) % 3))
            self.sent_bytes += x.numel() * x.element_size()
        requests = dist.batch_isend_irecv(operations)
        for request in requests:
            request.wait()
        self.phases += 1
        return ([x.to(self.device) for x in received]
                if self.backend == "gloo" else received)

    def gather_all(self, value):
        """Collect equal-sized CUDA tensors from all parties in one NCCL phase."""
        value = value.contiguous()
        if value.device != self.device:
            raise ValueError(f"CUDA tensor on {self.device} required")
        flat = value.reshape(-1)
        if self.backend == "gloo":
            wire = flat.cpu()
            parts = [torch.empty_like(wire) for _ in range(3)]
            dist.all_gather(parts, wire)
            gathered = torch.cat(parts).to(self.device)
        else:
            gathered = torch.empty(3 * flat.numel(), dtype=value.dtype, device=self.device)
            dist.all_gather_into_tensor(gathered, flat)
        # A three-party ring all-gather transmits two input-sized chunks per
        # rank.  This matches the payload accounting of the former two P2P
        # reconstruction exchanges and keeps CPU/GPU reports comparable.
        self.sent_bytes += 2 * value.numel() * value.element_size()
        self.phases += 1
        return gathered.reshape((3,) + tuple(value.shape))

    def reduce_all(self, value):
        """Reconstruct equal-sized shares using one NCCL integer all-reduce."""
        value = value.contiguous()
        if value.device != self.device:
            raise ValueError(f"CUDA tensor on {self.device} required")
        wire = value.cpu() if self.backend == "gloo" else value
        dist.all_reduce(wire, op=dist.ReduceOp.SUM)
        # Keep the same conservative two-input-sized accounting used for the
        # ring reconstruction paths. NCCL's selected algorithm may transfer
        # less or more than this application-level equivalent.
        self.sent_bytes += 2 * value.numel() * value.element_size()
        self.phases += 1
        return wire.to(self.device) if self.backend == "gloo" else wire

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
        if self.reveal_mode == "all_reduce":
            return self.reduce_all(value)
        if self.reveal_mode == "all_gather":
            return self.gather_all(value).sum(dim=0)
        predecessor, = self.exchange([value])
        other, = self.exchange([predecessor])
        return value + predecessor + other


class CudaRing3Wide:
    """96-bit limb wrapper matching the CPU ``Ring3Wide`` interface."""

    def __init__(self, base):
        self.base = base
        self.rank = base.rank
        self.zeros = CudaPairwiseZeroSharing(base, b"selma-plus/ring96/cuda-v2")

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
        if self.base.reveal_mode == "all_reduce":
            return normalize96(self.base.reduce_all(value))
        if self.base.reveal_mode == "all_gather":
            # Gather packed shares to retain the 3x32-bit wire format.  Packed
            # shares must be unpacked before addition so carries are normalized
            # in the 4x24-bit arithmetic representation rather than crossing
            # packed fields.
            gathered = self.base.gather_all(pack_wire(value))
            return normalize96(unpack_wire(gathered).sum(dim=0))
        predecessor, = self.exchange([value])
        other, = self.exchange([predecessor])
        return normalize96(value + predecessor + other)
