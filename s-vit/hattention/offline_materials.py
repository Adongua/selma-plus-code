"""Strict one-use preprocessing for the public-shape full-forward benchmark.

Warmup records provider requests and Ring3 factories. Each measured iteration
generates fresh materials before input sharing, then must consume exactly that
recorded sequence. This is an experimental execution mode, not a security proof.
"""
from collections import deque
from dataclasses import fields, is_dataclass, replace
import json

import torch
import torch.distributed as dist


def _material_nbytes(value):
    """Conservatively count tensor payload bytes, excluding Python overhead."""
    from crypten.cuda import CUDALongTensor
    from crypten.mpc.primitives import ArithmeticSharedTensor, BinarySharedTensor

    if isinstance(value, (ArithmeticSharedTensor, BinarySharedTensor)):
        value = value.share
    if isinstance(value, CUDALongTensor):
        value = value.tensor()
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (tuple, list)):
        return sum(_material_nbytes(item) for item in value)
    if isinstance(value, dict):
        return sum(_material_nbytes(item) for item in value.values())
    if is_dataclass(value) and not isinstance(value, type):
        if getattr(value.__dataclass_params__, "frozen", False):
            return 0
        return sum(_material_nbytes(getattr(value, field.name)) for field in fields(value)
                   if field.name not in {"base", "wide", "runtime", "wide_runtime"})
    return 0


class HostCacheBudget:
    """Bound estimated host tensor payload per party across both material caches."""

    def __init__(self, limit_bytes):
        if not isinstance(limit_bytes, int) or limit_bytes <= 0:
            raise ValueError("host cache budget must be a positive integer")
        self.limit_bytes = limit_bytes
        self.used_bytes = 0
        self.peak_bytes = 0

    def reserve(self, value):
        amount = _material_nbytes(value)
        if self.used_bytes + amount > self.limit_bytes:
            raise RuntimeError("host material cache exceeds the per-party budget "
                               f"({self.limit_bytes / 2**30:.2f} GiB); stop and reassess available RAM")
        self.used_bytes += amount
        self.peak_bytes = max(self.peak_bytes, self.used_bytes)

    def release(self, value):
        self.used_bytes -= _material_nbytes(value)
        if self.used_bytes < 0:
            raise RuntimeError("host material cache accounting became negative")


def _move_material(value, device):
    """Move material tensors without moving or replacing protocol runtimes.

    Provider objects retain their sharing/encoder type. Ring3 material
    dataclasses retain their runtime identity and one-use flags.
    """
    from crypten.cuda import CUDALongTensor
    from crypten.mpc.primitives import ArithmeticSharedTensor, BinarySharedTensor

    device = torch.device(device)
    if isinstance(value, (ArithmeticSharedTensor, BinarySharedTensor)):
        share = value.share.tensor() if isinstance(value.share, CUDALongTensor) else value.share
        share = share.to(device)
        value.share = CUDALongTensor(share) if share.is_cuda else share
        return value
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_material(item, device) for item in value)
    if isinstance(value, list):
        return [_move_material(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_material(item, device) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        if getattr(value.__dataclass_params__, "frozen", False):
            return value
        updates = {field.name: _move_material(getattr(value, field.name), device)
                   for field in fields(value)
                   if field.name not in {"base", "wide", "runtime", "wide_runtime"}}
        return replace(value, **updates)
    return value


def _public_signature(value):
    """Reject tensors: preprocessing requests may depend only on public metadata."""
    if isinstance(value, torch.Tensor):
        raise TypeError("offline request must not contain a tensor")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, (tuple, list, torch.Size)):
        return tuple(_public_signature(x) for x in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), _public_signature(v)) for k, v in value.items()))
    raise TypeError(f"unsupported offline request metadata: {type(value).__name__}")


class StrictProviderCache:
    """Record provider requests once; replay only fresh, precomputed results."""

    METHODS = frozenset(("generate_relu_quintuple", "generate_additive_triple", "square",
                         "generate_binary_triple", "wrap_rng", "B2A_rng"))

    def __init__(self, crypten, storage="device", device=None, budget=None):
        if storage not in ("device", "cpu") or (storage == "cpu" and device is None):
            raise ValueError("CPU material storage requires a replay device")
        if storage == "cpu" and budget is None:
            raise ValueError("CPU material storage requires a shared host cache budget")
        if storage == "device" and budget is not None:
            raise ValueError("device material storage does not use a host cache budget")
        self.crypten = crypten
        self.storage, self.device, self.budget = storage, device, budget
        self.original_getter = crypten.mpc.get_default_provider
        self.provider = self.original_getter()
        if self.provider.tracing or self.provider.tuple_cache:
            raise RuntimeError("offline benchmark requires an empty, non-tracing CrypTen provider cache")
        if set(self.provider.TRACEABLE_FUNCTIONS) != set(self.METHODS):
            raise RuntimeError("provider preprocessing methods changed; review the strict offline cache")
        self.mode = "idle"
        self.requests = []
        self.values = deque()
        self.calls = 0
        self.expected_count = None

    def install(self):
        self.crypten.mpc.get_default_provider = lambda: self

    def close(self):
        self.crypten.mpc.get_default_provider = self.original_getter
        self.values.clear()

    def __getattr__(self, name):
        if name not in self.METHODS:
            return getattr(self.provider, name)

        def call(*args, **kwargs):
            signature = (name, _public_signature(args), _public_signature(kwargs))
            if self.mode == "trace":
                self.requests.append((signature, args, kwargs))
                return getattr(self.provider, name)(*args, **kwargs)
            if self.mode != "replay" or not self.values:
                raise RuntimeError(f"offline provider cache miss: {name}")
            expected, value = self.values.popleft()
            if signature != expected:
                raise RuntimeError(f"offline provider request changed: expected {expected}, got {signature}")
            self.calls += 1
            if self.budget:
                self.budget.release(value)
            return _move_material(value, self.device) if self.storage == "cpu" else value
        return call

    def begin_trace(self):
        if self.requests or self.values or self.mode != "idle":
            raise RuntimeError("provider trace is not fresh")
        self.mode = "trace"

    def end_trace(self):
        if self.mode != "trace":
            raise RuntimeError("provider is not tracing")
        self.mode = "idle"

    def prepare(self, start=0, end=None):
        if self.mode != "idle" or self.values:
            raise RuntimeError("provider material remains from an earlier iteration")
        # Direct calls bypass this proxy: no online values or tensor inputs are used.
        requests = self.requests[start:end]
        self.expected_count = len(requests)
        for index, (signature, args, kwargs) in enumerate(requests, 1):
            name = signature[0]
            value = getattr(self.provider, name)(*args, **kwargs)
            if self.storage == "cpu":
                value = _move_material(value, "cpu")
                self.budget.reserve(value)
            self.values.append((signature, value))
            if (not dist.is_initialized() or dist.get_rank() == 0) and (index % 500 == 0 or index == len(requests)):
                print(json.dumps({"event": "offline_provider_progress", "completed": index,
                                  "total": len(requests), "host_cache_gib_per_rank":
                                  round(self.budget.used_bytes / 2**30, 3) if self.budget else None}), flush=True)
        self.calls = 0

    def begin_replay(self):
        if self.mode != "idle" or self.expected_count is None or len(self.values) != self.expected_count:
            raise RuntimeError("provider materials not completely prepared")
        self.mode = "replay"

    def end_replay(self):
        if self.mode != "replay" or self.values or self.calls != self.expected_count:
            raise RuntimeError("provider material sequence was not consumed exactly once")
        self.mode = "idle"
        self.expected_count = None


class StrictRingCache:
    """Record public Ring3 material requests and consume fresh results in order."""

    def __init__(self, storage="device", device=None, budget=None):
        if storage not in ("device", "cpu") or (storage == "cpu" and device is None):
            raise ValueError("CPU material storage requires a replay device")
        if storage == "cpu" and budget is None:
            raise ValueError("CPU material storage requires a shared host cache budget")
        if storage == "device" and budget is not None:
            raise ValueError("device material storage does not use a host cache budget")
        self.storage, self.device, self.budget = storage, device, budget
        self.mode = "idle"
        self.requests = []
        self.values = deque()
        self.calls = 0
        self.expected_count = None

    def request(self, key, factory):
        key = _public_signature(key)
        if self.mode == "trace":
            self.requests.append((key, factory))
            return factory()
        if self.mode != "replay" or not self.values:
            raise RuntimeError("offline Ring3 material cache miss")
        expected, value = self.values.popleft()
        if key != expected:
            raise RuntimeError(f"offline Ring3 request changed: expected {expected}, got {key}")
        self.calls += 1
        if self.budget:
            self.budget.release(value)
        return _move_material(value, self.device) if self.storage == "cpu" else value

    def begin_trace(self):
        if self.requests or self.values or self.mode != "idle":
            raise RuntimeError("Ring3 trace is not fresh")
        self.mode = "trace"

    def end_trace(self):
        if self.mode != "trace":
            raise RuntimeError("Ring3 is not tracing")
        self.mode = "idle"

    def prepare(self, start=0, end=None):
        if self.mode != "idle" or self.values:
            raise RuntimeError("Ring3 material remains from an earlier iteration")
        requests = self.requests[start:end]
        self.expected_count = len(requests)
        for index, (key, factory) in enumerate(requests, 1):
            value = factory()
            if self.storage == "cpu":
                value = _move_material(value, "cpu")
                self.budget.reserve(value)
            self.values.append((key, value))
            if (not dist.is_initialized() or dist.get_rank() == 0) and (index % 5 == 0 or index == len(requests)):
                print(json.dumps({"event": "offline_ring_progress", "completed": index,
                                  "total": len(requests), "host_cache_gib_per_rank":
                                  round(self.budget.used_bytes / 2**30, 3) if self.budget else None}), flush=True)
        self.calls = 0

    def begin_replay(self):
        if self.mode != "idle" or self.expected_count is None or len(self.values) != self.expected_count:
            raise RuntimeError("Ring3 materials not completely prepared")
        self.mode = "replay"

    def end_replay(self):
        if self.mode != "replay" or self.values or self.calls != self.expected_count:
            raise RuntimeError("Ring3 material sequence was not consumed exactly once")
        self.mode = "idle"
        self.expected_count = None
