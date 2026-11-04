"""Strict one-use preprocessing for the public-shape full-forward benchmark.

Warmup records provider requests and Ring3 factories. Each measured iteration
generates fresh materials before input sharing, then must consume exactly that
recorded sequence. This is an experimental execution mode, not a security proof.
"""
from collections import deque
import json

import torch
import torch.distributed as dist


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

    def __init__(self, crypten):
        self.crypten = crypten
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
            return value
        return call

    def begin_trace(self):
        if self.requests or self.values or self.mode != "idle":
            raise RuntimeError("provider trace is not fresh")
        self.mode = "trace"

    def end_trace(self):
        if self.mode != "trace":
            raise RuntimeError("provider is not tracing")
        self.mode = "idle"

    def prepare(self):
        if self.mode != "idle" or self.values:
            raise RuntimeError("provider material remains from an earlier iteration")
        # Direct calls bypass this proxy: no online values or tensor inputs are used.
        for index, (signature, args, kwargs) in enumerate(self.requests, 1):
            name = signature[0]
            self.values.append((signature, getattr(self.provider, name)(*args, **kwargs)))
            if (not dist.is_initialized() or dist.get_rank() == 0) and (index % 500 == 0 or index == len(self.requests)):
                print(json.dumps({"event": "offline_provider_progress", "completed": index,
                                  "total": len(self.requests)}), flush=True)
        self.calls = 0

    def begin_replay(self):
        if self.mode != "idle" or len(self.values) != len(self.requests):
            raise RuntimeError("provider materials not completely prepared")
        self.mode = "replay"

    def end_replay(self):
        if self.mode != "replay" or self.values or self.calls != len(self.requests):
            raise RuntimeError("provider material sequence was not consumed exactly once")
        self.mode = "idle"


class StrictRingCache:
    """Record public Ring3 material requests and consume fresh results in order."""

    def __init__(self):
        self.mode = "idle"
        self.requests = []
        self.values = deque()
        self.calls = 0

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
        return value

    def begin_trace(self):
        if self.requests or self.values or self.mode != "idle":
            raise RuntimeError("Ring3 trace is not fresh")
        self.mode = "trace"

    def end_trace(self):
        if self.mode != "trace":
            raise RuntimeError("Ring3 is not tracing")
        self.mode = "idle"

    def prepare(self):
        if self.mode != "idle" or self.values:
            raise RuntimeError("Ring3 material remains from an earlier iteration")
        for index, (key, factory) in enumerate(self.requests, 1):
            self.values.append((key, factory()))
            if (not dist.is_initialized() or dist.get_rank() == 0) and (index % 5 == 0 or index == len(self.requests)):
                print(json.dumps({"event": "offline_ring_progress", "completed": index,
                                  "total": len(self.requests)}), flush=True)
        self.calls = 0

    def begin_replay(self):
        if self.mode != "idle" or len(self.values) != len(self.requests):
            raise RuntimeError("Ring3 materials not completely prepared")
        self.mode = "replay"

    def end_replay(self):
        if self.mode != "replay" or self.values or self.calls != len(self.requests):
            raise RuntimeError("Ring3 material sequence was not consumed exactly once")
        self.mode = "idle"
