"""Instrument just-in-time provider calls; estimates, not precomputed execution."""
from contextlib import contextmanager
import time

import torch
import torch.distributed as dist

from .offline_materials import StrictProviderCache


class InlineProviderTimer:
    def __init__(self, crypten, device, sync=None, clock=None):
        self.crypten, self.device = crypten, device
        self.sync = sync or (lambda: torch.cuda.synchronize(device))
        self.clock = clock or time.perf_counter
        self.getter = crypten.mpc.get_default_provider
        self.provider = self.getter()
        if set(self.provider.TRACEABLE_FUNCTIONS) != set(StrictProviderCache.METHODS):
            raise RuntimeError("provider methods changed; review timing coverage")
        if self.provider.tracing or self.provider.tuple_cache:
            raise RuntimeError("inline timing requires an empty provider cache")
        self.seconds = 0.0
        self.calls = 0
        self.depth = 0

    def __getattr__(self, name):
        if name not in StrictProviderCache.METHODS:
            return getattr(self.provider, name)
        def call(*args, **kwargs):
            if self.depth:
                return getattr(self.provider, name)(*args, **kwargs)
            self.sync()
            start = self.clock()
            self.depth += 1
            try:
                return getattr(self.provider, name)(*args, **kwargs)
            finally:
                self.sync()
                self.seconds += self.clock() - start
                self.calls += 1
                self.depth -= 1
        return call

    @contextmanager
    def installed(self):
        self.crypten.mpc.get_default_provider = lambda: self
        try:
            yield self
        finally:
            self.crypten.mpc.get_default_provider = self.getter


def local_accounting(wall, provider, ring):
    values = (wall, provider, ring)
    if any(not 0 <= x < float('inf') for x in values):
        raise ValueError('invalid timing value')
    online = wall - provider - ring
    if online < -1e-6:
        raise RuntimeError('preprocessing intervals overlap or exceed forward wall')
    return [wall, provider, ring, max(online, 0.0)]


def gather_accounting(wall, provider, ring, calls):
    local = torch.tensor(local_accounting(wall, provider, ring) + [calls], dtype=torch.float64)
    rows = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(rows, local)
    names = ('wall_seconds', 'crypten_provider_seconds', 'ring_preprocessing_seconds', 'online_estimate_seconds', 'provider_calls')
    ranks = [dict(zip(names, r.tolist())) for r in rows]
    return {'per_rank': ranks, 'online_estimate_seconds_max': max(r['online_estimate_seconds'] for r in ranks),
            'note': 'Local wall minus non-overlapping instrumented provider and Ring3 preparation; not precomputed end-to-end latency. Provider timing synchronizes GPU; local randomness outside these boundaries remains online.'}
