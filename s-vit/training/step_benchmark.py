"""Instrument one 3PC step; inline subtraction produces estimates, not an offline replay."""
from contextlib import contextmanager
import time

import torch
import torch.distributed as dist

from hattention.offline_materials import StrictProviderCache


class StepBenchmark:
    def __init__(self, crypten, device, base):
        self.crypten, self.device, self.base = crypten, device, base
        self.getter = crypten.mpc.get_default_provider
        self.provider = self.getter()
        if set(self.provider.TRACEABLE_FUNCTIONS) != set(StrictProviderCache.METHODS):
            raise RuntimeError('CrypTen provider methods changed')
        if self.provider.tracing or self.provider.tuple_cache:
            raise RuntimeError('benchmark requires an empty provider cache')
        self.provider_seconds = self.provider_bytes = 0
        self.ring_seconds = self.ring_bytes = self.ring_crypto_bytes = 0
        self.depth = 0

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def crypto_bytes(self):
        return self.crypten.communicator.get().get_communication_stats()['bytes']

    def __getattr__(self, name):
        if name not in StrictProviderCache.METHODS:
            return getattr(self.provider, name)
        def invoke(*args, **kwargs):
            if self.depth:
                return getattr(self.provider, name)(*args, **kwargs)
            self.sync()
            started, before = time.perf_counter(), self.crypto_bytes()
            self.depth += 1
            try:
                return getattr(self.provider, name)(*args, **kwargs)
            finally:
                self.sync()
                self.provider_seconds += time.perf_counter() - started
                self.provider_bytes += self.crypto_bytes() - before
                self.depth -= 1
        return invoke

    def install(self):
        self.crypten.mpc.get_default_provider = lambda: self

    def close(self):
        self.crypten.mpc.get_default_provider = self.getter

    @contextmanager
    def preparation(self):
        self.sync()
        started, crypto, ring = time.perf_counter(), self.crypto_bytes(), self.base.sent_bytes
        provider_seconds, provider_bytes = self.provider_seconds, self.provider_bytes
        try:
            yield
        finally:
            self.sync()
            # Provider calls nested inside Ring3 preparation are accounted once.
            self.ring_seconds += time.perf_counter() - started - (self.provider_seconds - provider_seconds)
            self.ring_crypto_bytes += self.crypto_bytes() - crypto - (self.provider_bytes - provider_bytes)
            self.ring_bytes += self.base.sent_bytes - ring

    def start(self):
        self.sync()
        return (time.perf_counter(), self.crypto_bytes(), self.base.sent_bytes,
                self.provider_seconds, self.provider_bytes, self.ring_seconds,
                self.ring_bytes, self.ring_crypto_bytes)

    def finish(self, snapshot):
        self.sync()
        start, crypto, ring, ps, pb, rs, rb, rcb = snapshot
        wall = time.perf_counter() - start
        provider = self.provider_seconds - ps
        ring_prep = self.ring_seconds - rs
        online = wall - provider - ring_prep
        if online < -1e-5:
            raise RuntimeError('overlapping preprocessing measurements')
        crypto_total = self.crypto_bytes() - crypto
        crypto_offline = self.provider_bytes - pb + self.ring_crypto_bytes - rcb
        ring_total, ring_offline = self.base.sent_bytes - ring, self.ring_bytes - rb
        if not (0 <= crypto_offline <= crypto_total and 0 <= ring_offline <= ring_total):
            raise RuntimeError('preprocessing communication counters overlap or exceed stage totals')
        return {'wall_seconds': wall, 'provider_seconds': provider, 'ring_prepare_seconds': ring_prep,
                'online_estimate_seconds': max(0., online),
                'crypten_total_bytes': crypto_total, 'crypten_preprocessing_bytes': crypto_offline,
                'ring_total_bytes': ring_total, 'ring_preprocessing_bytes': ring_offline}


def aggregate_stages(local):
    """Gather stages after measurement; system bytes use CrypTen's sent+recv / 2 convention."""
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    stages = {}
    for name in ('forward', 'backward', 'optimizer'):
        if name not in local:
            stages[name] = {'online_estimate_seconds': 0.0, 'online_communication_bytes': 0}
            continue
        rows = [r[name] for r in gathered]
        stages[name] = {
            'online_estimate_seconds': max(r['online_estimate_seconds'] for r in rows),
            'online_communication_bytes': sum(r['crypten_total_bytes']-r['crypten_preprocessing_bytes'] for r in rows)/2
                + sum(r['ring_total_bytes']-r['ring_preprocessing_bytes'] for r in rows),
            'per_rank': rows,
        }
    stages['total'] = {
        'online_estimate_seconds': max(sum(stage['online_estimate_seconds'] for stage in r.values()) for r in gathered),
        'online_communication_bytes': sum(stages[s]['online_communication_bytes'] for s in ('forward', 'backward', 'optimizer')),
    }
    return stages
