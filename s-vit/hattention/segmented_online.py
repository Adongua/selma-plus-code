"""Bounded-memory segmented online timing for fixed-shape public fixtures.

Each segment uses fresh material generated immediately before that segment.
Only online segment durations are summed; this is an instrumented estimate,
not the wall time of an uninterrupted run with all materials preloaded.
"""
import json
import time

import torch
import torch.distributed as dist

from slayernorm.benchmark_crypten_cuda import _aggregate, _barrier


def _zero_stats():
    return {"bytes": 0, "rounds": 0, "time": 0.0}


def _add_stats(total, current):
    for key in total:
        total[key] += current[key]


class SegmentedOnline:
    """Trace public-shape requests then replay each segment with fresh material."""

    def __init__(self, crypten, device, provider, ring=None, base=None, budget=None):
        self.crypten, self.device = crypten, torch.device(device)
        self.provider, self.ring, self.base, self.budget = provider, ring, base, budget
        self.segments = []
        self.open_segment = None
        self.mode = "trace"

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def trace_event(self, event, name):
        if self.mode != "trace":
            raise RuntimeError("segmented trace is not active")
        if event == "start":
            if self.open_segment is not None:
                raise RuntimeError("nested segments are unsupported")
            if self.segments and (self.segments[-1][2] != len(self.provider.requests) or
                                  (self.ring and self.segments[-1][4] != len(self.ring.requests))):
                raise RuntimeError("material requests occurred between segments")
            self.open_segment = (name, len(self.provider.requests),
                                 len(self.ring.requests) if self.ring else 0)
        elif event == "end":
            if self.open_segment is None or self.open_segment[0] != name:
                raise RuntimeError("segment trace order changed")
            _, p0, r0 = self.open_segment
            self.segments.append((name, p0, len(self.provider.requests), r0,
                                  len(self.ring.requests) if self.ring else 0))
            self.open_segment = None
        else:
            raise ValueError("unknown segment event")

    def end_trace(self):
        if self.open_segment is not None or not self.segments:
            raise RuntimeError("incomplete segmented trace")
        if self.segments[0][1] != 0 or self.segments[-1][2] != len(self.provider.requests):
            raise RuntimeError("provider requests outside recorded segments")
        if self.ring and (self.segments[0][3] != 0 or self.segments[-1][4] != len(self.ring.requests)):
            raise RuntimeError("Ring3 requests outside recorded segments")
        self.mode = "idle"

    def begin_run(self):
        if self.mode != "idle":
            raise RuntimeError("segmented profiler is not idle")
        if self.budget and self.budget.used_bytes:
            raise RuntimeError("host cache contains stale material")
        if self.budget:
            self.budget.peak_bytes = 0
        self.mode, self.index, self.open_segment = "run", 0, None
        self.online_seconds, self.segment_seconds = 0.0, []
        self.provider_seconds, self.ring_seconds = 0.0, 0.0
        self.online_crypto, self.provider_crypto = _zero_stats(), _zero_stats()
        self.online_ring, self.offline_ring = _zero_stats(), _zero_stats()

    def measure_event(self, event, name):
        if self.mode != "run" or self.index >= len(self.segments):
            raise RuntimeError("unexpected segment event")
        expected, p0, p1, r0, r1 = self.segments[self.index]
        if name != expected:
            raise RuntimeError(f"segment order changed: expected {expected}, got {name}")
        if event == "start":
            if self.open_segment is not None:
                raise RuntimeError("segment already open")
            if dist.get_rank() == 0:
                print(json.dumps({"event": "segment_preprocessing_started", "segment": name,
                                  "provider_requests": p1 - p0, "ring_requests": r1 - r0}), flush=True)
            _barrier(self.device)
            self.crypten.reset_communication_stats()
            self.sync()
            start = time.perf_counter()
            self.provider.prepare(p0, p1)
            self.sync()
            self.provider_seconds += time.perf_counter() - start
            _add_stats(self.provider_crypto, self.crypten.communicator.get().get_communication_stats())
            if self.ring:
                sent, phases = self.base.sent_bytes, self.base.phases
                self.sync()
                start = time.perf_counter()
                self.ring.prepare(r0, r1)
                self.sync()
                self.ring_seconds += time.perf_counter() - start
                self.offline_ring["bytes"] += self.base.sent_bytes - sent
                self.offline_ring["rounds"] += self.base.phases - phases
            self.provider.begin_replay()
            if self.ring:
                self.ring.begin_replay()
            # Preparation and inter-segment alignment are excluded explicitly.
            _barrier(self.device)
            self.crypten.reset_communication_stats()
            self.ring_sent = self.base.sent_bytes if self.base else 0
            self.ring_phases = self.base.phases if self.base else 0
            self.sync()
            self.started = time.perf_counter()
            self.open_segment = name
        elif event == "end":
            if self.open_segment != name:
                raise RuntimeError("segment ended without matching start")
            self.sync()
            elapsed = time.perf_counter() - self.started
            self.online_seconds += elapsed
            self.segment_seconds.append(elapsed)
            _add_stats(self.online_crypto, self.crypten.communicator.get().get_communication_stats())
            if self.base:
                self.online_ring["bytes"] += self.base.sent_bytes - self.ring_sent
                self.online_ring["rounds"] += self.base.phases - self.ring_phases
            self.provider.end_replay()
            if self.ring:
                self.ring.end_replay()
            if self.budget and self.budget.used_bytes:
                raise RuntimeError("segment left unconsumed host material")
            if dist.get_rank() == 0:
                print(json.dumps({"event": "segment_online_complete", "segment": name,
                                  "seconds_rank0": elapsed}), flush=True)
            self.open_segment = None
            self.index += 1
        else:
            raise ValueError("unknown segment event")

    def finish_run(self):
        if self.mode != "run" or self.open_segment is not None or self.index != len(self.segments):
            raise RuntimeError("segmented run incomplete")
        self.mode = "idle"
        local = torch.tensor(self.segment_seconds, dtype=torch.float64)
        gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, local)
        values = torch.stack(gathered)
        return {
            "online_crypten": _aggregate(self.device, self.online_seconds, self.online_crypto,
                                         bidirectional_counter=True),
            "online_ring": _aggregate(self.device, self.online_seconds, self.online_ring),
            "crypten_preprocessing": _aggregate(self.device, self.provider_seconds, self.provider_crypto,
                                                bidirectional_counter=True),
            "ring_preprocessing": _aggregate(self.device, self.ring_seconds, self.offline_ring),
            "segments": [{"name": spec[0], "seconds_max": float(values[:, i].max())}
                         for i, spec in enumerate(self.segments)],
            "sum_segment_max_seconds": float(values.max(dim=0).values.sum()),
            "note": "online_crypten/online_ring seconds_max is the maximum party sum; sum_segment_max_seconds sums the slowest party per segment; both are estimates with extra synchronization and offline preparation between segments, not uninterrupted wall time",
        }
