"""Optional rank-0 wall-clock diagnostics; never logs secret values."""
from contextlib import contextmanager
import json
import threading
import time
import torch


class ProgressTrace:
    def __init__(self, enabled, device, heartbeat_seconds=30, sink=None):
        self.enabled = enabled
        self.device = device
        self.context = {}
        self.totals = {'ring3_prepare': 0., 'range_check': 0.}
        self.blocks = {}
        self.location = 'forward'
        self.heartbeat_seconds = heartbeat_seconds
        self.sink = sink

    def sync(self):
        if self.enabled and self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def event(self, event, **fields):
        if self.enabled:
            record = {'event': event, **self.context, **fields}
            if self.sink:
                self.sink(record)
            else:
                print(json.dumps(record), flush=True)

    @contextmanager
    def heartbeat(self):
        if not self.enabled:
            yield
            return
        stop = threading.Event()
        started = time.perf_counter()
        def run():
            while not stop.wait(self.heartbeat_seconds):
                self.event('forward_heartbeat', location=self.location,
                           elapsed_seconds=time.perf_counter()-started)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()

    @contextmanager
    def span(self, category, site):
        if not self.enabled:
            yield
            return
        self.sync()
        start = time.perf_counter()
        previous = self.location
        if category == 'ring3_prepare':
            self.location = 'prepare:'+site
            self.event('preparation_started', site=site)
        success = False
        try:
            yield
            success = True
        except Exception:
            self.event('stage_failed', category=category, site=site)
            raise
        finally:
            self.sync()
            elapsed = time.perf_counter() - start
            self.totals[category] += elapsed
            self.location = previous
            if category == 'ring3_prepare' and success:
                self.event('preparation_complete', site=site, seconds_rank0=elapsed)

    def block_start(self, layer):
        if self.enabled:
            self.sync()
            self.blocks[layer] = (time.perf_counter(), self.totals.copy())
            self.location = 'block:'+str(layer)
            self.event('block_started', layer=layer)

    def block_end(self, layer):
        if self.enabled:
            self.sync()
            start, before = self.blocks.pop(layer)
            elapsed = time.perf_counter() - start
            prep = self.totals['ring3_prepare'] - before['ring3_prepare']
            checks = self.totals['range_check'] - before['range_check']
            self.event('block_complete', layer=layer, seconds_rank0=elapsed,
                       ring3_prepare_seconds_rank0=prep, range_check_seconds_rank0=checks,
                       remaining_seconds_rank0=max(0., elapsed - prep - checks))
            self.location = 'forward'
