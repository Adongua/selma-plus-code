"""Optional synchronized operator timing for continuous full-ViT diagnostics."""
import time

import torch
import torch.distributed as dist

from .inline_preprocessing import local_accounting


class OperatorProfiler:
    """Measure non-overlapping calls; collectives are delayed until forward ends."""
    NAMES = ("conv", "cat_cls", "linear", "norm", "gelu", "attention")

    def __init__(self, operators, provider_timer, device, sync=None, clock=None):
        self.operators, self.provider_timer = operators, provider_timer
        self.sync = sync or (lambda: torch.cuda.synchronize(device))
        self.clock = clock or time.perf_counter
        self.entries = []

    def __getattr__(self, name):
        value = getattr(self.operators, name)
        if name not in self.NAMES:
            return value

        def measured(*args, **kwargs):
            self.sync()
            provider_before = self.provider_timer.seconds
            ring_before = getattr(self.operators, 'offline_seconds', 0.0)
            start = self.clock()
            result = value(*args, **kwargs)
            self.sync()
            wall = self.clock() - start
            provider = self.provider_timer.seconds - provider_before
            ring = getattr(self.operators, 'offline_seconds', 0.0) - ring_before
            _, _, _, online = local_accounting(wall, provider, ring)
            shapes = [list(x.shape) for x in args if hasattr(x, 'shape')]
            self.entries.append({'operator': name, 'input_shapes': shapes,
                                 'wall_seconds': wall, 'provider_seconds': provider,
                                 'ring_preprocessing_seconds': ring,
                                 'online_estimate_seconds': online})
            return result
        return measured

    def finish(self, accounting):
        per_rank = [None] * dist.get_world_size()
        dist.all_gather_object(per_rank, self.entries)
        signatures = [[(e['operator'], e['input_shapes']) for e in rank] for rank in per_rank]
        if any(s != signatures[0] for s in signatures[1:]):
            raise RuntimeError('operator call sequences or shapes differ across parties')
        totals = []
        for rank, entries in enumerate(per_rank):
            groups = {}
            for row in entries:
                group = groups.setdefault(row['operator'], {'calls': 0, 'wall_seconds': 0.0,
                    'provider_seconds': 0.0, 'ring_preprocessing_seconds': 0.0,
                    'online_estimate_seconds': 0.0})
                group['calls'] += 1
                for key in ('wall_seconds', 'provider_seconds', 'ring_preprocessing_seconds',
                            'online_estimate_seconds'):
                    group[key] += row[key]
            overall = accounting['per_rank'][rank]['online_estimate_seconds']
            totals.append({'operators': groups, 'online_estimate_seconds': overall,
                           'unattributed_online_seconds': overall - sum(g['online_estimate_seconds'] for g in groups.values())})
        slowest = max(range(len(totals)), key=lambda i: totals[i]['online_estimate_seconds'])
        return {'per_rank': totals, 'calls_per_rank': per_rank, 'slowest_online_rank': slowest,
                'note': 'Synchronized per-operator diagnostics, not uninstrumented latency. Each time excludes timed provider and Ring3 preparation on that rank; residual includes adds, reshapes, synchronization and profiling overhead. Group times shown on one slowest overall rank are additive; separate rank maxima are not.'}
