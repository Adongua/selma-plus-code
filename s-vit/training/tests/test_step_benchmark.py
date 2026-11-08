import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from hattention.offline_materials import StrictProviderCache
from training.step_benchmark import StepBenchmark, aggregate_stages
from training.summarize_step_benchmark import summarize


def stage(online, crypto, crypto_offline, ring, ring_offline):
    return {'online_estimate_seconds': online, 'crypten_total_bytes': crypto,
            'crypten_preprocessing_bytes': crypto_offline, 'ring_total_bytes': ring,
            'ring_preprocessing_bytes': ring_offline}


class StepBenchmarkTests(unittest.TestCase):
    def test_nested_provider_is_counted_only_once(self):
        stats = {'bytes': 0}
        class Provider:
            TRACEABLE_FUNCTIONS = StrictProviderCache.METHODS
            tracing = False
            tuple_cache = {}
            def square(self):
                stats['bytes'] += 20
        provider = Provider()
        crypten = SimpleNamespace(mpc=SimpleNamespace(get_default_provider=lambda: provider),
                                  communicator=SimpleNamespace(get=lambda: SimpleNamespace(
                                      get_communication_stats=lambda: stats)))
        base = SimpleNamespace(sent_bytes=0)
        meter = StepBenchmark(crypten, torch.device('cpu'), base)
        with patch('training.step_benchmark.time.perf_counter', side_effect=[0, 1, 2, 3, 5, 10]):
            snapshot = meter.start()
            with meter.preparation():
                meter.square()
                stats['bytes'] += 4
                base.sent_bytes += 7
            result = meter.finish(snapshot)
        self.assertEqual(result['provider_seconds'], 1)
        self.assertEqual(result['ring_prepare_seconds'], 3)
        self.assertEqual(result['online_estimate_seconds'], 6)
        self.assertEqual(result['crypten_preprocessing_bytes'], 24)
        self.assertEqual(result['ring_preprocessing_bytes'], 7)

    def test_rankwise_total_and_offline_communication_subtraction(self):
        rows = [
            {'forward': stage(5, 20, 4, 12, 2), 'optimizer': stage(1, 8, 2, 0, 0)},
            {'forward': stage(2, 20, 4, 12, 2), 'optimizer': stage(5, 8, 2, 0, 0)},
            {'forward': stage(3, 20, 4, 12, 2), 'optimizer': stage(2, 8, 2, 0, 0)},
        ]
        def gather(output, _local):
            output[:] = rows
        with patch('training.step_benchmark.dist.get_world_size', return_value=3), \
             patch('training.step_benchmark.dist.all_gather_object', side_effect=gather):
            result = aggregate_stages(rows[0])
        self.assertEqual(result['forward']['online_estimate_seconds'], 5)
        self.assertEqual(result['optimizer']['online_estimate_seconds'], 5)
        self.assertEqual(result['total']['online_estimate_seconds'], 7)
        self.assertEqual(result['forward']['online_communication_bytes'], 54)
        self.assertEqual(result['optimizer']['online_communication_bytes'], 9)
        self.assertEqual(result['total']['online_communication_bytes'], 63)
        self.assertEqual(result['backward']['online_communication_bytes'], 0)

    def test_summary_excludes_warmup(self):
        def record(value):
            metrics = {'online_estimate_seconds': value, 'online_communication_bytes': value * 10}
            return {'benchmark': {name: metrics for name in ('forward', 'backward', 'optimizer', 'total')}}
        result = summarize({'steps': [record(100), record(2), record(4)]}, 1)
        self.assertEqual(result['forward']['online_estimate_seconds'], 3)
        self.assertEqual(result['optimizer']['online_communication_bytes'], 30)
