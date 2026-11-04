"""Check strict offline replay refuses altered requests and reused material."""
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hattention.offline_materials import HostCacheBudget, StrictProviderCache, StrictRingCache
from hattention.segmented_online import SegmentedOnline


class FakeProvider:
    TRACEABLE_FUNCTIONS = list(StrictProviderCache.METHODS)
    tracing = False
    tuple_cache = {}

    def __init__(self):
        self.counter = 0

    def square(self, shape, device=None):
        self.counter += 1
        return self.counter


class FakeMPC:
    def __init__(self):
        self.provider = FakeProvider()

    def get_default_provider(self):
        return self.provider


class FakeCrypten:
    def __init__(self):
        self.mpc = FakeMPC()


@dataclass
class FakeRingMaterial:
    runtime: object
    values: tuple
    used: bool = False


class OfflineMaterialTests(unittest.TestCase):
    def test_provider_sequence_freshness_and_tensor_rejection(self):
        crypten = FakeCrypten()
        original = crypten.mpc.get_default_provider
        cache = StrictProviderCache(crypten)
        cache.install()
        try:
            cache.begin_trace()
            self.assertEqual(crypten.mpc.get_default_provider().square((3,)), 1)
            with self.assertRaises(TypeError):
                crypten.mpc.get_default_provider().square(torch.ones(2))
            cache.end_trace()
            cache.prepare()
            cache.begin_replay()
            self.assertEqual(crypten.mpc.get_default_provider().square((3,)), 2)
            with self.assertRaises(RuntimeError):
                crypten.mpc.get_default_provider().square((3,))
            cache.end_replay()
            with self.assertRaises(RuntimeError):
                cache.begin_replay()
            cache.prepare()
            cache.begin_replay()
            with self.assertRaisesRegex(RuntimeError, "request changed"):
                crypten.mpc.get_default_provider().square((4,))
        finally:
            cache.close()
        self.assertEqual(crypten.mpc.get_default_provider, original)

    def test_ring_material_is_fresh_and_must_match(self):
        counter = iter(range(10))
        cache = StrictRingCache()
        cache.begin_trace()
        self.assertEqual(cache.request(("norm", (2,)), lambda: next(counter)), 0)
        cache.end_trace()
        cache.prepare()
        cache.begin_replay()
        self.assertEqual(cache.request(("norm", (2,)), lambda: -1), 1)
        cache.end_replay()
        cache.prepare()
        cache.begin_replay()
        with self.assertRaisesRegex(RuntimeError, "request changed"):
            cache.request(("norm", (3,)), lambda: -1)

    def test_host_storage_keeps_runtime_identity_and_enforces_budget(self):
        runtime = object()
        budget = HostCacheBudget(64)
        cache = StrictRingCache("cpu", torch.device("cpu"), budget)
        cache.begin_trace()
        cache.request(("norm", (4,)), lambda: FakeRingMaterial(runtime, (torch.ones(4, dtype=torch.long),)))
        cache.end_trace()
        cache.prepare()
        self.assertEqual(budget.used_bytes, 32)
        cache.begin_replay()
        material = cache.request(("norm", (4,)), lambda: None)
        cache.end_replay()
        self.assertIs(material.runtime, runtime)
        self.assertTrue(torch.equal(material.values[0], torch.ones(4, dtype=torch.long)))
        self.assertEqual(budget.used_bytes, 0)
        too_small = StrictRingCache("cpu", torch.device("cpu"), HostCacheBudget(16))
        too_small.begin_trace()
        too_small.request(("norm", (4,)), lambda: FakeRingMaterial(runtime, (torch.ones(4, dtype=torch.long),)))
        too_small.end_trace()
        with self.assertRaisesRegex(RuntimeError, "exceeds the per-party budget"):
            too_small.prepare()

    def test_segmented_cache_fits_when_whole_model_does_not(self):
        cache = StrictRingCache("cpu", torch.device("cpu"), HostCacheBudget(48))
        cache.begin_trace()
        for layer in range(2):
            cache.request(("layer", layer), lambda: torch.ones(4, dtype=torch.long))
        cache.end_trace()
        with self.assertRaisesRegex(RuntimeError, "exceeds the per-party budget"):
            cache.prepare()
        # Failed runs discard the entire cache; simulate a fresh process.
        cache = StrictRingCache("cpu", torch.device("cpu"), HostCacheBudget(48))
        cache.begin_trace()
        for layer in range(2):
            cache.request(("layer", layer), lambda: torch.ones(4, dtype=torch.long))
        cache.end_trace()
        for layer in range(2):
            cache.prepare(layer, layer + 1)
            cache.begin_replay()
            actual = cache.request(("layer", layer), lambda: None)
            cache.end_replay()
            self.assertEqual(cache.budget.used_bytes, 0)
            self.assertEqual(actual.numel(), 4)

    def test_segment_timer_excludes_material_generation(self):
        crypten = FakeCrypten()
        crypten.reset_communication_stats = lambda: None
        crypten.communicator = SimpleNamespace(get=lambda: SimpleNamespace(
            get_communication_stats=lambda: {"bytes": 0, "rounds": 0, "time": 0.0}))
        cache = StrictProviderCache(crypten)
        cache.install()
        try:
            profiler = SegmentedOnline(crypten, torch.device("cpu"), cache)
            cache.begin_trace()
            profiler.trace_event("start", "block_1")
            crypten.mpc.get_default_provider().square((3,))
            profiler.trace_event("end", "block_1")
            cache.end_trace()
            profiler.end_trace()
            profiler.begin_run()
            with patch("hattention.segmented_online._barrier"), \
                    patch("hattention.segmented_online.dist.get_rank", return_value=0), \
                    patch("hattention.segmented_online.time.perf_counter", side_effect=[0, 5, 10, 12]):
                profiler.measure_event("start", "block_1")
                crypten.mpc.get_default_provider().square((3,))
                profiler.measure_event("end", "block_1")
            self.assertEqual(profiler.provider_seconds, 5)
            self.assertEqual(profiler.online_seconds, 2)
        finally:
            cache.close()


if __name__ == "__main__":
    unittest.main()
