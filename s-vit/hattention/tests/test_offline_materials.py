"""Check strict offline replay refuses altered requests and reused material."""
import unittest
from dataclasses import dataclass

import torch

from hattention.offline_materials import HostCacheBudget, StrictProviderCache, StrictRingCache


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


if __name__ == "__main__":
    unittest.main()
