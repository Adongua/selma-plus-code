"""Check strict offline replay refuses altered requests and reused material."""
import unittest

import torch

from hattention.offline_materials import StrictProviderCache, StrictRingCache


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


if __name__ == "__main__":
    unittest.main()
