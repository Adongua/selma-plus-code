import types
import unittest

from hattention.inline_preprocessing import InlineProviderTimer, local_accounting
from hattention.offline_materials import StrictProviderCache


class Provider:
    TRACEABLE_FUNCTIONS = list(StrictProviderCache.METHODS)
    tracing = False
    tuple_cache = {}

    def square(self, x):
        return x * x


class InlineTimingTest(unittest.TestCase):
    def test_timed_call_and_restore(self):
        provider = Provider()
        mpc = types.SimpleNamespace(get_default_provider=lambda: provider)
        crypten = types.SimpleNamespace(mpc=mpc)
        moments = iter((10.0, 12.5))
        syncs = []
        timer = InlineProviderTimer(crypten, None, sync=lambda: syncs.append(1), clock=lambda: next(moments))
        with timer.installed():
            self.assertEqual(mpc.get_default_provider().square(3), 9)
        self.assertIs(mpc.get_default_provider(), provider)
        self.assertEqual(timer.seconds, 2.5)
        self.assertEqual(timer.calls, 1)
        self.assertEqual(len(syncs), 2)

    def test_per_rank_subtraction_and_overlap_guard(self):
        self.assertEqual(local_accounting(100, 30, 20), [100, 30, 20, 50])
        with self.assertRaises(RuntimeError):
            local_accounting(10, 8, 4)


if __name__ == '__main__':
    unittest.main()


def _cpu_worker(rank, rendezvous):
    import os
    import torch
    import crypten
    from hattention.inline_preprocessing import InlineProviderTimer, gather_accounting
    os.environ.update(RANK=str(rank), WORLD_SIZE='3', RENDEZVOUS=rendezvous, DISTRIBUTED_BACKEND='gloo')
    crypten.init(device=torch.device('cpu'))
    try:
        x = crypten.cryptensor(torch.tensor([2.0]), src=0)
        timer = InlineProviderTimer(crypten, torch.device('cpu'), sync=lambda: None)
        import time
        start = time.perf_counter()
        with timer.installed():
            y = x * x
        wall = time.perf_counter() - start
        result = gather_accounting(wall, timer.seconds, 0, timer.calls)
        assert timer.calls > 0, 'provider timing did not intercept secure multiplication'
        assert result['online_estimate_seconds_max'] >= 0
        assert abs(y.get_plain_text().item() - 4) < 0.01
    finally:
        crypten.uninit()


class ThreePartyInlineTest(unittest.TestCase):
    def test_real_provider(self):
        import tempfile
        from pathlib import Path
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_cpu_worker, args=(Path(directory, 'store').as_uri(),), nprocs=3, join=True)
