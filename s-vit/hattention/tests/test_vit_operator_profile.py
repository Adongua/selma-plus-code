"""Check full-forward profiling and slowest-party accounting."""
import os
from pathlib import Path
import tempfile
import time
import unittest

import torch
import torch.multiprocessing as mp

from hattention.vit_operator_profile import OperatorProfiler
from hattention.summarize_vit_operators import summarize
from hattention.inline_preprocessing import InlineProviderTimer, gather_accounting

ROOT = Path(__file__).resolve().parents[3]


def _worker(rank, rendezvous):
    os.environ.update(RANK=str(rank), WORLD_SIZE='3', RENDEZVOUS=rendezvous, DISTRIBUTED_BACKEND='gloo')
    import crypten
    from hattention.student_checkpoint import import_mpcvit
    from hattention.vit_executor import forward_vit
    from hattention.vit_integration import make_fixed_student
    from hattention.config import HAttentionConfig
    from hattention.benchmark_vit_native_cuda import CrypTenNativeOperators, PlainStandardOperators
    crypten.cfg.encoder.precision_bits = 20
    crypten.init(device=torch.device('cpu'))
    try:
        import_mpcvit(ROOT / 'mpcvit')
        from src.vit import ViTLite
        torch.manual_seed(25)
        source = ViTLite(img_size=8, kernel_size=4, embedding_dim=12, num_layers=1,
                         num_heads=3, mlp_ratio=2, num_classes=3, dropout=0,
                         attention_dropout=0, stochastic_depth=0)
        model = make_fixed_student(source, [HAttentionConfig(3, ('rsa', 'lsa', 'mixer'), patch_grid=(2, 2))]).eval()
        image = torch.randn(1, 3, 8, 8) * 0.1
        plain = forward_vit(model, image, PlainStandardOperators())
        secret = crypten.cryptensor(image, src=0, precision=20)
        timer = InlineProviderTimer(crypten, torch.device('cpu'), sync=lambda: None)
        profiler = OperatorProfiler(CrypTenNativeOperators(), timer, torch.device('cpu'), sync=lambda: None)
        start = time.perf_counter()
        with timer.installed():
            output = forward_vit(model, secret, profiler)
        wall = time.perf_counter() - start
        accounting = gather_accounting(wall, timer.seconds, 0.0, timer.calls)
        profile = profiler.finish(accounting)
        groups = profile['per_rank'][rank]['operators']
        assert {key: group['calls'] for key, group in groups.items()} == {
            'conv': 1, 'cat_cls': 1, 'linear': 5, 'norm': 3, 'gelu': 1, 'attention': 1}
        assert len(profile['calls_per_rank']) == 3 and timer.calls > 0
        assert torch.allclose(output.get_plain_text(), plain, atol=1.0, rtol=0)
        local = profile['per_rank'][rank]
        assert abs(sum(group['online_estimate_seconds'] for group in groups.values()) +
                   local['unattributed_online_seconds'] - local['online_estimate_seconds']) < 1e-8
    finally:
        crypten.uninit()


class ProfilingTest(unittest.TestCase):
    def test_full_vit_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_worker, args=(Path(directory, 'store').as_uri(),), nprocs=3, join=True)

    def test_subtracts_both_preprocessing_intervals(self):
        class Timer:
            seconds = 0.0
        class Ops:
            offline_seconds = 0.0
            def attention(self, value):
                timer.seconds += 2.0
                self.offline_seconds += 3.0
                return value
        timer = Timer()
        moments = iter((0.0, 10.0))
        profiler = OperatorProfiler(Ops(), timer, None, sync=lambda: None, clock=lambda: next(moments))
        self.assertEqual(profiler.attention(torch.zeros(1)).shape, (1,))
        row = profiler.entries[0]
        self.assertEqual(row['wall_seconds'], 10.0)
        self.assertEqual(row['provider_seconds'], 2.0)
        self.assertEqual(row['ring_preprocessing_seconds'], 3.0)
        self.assertEqual(row['online_estimate_seconds'], 5.0)

    def test_summary_uses_single_slowest_party(self):
        import json
        ranks = [
            {'online_estimate_seconds': 10.0, 'unattributed_online_seconds': 1.0,
             'operators': {'attention': {'calls': 1, 'online_estimate_seconds': 9.0}}},
            {'online_estimate_seconds': 9.0, 'unattributed_online_seconds': 8.0,
             'operators': {'attention': {'calls': 1, 'online_estimate_seconds': 1.0}}},
        ]
        calls = [[{'operator': 'attention', 'input_shapes': [[1, 3, 5, 4]]}]] * 2
        report = {'preprocessing_mode': 'inline_instrumented', 'operator_profiling': True,
                  'status': 'benchmark_passed_not_security_validated', 'backend': 'gloo',
                  'world_size': 2, 'image_sha256': 'example', 'precision': 'Q20',
                  'records': [{'warmup': False, 'operator_profile': {'per_rank': ranks,
                      'slowest_online_rank': 0, 'calls_per_rank': calls}}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            path.write_text(json.dumps(report))
            result = summarize('baseline', path)
        self.assertEqual(result['groups']['attention']['online_seconds'], 9.0)
        self.assertEqual(result['unattributed_seconds'], 1.0)
        self.assertEqual(result['online_seconds'], 10.0)

    def test_reject_missing_profile(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            path.write_text(json.dumps({'preprocessing_mode': 'inline_instrumented'}))
            with self.assertRaises(ValueError):
                summarize('baseline', path)
