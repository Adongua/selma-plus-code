import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hattention import hybrid
from hattention.benchmark_compare_cuda import HybridProfiler, collect_hybrid_profile, _summary, standard_attention
from hattention.reference import mixer_attention


class HybridTests(unittest.TestCase):
    def test_profile_preserves_output_and_records_each_present_group(self):
        base = SimpleNamespace(device=torch.device("cpu"), sent_bytes=0, phases=0)
        wide = SimpleNamespace(base=base)
        shape = (1, 3, 17, 2)
        paths = ("mixer", "rsa", "lsa")
        value = torch.arange(102, dtype=torch.int64).reshape(shape)
        profiler = HybridProfiler(base.device, base)
        with patch.object(hybrid, "prepare_rsa", return_value="r"), \
             patch.object(hybrid, "prepare_lsa", return_value="l"), \
             patch.object(hybrid, "secure_rsa", side_effect=lambda b, w, q, k, v, m: v + 10), \
             patch.object(hybrid, "secure_lsa", side_effect=lambda b, q, k, v, m: v + 20), \
             patch("hattention.benchmark_compare_cuda.torch.cuda.synchronize"):
            normal = hybrid.secure_hybrid(base, wide, value, value, value,
                hybrid.prepare_hybrid(base, wide, shape, paths), patch_grid=(4, 4))
            profiled = hybrid.secure_hybrid(base, wide, value, value, value,
                hybrid.prepare_hybrid(base, wide, shape, paths), patch_grid=(4, 4), profiler=profiler)
        self.assertTrue(torch.equal(normal, profiled))
        self.assertEqual(list(profiler.stages), [f"{path}_{stage}" for path in ("rsa", "lsa", "mixer")
                                                  for stage in ("select", "compute", "scatter")])
        self.assertTrue(all(stage["seconds"] >= 0 for stage in profiler.stages.values()))

    def test_profile_aggregates_rank_times_without_summing_maxima(self):
        local = {"rsa_compute": {"seconds": 1.0, "sent_bytes": 8, "phases": 1}}
        def gather(out, _):
            out[:] = [local, {"rsa_compute": {"seconds": 2.0, "sent_bytes": 9, "phases": 2}}]
        with patch("hattention.benchmark_compare_cuda.dist.get_world_size", return_value=2), \
             patch("hattention.benchmark_compare_cuda.dist.all_gather_object", side_effect=gather):
            result = collect_hybrid_profile(local)["rsa_compute"]
        self.assertEqual(result["seconds_per_rank"], [1.0, 2.0])
        self.assertEqual(result["seconds_max"], 2.0)
        self.assertEqual(result["sent_bytes_per_rank"], [8, 9])

    def test_grouping_restores_order_and_material_is_one_use(self):
        base = SimpleNamespace(device=torch.device("cpu"))
        wide = SimpleNamespace(base=base)
        shape = (1, 5, 17, 2)
        paths = ("mixer", "rsa", "lsa", "rsa", "mixer")
        q = torch.arange(170, dtype=torch.int64).reshape(shape)
        k, v = q + 1000, q + 2000
        with patch.object(hybrid, "prepare_rsa", return_value="r") as prep_rsa, \
             patch.object(hybrid, "prepare_lsa", return_value="l") as prep_lsa, \
             patch.object(hybrid, "secure_rsa", side_effect=lambda b, w, q, k, v, m: v + 10), \
             patch.object(hybrid, "secure_lsa", side_effect=lambda b, q, k, v, m: v + 20):
            material = hybrid.prepare_hybrid(base, wide, shape, paths)
            actual = hybrid.secure_hybrid(base, wide, q, k, v, material, patch_grid=(4, 4))
            prep_rsa.assert_called_once()
            prep_lsa.assert_called_once()
            self.assertEqual(prep_rsa.call_args.args[2], (1, 2, 17, 2))
            self.assertEqual(prep_lsa.call_args.args[1], (1, 1, 17, 2))
            for i, path in enumerate(paths):
                expected = v[:, i:i+1]
                if path == "rsa":
                    expected = expected + 10
                elif path == "lsa":
                    expected = expected + 20
                else:
                    expected = mixer_attention(expected, (4, 4))
                self.assertTrue(torch.equal(actual[:, i:i+1], expected))
            with self.assertRaises(RuntimeError):
                hybrid.secure_hybrid(base, wide, q, k, v, material, patch_grid=(4, 4))

    def test_standard_attention_and_summary_correctness_gate(self):
        torch.manual_seed(4)
        q, k, v = [torch.randn(1, 2, 3, 4, dtype=torch.float64) for _ in range(3)]
        expected = torch.softmax(q @ k.transpose(-2, -1) / 2, dim=-1) @ v
        self.assertTrue(torch.allclose(standard_attention(q, k, v), expected))
        phase = {"seconds_max": 1.0, "system_bytes": 1024, "rounds_per_rank": [2, 2, 2]}
        result = {"online": phase, "offline": phase,
                  "error_vs_own_reference": {"mae": 0.1, "max_abs": 0.2},
                  "correctness_passed": False}
        report = {"methods": ["standard"], "head_types": {"standard": ["softmax"]},
                  "crypten_precision_bits": 20,
                  "records": [{"warmup": True, "methods": {"standard": result}},
                              {"warmup": False, "methods": {"standard": result}}]}
        self.assertIn("INVALID", _summary(report))


if __name__ == "__main__":
    unittest.main()
