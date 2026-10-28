import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hattention import hybrid
from hattention.benchmark_compare_cuda import _summary, standard_attention
from hattention.reference import mixer_attention


class HybridTests(unittest.TestCase):
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
                  "records": [{"warmup": True, "methods": {"standard": result}},
                              {"warmup": False, "methods": {"standard": result}}]}
        self.assertIn("INVALID", _summary(report))


if __name__ == "__main__":
    unittest.main()
