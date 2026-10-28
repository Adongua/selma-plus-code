"""CPU-only report validation for the mixed-head benchmark."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# Load the runner without importing hattention.__init__, which requires torch.
package = types.ModuleType("_hattention_runner_test")
package.__path__ = [str(ROOT / "hattention")]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location(
    package.__name__ + ".run_hybrid_microbenchmark", ROOT / "hattention" / "run_hybrid_microbenchmark.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class HybridReportTests(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(backend="gloo", repeats=2, seed=7,
                                       crypten_precision_bits=20, max_abs_error=0.005)
        self.shape = (1, 4, 17, 16)
        self.name = "rsa1_lsa1_mixer2"
        paths = list(runner.CONFIGS[self.name][1])
        def row(warmup, seconds, passed=True):
            return {"warmup": warmup, "methods": {"hybrid": {
                "correctness_passed": passed,
                "per_head_error_vs_own_reference": [{"head": i, "path": path} for i, path in enumerate(paths)],
                "error_vs_own_reference": {"max_abs": 0.001},
                "online": {"seconds_max": seconds, "system_bytes": 2048,
                           "rounds_per_rank": [4, 5, 4]}}}}
        self.report = {"backend": "gloo", "world_size": 3, "shape_b_h_n_d": list(self.shape),
                       "methods": ["hybrid"], "repeats": 2, "seed": 7,
                       "input_precision_bits": 17, "ring3_precision_bits": 17,
                       "crypten_precision_bits": 20, "max_abs_error_threshold": 0.005,
                       "input_std": 0.25, "has_cls_token": True,
                       "head_types": {"hybrid": paths},
                       "records": [row(True, 100), row(False, 2), row(False, 4)]}

    def collect(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.json"
            path.write_text(json.dumps(self.report), encoding="utf-8")
            return runner.collect(path, self.shape, self.name, self.args)

    def test_warmup_excluded_and_failed_run_invalid(self):
        self.assertEqual(self.collect(), (3.0, 2.0, 5, 0.001))
        self.report["records"][2]["methods"]["hybrid"]["correctness_passed"] = False
        self.assertIsNone(self.collect())

    def test_wrong_head_order_or_missing_diagnostics_rejected(self):
        self.report["head_types"]["hybrid"][0] = "mixer"
        with self.assertRaises(ValueError):
            self.collect()
        self.report["head_types"]["hybrid"][0] = "rsa"
        self.report["records"][1]["methods"]["hybrid"]["per_head_error_vs_own_reference"].pop()
        with self.assertRaises(ValueError):
            self.collect()

    def test_summary_identifies_counter_and_reference(self):
        values = {(self.shape, name): (0.1, 2, 3, 0.001) for name in runner.CONFIGS}
        values[self.shape, "rsa4"] = None
        text = runner.render([self.shape], values, self.args)
        self.assertIn("INVALID", text)
        self.assertIn("Rounds / phases", text)
        self.assertIn("own plaintext formula", text)


if __name__ == "__main__":
    unittest.main()
