"""CPU-only checks for microbenchmark report validation and table rendering."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("hattention_microbenchmark_runner", ROOT / "hattention" / "run_microbenchmark.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class ReportTests(unittest.TestCase):
    def test_default_shapes_are_conservative_and_mixer_compatible(self):
        self.assertEqual(runner.DEFAULT_SHAPES,
            ("1x1x17x32", "1x1x65x32", "1x1x101x32", "1x1x197x32"))
        self.assertTrue(all(runner.parse_shape(shape)[0] == 1 for shape in runner.DEFAULT_SHAPES))

    def setUp(self):
        self.args = argparse.Namespace(backend="gloo", repeats=2, seed=7,
                                       crypten_precision_bits=20, max_abs_error=0.005)
        self.shape = (1, 1, 17, 16)
        self.report = {"backend": "gloo", "world_size": 3, "shape_b_h_n_d": list(self.shape),
                       "methods": ["standard"], "repeats": 2, "seed": 7,
                       "input_precision_bits": 17, "crypten_precision_bits": 20,
                       "ring3_precision_bits": 17, "max_abs_error_threshold": 0.005,
                       "input_std": 0.25, "has_cls_token": True,
                       "records": [self.row(True, 100, True), self.row(False, 2, True),
                                   self.row(False, 4, True)]}

    @staticmethod
    def row(warmup, seconds, passed):
        return {"warmup": warmup, "methods": {"standard": {"correctness_passed": passed,
                "online": {"seconds_max": seconds, "system_bytes": 2048,
                           "rounds_per_rank": [3, 4, 3]}}}}

    def read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.json"
            path.write_text(json.dumps(self.report), encoding="utf-8")
            return runner.collect_report(path, self.shape, "standard", self.args)

    def test_mean_excludes_warmup_and_checks_each_repeat(self):
        self.assertEqual(self.read(), (3.0, 2.0, 4))
        self.report["records"][2]["methods"]["standard"]["correctness_passed"] = False
        self.assertIsNone(self.read())

    def test_rejects_mixed_precision_and_missing_repeat(self):
        self.report["crypten_precision_bits"] = 17
        with self.assertRaises(ValueError):
            self.read()
        self.report["crypten_precision_bits"] = 20
        self.report["records"].pop()
        with self.assertRaises(ValueError):
            self.read()

    def test_tables_mark_invalid_and_omit_mae(self):
        values = {(self.shape, method): (0.25, 3.0, 4.0) for method in runner.METHODS}
        values[self.shape, "rsa"] = None
        markdown = runner.render([self.shape], values, self.args)
        latex = runner.render_latex([self.shape], values, self.args)
        self.assertIn("| rsa | INVALID | INVALID | INVALID |", markdown)
        self.assertNotIn("MAE", markdown)
        self.assertIn(r"\begin{table*}", latex)
        self.assertIn("RSA & INVALID & INVALID & INVALID", latex)
        self.assertIn("CrypTen standard (Q20)", latex)


if __name__ == "__main__":
    unittest.main()
