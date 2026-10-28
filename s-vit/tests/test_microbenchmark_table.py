"""Checks table aggregation using synthetic reports, not performance results."""

import json
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import microbenchmark_table as table
import run_gloo_microbenchmarks as runner


class TableTests(unittest.TestCase):
    def setUp(self):
        output_root = Path(__file__).resolve().parents[3] / "tmp"
        output_root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=output_root, prefix="table-test-")
        self.addCleanup(self.tmp.cleanup)

    def report(self, operator, shape, *, distribution="normal"):
        config = table.CONFIG[operator]
        def record(warmup, n):
            methods = {}
            for index, (name, _, _) in enumerate(config["methods"]):
                methods[name] = {
                    "online": {"seconds_max": n + index + 1,
                               "system_bytes": 1024 * (index + 1),
                               "sent_bytes_sum": 2048 * (index + 1),
                               "rounds_per_rank": [index + 1] * 3},
                    config["error"]: {"mae": 0.00001 * (index + 1)},
                }
            row = {"warmup": warmup, "methods": methods}
            if operator == "slayernorm":
                row.update(shape=list(shape), distribution=distribution)
            return row
        report = {"world_size": 3, "backend": "gloo", "crypten_import": "/test/crypten",
                  "crypten_default_precision_bits": 17, "python": "test", "torch": "test",
                  "seed": 123, "repeats": 2, "records": [record(True, 99), record(False, 0),
                                                              record(False, 2)]}
        if operator == "sgelu":
            report.update(shape=list(shape), mu=1.46)
        else:
            report.update(epsilon=1e-5, scale=1.0)
        return report

    def write(self, name, obj):
        path = Path(self.tmp.name) / name
        path.write_text(json.dumps(obj), encoding="utf-8")
        return path

    def test_gelu_means_and_missing_case(self):
        paths = [self.write(f"{i}.json", self.report("sgelu", shape))
                 for i, shape in enumerate(table.CONFIG["sgelu"]["shapes"])]
        values, meta = table.load_measurements(paths, "sgelu", "normal", "system_bytes")
        self.assertEqual(values[((1, 197), "sgelu")], (2.0, 1.0, 1.0, 0.00001))
        output = table.render("sgelu", values, meta, "normal", "system_bytes")
        self.assertIn(r"\begin{table*}", output)
        self.assertIn("SEAF adapted 3PC", output)
        self.assertIn(r"\times 10^{-5}", output)
        with self.assertRaisesRegex(ValueError, "missing required"):
            table.load_measurements(paths[:-1], "sgelu", "normal", "system_bytes")

    def test_layernorm_distribution_and_mixed_backends(self):
        shapes = table.CONFIG["slayernorm"]["shapes"]
        report = self.report("slayernorm", shapes[0])
        report["records"] = []
        for shape in shapes:
            for distribution in ("normal", "uniform"):
                report["records"].extend(self.report("slayernorm", shape, distribution=distribution)["records"])
        path = self.write("layer.json", report)
        values, meta = table.load_measurements([path], "slayernorm", "uniform", "system_bytes")
        self.assertEqual(values[(shapes[0], "crypten_native")][0], 2.0)
        output = table.render("slayernorm", values, meta, "uniform", "system_bytes")
        self.assertIn("Newton engineering baseline", output)
        other = self.report("slayernorm", shapes[0])
        other["backend"] = "nccl"
        with self.assertRaises(ValueError):
            table.load_measurements([path, self.write("mixed.json", other)],
                                    "slayernorm", "normal", "system_bytes")

    def test_batch_runner_uses_all_shapes_and_refuses_overwrite(self):
        def fake_run(command, check):
            self.assertTrue(check)
            self.assertEqual(command[command.index("--backend") + 1], "gloo")
            shape = tuple(int(part) for part in command[command.index("--shape") + 1].split("x"))
            output = Path(command[command.index("--output") + 1])
            output.mkdir(parents=True)
            (output / "comparison.json").write_text(json.dumps(self.report("sgelu", shape)))

        root = Path(self.tmp.name) / "batch"
        with patch.object(runner.subprocess, "run", side_effect=fake_run) as invoke:
            with contextlib.redirect_stdout(io.StringIO()):
                runner.run_operator("sgelu", root, 2, ["normal"])
            self.assertEqual(invoke.call_count, 4)
            self.assertTrue((root / "sgelu/sgelu_microbenchmark_normal.md").exists())
            with self.assertRaisesRegex(RuntimeError, "overwritten"):
                runner.run_operator("sgelu", root, 2, ["normal"])


if __name__ == "__main__":
    unittest.main()
