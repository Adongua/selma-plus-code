import json
from pathlib import Path
import tempfile
import unittest

from hattention.summarize_vit_forward import summarize


class ForwardSummaryTests(unittest.TestCase):
    def test_optimized_and_native_timing_scopes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            phase = lambda seconds, size: {"seconds_max": seconds, "system_bytes": size,
                                           "sent_bytes_sum": size}
            error = {"finite": True, "max_abs": 1.2}
            base = {"backend": "gloo", "world_size": 3, "image_sha256": "abc",
                    "precision": "Q17", "status": "benchmark_failed_correctness"}
            row = {"warmup": False, "forward_wall": phase(270, 300),
                   "ring_preprocessing": phase(260, 250),
                   "online_ring_excluding_preprocessing": phase(10, 50),
                   "online_crypten": phase(0, 8), "error_vs_plaintext_logits": error}
            path.write_text(json.dumps({**base, "records": [row]}))
            result = summarize("optimized", path)
            self.assertFalse(result["valid"])
            self.assertEqual(result["forward_excluding_ring_preprocessing_seconds"], 10)
            self.assertEqual(result["ring_preprocessing_seconds"], 260)
            self.assertEqual(result["max_logit_error"], 1.2)
            del row["ring_preprocessing"]
            del row["online_ring_excluding_preprocessing"]
            row["online_crypten"] = phase(270, 8)
            path.write_text(json.dumps({**base, "records": [row]}))
            result = summarize("native", path)
            self.assertIsNone(result["ring_preprocessing_seconds"])
            self.assertEqual(result["forward_excluding_ring_preprocessing_seconds"], 270)


if __name__ == "__main__":
    unittest.main()
