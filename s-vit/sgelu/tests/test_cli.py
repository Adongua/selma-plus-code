import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


class CalibrationCliTests(unittest.TestCase):
    def test_public_npz_roundtrip(self):
        repository = Path(__file__).resolve().parents[3]
        output_root = repository.parent / "tmp"
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="gelu-cli-", dir=output_root) as directory:
            source, result = Path(directory, "public.npz"), Path(directory, "calibration.json")
            # Synthetic fixtures only; this is not model calibration.
            np.savez(source, block0=np.linspace(-4, 4, 257), block1=np.zeros(16))
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["PYTHONPATH"] = str(repository / "s-vit") + os.pathsep + env.get("PYTHONPATH", "")
            env["OMP_NUM_THREADS"] = "1"
            completed = subprocess.run(
                [sys.executable, "-m", "sgelu.calibrate", str(source), "--output", str(result)],
                env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(result.read_text())
            self.assertEqual(report["formula"], "G2-v2")
            self.assertEqual(set(report["layers"]), {"block0", "block1"})
            for layer in report["layers"].values():
                self.assertLessEqual(layer["fitted_mse"], layer["initial_mse"])
            self.assertEqual(report["layers"]["block0"]["sample_count"], 257)
