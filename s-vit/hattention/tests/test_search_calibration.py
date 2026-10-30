import unittest
from pathlib import Path
from contextlib import redirect_stdout
import io
import json
import tempfile
from unittest.mock import patch

import torch

from hattention.calibrate_search_denominators import collect_denominators, summarize
from hattention import calibrate_search_denominators as calibration
from hattention.config import HAttentionConfig
from hattention.search import make_search_student, search_layers
from test_train_search import Fixture, Student, WideStudent, WideFixture


class CalibrationTests(unittest.TestCase):
    def test_row_statistics_and_history_are_preserved(self):
        torch.manual_seed(1)
        model = make_search_student(Student(), [HAttentionConfig(2, ("rsa", "lsa"), patch_grid=(2, 2))])
        layer = search_layers(model)[0]
        layer.observed_min_denominator.copy_(torch.tensor([0.001, 0.002]))
        before = layer.observed_min_denominator.clone()
        rows = collect_denominators(model, Fixture(), torch.device("cpu"), 2)
        self.assertEqual(rows[0].shape, (2, 20))
        self.assertTrue(torch.equal(before, layer.observed_min_denominator))
        self.assertFalse(hasattr(layer, "denominator_observer"))
        self.assertTrue(model.training)
        summary = summarize(rows[0], 0.01)
        self.assertEqual(summary[0]["rows"], 20)
        self.assertEqual(summary[0]["nonfinite"], 0)
        self.assertAlmostEqual(summary[0]["min"], float(rows[0][0].min()))

    def test_summary_records_nonfinite_values(self):
        rows = summarize(torch.tensor([[0.001, 0.02, float("nan")]]), 0.01)
        self.assertEqual(rows[0]["nonfinite"], 1)
        self.assertAlmostEqual(rows[0]["fraction_below_threshold"], 1 / 3)

    def test_entrypoint_exports_comparison_from_disjoint_subsets(self):
        def make_model(_):
            return make_search_student(WideStudent(), [HAttentionConfig(12, ("lsa",) * 12,
                                                                          patch_grid=(2, 2))])
        model = make_model(None)
        model.search_feature_projection = torch.nn.Linear(192, 6)
        search_layers(model)[0].observed_min_denominator.fill_(1)
        table = torch.zeros(13, 13, dtype=torch.float64)
        items = [(f"{c}/{i}", c) for c in range(2) for i in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, holdout = calibration.split_architecture_items(items, validation_per_class=2, seed=1)
            config = {"epochs": 1, "checkpoint_sha256": "source", "cost_table_sha256": "cost",
                      "seed": 1, "validation_per_class": 2, "training_split": {"seed": 1},
                      "search_holdout": holdout, "budget_online_seconds": 9}
            search_checkpoint = root / "epoch.pt"
            torch.save({"format": "hattention_search_checkpoint_v1", "config": config,
                        "completed_epochs": 1, "model": model.state_dict()}, search_checkpoint)
            argv = ["calibrate", "--search-checkpoint", str(search_checkpoint),
                    "--checkpoint", str(root / "model"), "--cost-table", str(root / "cost"),
                    "--mpcvit-root", str(root), "--dataset-root", str(root),
                    "--output-root", str(root / "output"), "--calibration-samples", "2",
                    "--eval-samples", "2", "--batch-size", "1", "--device", "cpu"]
            with patch("sys.argv", argv), \
                 patch.object(calibration, "file_sha256", side_effect=lambda p: "cost" if Path(p).name == "cost" else "source"), \
                 patch.object(calibration, "training_splits", return_value=([], [], items, {"seed": 1})), \
                 patch.object(calibration, "load_student", return_value=(None, {"sha256": "source"})), \
                 patch.object(calibration, "make_checkpoint_search_student", side_effect=make_model), \
                 patch.object(calibration, "load_complete_tables", return_value=({"online_seconds": table},
                     {"shape_b_h_n_d": [1, 12, 257, 16]})), \
                 patch.object(calibration, "SearchDataset", WideFixture), redirect_stdout(io.StringIO()):
                calibration.main()
            report = json.loads((root / "output/report.json").read_text())
            self.assertEqual(report["calibration_samples"], 2)
            self.assertEqual(report["mixed_validation"]["samples"], 2)
            self.assertEqual(report["comparisons"]["calibrated"]["validation"]["samples"], 2)
            self.assertTrue((root / "output/fixed-student-calibrated.pt").exists())


if __name__ == "__main__":
    unittest.main()
